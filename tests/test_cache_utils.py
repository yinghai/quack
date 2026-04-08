# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import subprocess
from pathlib import Path

import quack.cache_utils as cache_utils


def test_preload_runtime_libraries_uses_rtld_global_once(monkeypatch, tmp_path):
    lib_path = tmp_path / "libcute_runtime.so"
    lib_path.write_text("placeholder")

    calls = []

    monkeypatch.setattr(
        cache_utils.cute.runtime,
        "find_runtime_libraries",
        lambda enable_tvm_ffi=False: [str(lib_path), str(tmp_path / "missing.so")],
    )
    monkeypatch.setattr(cache_utils.ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))

    cache_utils._preload_runtime_libraries.cache_clear()
    cache_utils._preload_runtime_libraries()
    cache_utils._preload_runtime_libraries()

    assert calls == [(str(lib_path), cache_utils.ctypes.RTLD_GLOBAL)]


def test_jit_cache_disk_hit_uses_global_load_lock(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "fingerprint"
    cache_dir.mkdir(parents=True)
    o_path = cache_dir / "deadbeef.so"
    o_path.write_bytes(b"o")

    events = []

    class RecordingLock:
        def __init__(self, lock_path, exclusive, timeout=15):
            self.lock_path = Path(lock_path)
            self.exclusive = exclusive

        def __enter__(self):
            events.append(("enter", self.lock_path.name, self.exclusive))
            return self

        def __exit__(self, *exc):
            events.append(("exit", self.lock_path.name, self.exclusive))

    loaded = object()
    export_name = cache_utils._export_func_name("deadbeef")

    def fake_load_module(path, enable_tvm_ffi=True):
        events.append(("load", Path(path).name, enable_tvm_ffi))
        return type("FakeModule", (), {export_name: loaded})()

    monkeypatch.setattr(cache_utils, "CACHE_ENABLED", True)
    monkeypatch.setattr(cache_utils, "get_cache_path", lambda: cache_root)
    monkeypatch.setattr(cache_utils, "_compute_source_fingerprint", lambda: "fingerprint")
    monkeypatch.setattr(cache_utils, "_key_to_hash", lambda key: "deadbeef")
    monkeypatch.setattr(cache_utils, "FileLock", RecordingLock)
    monkeypatch.setattr(
        cache_utils,
        "_preload_runtime_libraries",
        lambda: events.append(("preload", "runtime", True)),
    )
    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", fake_load_module)

    compile_calls = []

    @cache_utils.jit_cache
    def compile_kernel(dtype, n):
        compile_calls.append((dtype, n))
        raise AssertionError("disk hit should not recompile")

    result = compile_kernel("bf16", 8192)

    assert isinstance(result, cache_utils.LoadedKernel)
    assert result._fn is loaded
    assert compile_calls == []
    assert ("enter", "deadbeef.lock", False) in events
    assert ("enter", cache_utils.LOAD_MODULE_LOCK_NAME, True) in events
    assert events.index(("enter", "deadbeef.lock", False)) < events.index(
        ("enter", cache_utils.LOAD_MODULE_LOCK_NAME, True)
    )
    assert events.index(("enter", cache_utils.LOAD_MODULE_LOCK_NAME, True)) < events.index(
        ("load", "deadbeef.so", True)
    )


def test_export_compiled_artifact_links_shared_library(monkeypatch, tmp_path):
    artifact_path = tmp_path / "deadbeef.so"
    export_calls = []
    link_calls = []

    class FakeCompiledFunction:
        def export_to_c(self, object_file_path, function_name):
            export_calls.append((Path(object_file_path).name, function_name))
            Path(object_file_path).write_bytes(b"fake-o")

    def fake_run(cmd, capture_output, text):
        link_calls.append(cmd)
        assert cmd[:3] == ["g++", "-shared", "-o"]
        Path(cmd[3]).write_bytes(b"fake-so")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cache_utils, "_find_shared_lib_linker", lambda: "g++")
    monkeypatch.setattr(cache_utils.subprocess, "run", fake_run)

    cache_utils._export_compiled_artifact(FakeCompiledFunction(), artifact_path, "func")

    assert export_calls == [("deadbeef.link.o", "func")]
    assert link_calls == [
        [
            "g++",
            "-shared",
            "-o",
            str(tmp_path / "deadbeef.tmp.so"),
            str(tmp_path / "deadbeef.link.o"),
        ]
    ]
    assert artifact_path.read_bytes() == b"fake-so"
    assert not (tmp_path / "deadbeef.link.o").exists()
