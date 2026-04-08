# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Persistent shared-library cache for CuTe DSL compiled kernels.

Compiled kernels are exported as shared libraries (.so).
On subsequent runs the .so is loaded via tvm_ffi (~1ms) instead of
re-generating IR + re-JIT'ing (~100ms per kernel).

Controls:
  QUACK_CACHE_ENABLED=0       — disable persistent cache (default: enabled)
  QUACK_CACHE_DIR=path        — override default cache directory
"""

import ctypes
import fcntl
import functools
import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections import namedtuple
from getpass import getuser
from pathlib import Path

import cutlass
import cutlass.cute as cute
import tvm_ffi

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: str | None = os.getenv("QUACK_CACHE_DIR", None)
COMPILE_ONLY: bool = False
DEBUG_CACHE: bool = os.getenv("QUACK_CACHE_DEBUG", "0") == "1"

# Downstream projects can append directories here to include their sources
# in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: list[Path] = []

EXPORT_FUNC_NAME_PREFIX = "func"
LOCK_TIMEOUT = 60
LOAD_MODULE_LOCK_NAME = "load_module.lock"
CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])
_LOADED_CACHE_SHAS: set[str] = set()


def _noop_kernel(*args, **kwargs):
    pass


def _first_call_lock_path(sha: str) -> Path:
    return get_cache_path() / f"first_call_{sha}.lock"


class LoadedKernel:
    """Callable wrapper that keeps the loaded runtime module alive."""

    def __init__(self, module, fn, sha: str):
        self._module = module
        self._fn = fn
        self._sha = sha
        self._first_call_done = False

    def __call__(self, *args, **kwargs):
        _debug_log("loaded_fn_call_begin", fn=type(self._fn).__name__)
        if not self._first_call_done:
            with FileLock(_first_call_lock_path(self._sha), exclusive=True, timeout=LOCK_TIMEOUT):
                out = self._fn(*args, **kwargs)
            self._first_call_done = True
        else:
            out = self._fn(*args, **kwargs)
        _debug_log("loaded_fn_call_end", fn=type(self._fn).__name__)
        return out

    def __getattr__(self, name):
        return getattr(self._fn, name)


class CompiledKernel:
    """Callable wrapper for freshly compiled kernels with debug logging."""

    def __init__(self, fn, sha: str):
        self._fn = fn
        self._sha = sha

    def __call__(self, *args, **kwargs):
        _debug_log("compiled_fn_call_begin", sha=self._sha, fn=type(self._fn).__name__)
        out = self._fn(*args, **kwargs)
        _debug_log("compiled_fn_call_end", sha=self._sha, fn=type(self._fn).__name__)
        return out

    def __getattr__(self, name):
        return getattr(self._fn, name)


def _debug_log(event: str, **fields) -> None:
    if not DEBUG_CACHE:
        return
    rank = os.getenv("RANK", "?")
    local_rank = os.getenv("LOCAL_RANK", "?")
    host = os.uname().nodename
    pid = os.getpid()
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[quack-cache] event={event} host={host} pid={pid} rank={rank} local_rank={local_rank} {payload}",
        flush=True,
    )


def _format_cache_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, tuple):
        return "(" + ",".join(_format_cache_value(v) for v in value) + ")"
    return str(value)


def _summarize_cache_key(fn_name: str, cache_key: tuple) -> str:
    if fn_name == "_compile_rmsnorm_fwd" and len(cache_key) >= 10:
        dtype, out_dtype, res_dtype, weight_dtype, bias_dtype, res_out_dtype, N, has_rstd, has_mean, is_layernorm = cache_key[:10]
        return (
            "rmsnorm_fwd("
            f"N={N},dtype={_format_cache_value(dtype)},out={_format_cache_value(out_dtype)},"
            f"res={_format_cache_value(res_dtype)},w={_format_cache_value(weight_dtype)},"
            f"bias={_format_cache_value(bias_dtype)},res_out={_format_cache_value(res_out_dtype)},"
            f"rstd={has_rstd},mean={has_mean},layernorm={is_layernorm})"
        )
    if fn_name == "_compile_rmsnorm_bwd" and len(cache_key) >= 9:
        N, dtype, dout_dtype, dx_dtype, weight_dtype, has_db_partial, dres_dtype, dres_out_dtype, has_dw_partial = cache_key[:9]
        return (
            "rmsnorm_bwd("
            f"N={N},dtype={_format_cache_value(dtype)},dout={_format_cache_value(dout_dtype)},"
            f"dx={_format_cache_value(dx_dtype)},w={_format_cache_value(weight_dtype)},"
            f"db_partial={has_db_partial},dres={_format_cache_value(dres_dtype)},"
            f"dres_out={_format_cache_value(dres_out_dtype)},dw_partial={has_dw_partial})"
        )
    return repr(cache_key)


def get_cache_path() -> Path:
    if CACHE_DIR is not None:
        cache_dir = Path(CACHE_DIR)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "quack_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@functools.lru_cache(maxsize=1)
def _preload_runtime_libraries() -> None:
    """Load CuTe DSL runtime libs with RTLD_GLOBAL before loading cached objects.

    FlashAttention does the same because upstream cute.runtime.load_module does
    not preload these globally. Reusing that hardening here avoids relying on
    dlopen order when disk-cached kernels are loaded back into a fresh process.
    """
    for lib_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
        if Path(lib_path).exists():
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"quack cache: failed to preload runtime library {lib_path}: {e}")


def _load_module_lock_path() -> Path:
    """Serialize cached-module loads across processes.

    Per-artifact locks prevent concurrent writers, but they do not stop two
    processes from calling cute.runtime.load_module() at the same time for
    different cached variants. The benchmark crash matches that pattern, so we
    take a cache-wide lock around the actual load.
    """
    return get_cache_path() / LOAD_MODULE_LOCK_NAME


def _hash_source_dir(h, root: Path) -> None:
    """Hash all Python sources under *root* into *h*."""
    for src in sorted(root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)


@functools.lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash quack + extra source dirs plus runtime ABI stamps into a fingerprint."""
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())
    _hash_source_dir(h, Path(__file__).resolve().parent)
    for extra_dir in EXTRA_SOURCE_DIRS:
        _hash_source_dir(h, Path(extra_dir).resolve())
    return h.hexdigest()


def _key_to_hash(key: tuple) -> str:
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


def _export_func_name(sha: str) -> str:
    return EXPORT_FUNC_NAME_PREFIX


def _load_cached_module(o_path: Path, export_func_name: str):
    _debug_log(
        "load_module_begin",
        path=o_path,
        export=export_func_name,
    )
    _preload_runtime_libraries()
    with FileLock(_load_module_lock_path(), exclusive=True, timeout=LOCK_TIMEOUT):
        module = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)
    _debug_log(
        "load_module_end",
        path=o_path,
        export=export_func_name,
    )
    return module


@functools.lru_cache(maxsize=1)
def _find_shared_lib_linker() -> str:
    for linker in ("c++", "g++", "clang++"):
        if shutil.which(linker) is not None:
            return linker
    raise RuntimeError(
        "Could not find a C++ linker for QuACK persistent cache export; "
        "tried c++, g++, and clang++"
    )


def _export_compiled_artifact(compiled_fn, artifact_path: Path, export_func_name: str) -> None:
    tmp_o_path = artifact_path.with_suffix(".link.o")
    tmp_so_path = artifact_path.with_suffix(".tmp.so")
    compiled_fn.export_to_c(
        object_file_path=str(tmp_o_path),
        function_name=export_func_name,
    )
    linker = _find_shared_lib_linker()
    link = subprocess.run(
        [linker, "-shared", "-o", str(tmp_so_path), str(tmp_o_path)],
        capture_output=True,
        text=True,
    )
    if link.returncode != 0:
        raise RuntimeError(
            f"{linker} link failed ({link.returncode}): "
            f"{link.stderr.strip() or link.stdout.strip()}"
        )
    os.replace(tmp_so_path, artifact_path)
    try:
        tmp_o_path.unlink()
    except FileNotFoundError:
        pass


def _load_cached_function(module, export_func_name: str):
    return getattr(module, export_func_name)


def _should_fallback_to_compile(sha: str) -> bool:
    """Load at most one distinct cached module from disk per process.

    The shared-library cache path avoids the warm-cache SIGSEGV, but warm runs
    can still fail when the same process loads multiple distinct cached RMSNorm
    variants from disk. Keep the first cached variant as a disk hit and
    recompile later distinct variants in-process, where they remain memoized for
    the rest of the process lifetime.
    """
    return bool(_LOADED_CACHE_SHAS) and sha not in _LOADED_CACHE_SHAS


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory file lock using fcntl.flock with timeout."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "FileLock":
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), flags)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# JIT cache decorator
# ---------------------------------------------------------------------------


def jit_cache(fn):
    """Decorator that caches compiled CuTe DSL kernels in-memory and on disk.

    The decorated function should return a compiled kernel (i.e. call cute.compile).
    The disk cache key is (fn.__qualname__, *args, **sorted_kwargs).
    """
    cache = {}
    hits = 0
    misses = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = args + tuple(sorted(kwargs.items())) if kwargs else args
        key_summary = _summarize_cache_key(fn.__qualname__, cache_key)
        disk_key = (fn.__qualname__,) + cache_key
        sha = _key_to_hash(disk_key)

        # 1. In-memory hit
        if cache_key in cache:
            hits += 1
            return _noop_kernel if COMPILE_ONLY else cache[cache_key]

        # 2. Disk hit
        if CACHE_ENABLED:
            cache_path = get_cache_path() / _compute_source_fingerprint()
            cache_path.mkdir(parents=True, exist_ok=True)
            o_path = cache_path / f"{sha}.so"
            lock_path = cache_path / f"{sha}.lock"
            export_func_name = _export_func_name(sha)
            try:
                with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                    if o_path.exists():
                        if _should_fallback_to_compile(sha):
                            _debug_log(
                                "disk_hit_fallback_compile",
                                sha=sha,
                                path=o_path,
                                loaded=len(_LOADED_CACHE_SHAS),
                                key=key_summary,
                            )
                        else:
                            _debug_log(
                                "disk_hit",
                                sha=sha,
                                path=o_path,
                                fn=fn.__qualname__,
                                key=key_summary,
                            )
                            m = _load_cached_module(o_path, export_func_name)
                            loaded_fn = _load_cached_function(m, export_func_name)
                            loaded = LoadedKernel(m, loaded_fn, sha)
                            cache[cache_key] = loaded
                            _LOADED_CACHE_SHAS.add(sha)
                            hits += 1
                            return _noop_kernel if COMPILE_ONLY else loaded
            except RuntimeError:
                pass

        # 3. Compile
        misses += 1
        if CACHE_ENABLED:
            _debug_log("compile_miss", sha=sha, fn=fn.__qualname__, key=key_summary)
        compiled_fn = fn(*args, **kwargs)
        compiled_entry = CompiledKernel(compiled_fn, sha) if DEBUG_CACHE else compiled_fn

        # 4. Store
        cache[cache_key] = compiled_entry
        if CACHE_ENABLED:
            try:
                with FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT):
                    if not o_path.exists():
                        o_path.parent.mkdir(parents=True, exist_ok=True)
                        _debug_log(
                            "export_begin",
                            sha=sha,
                            path=o_path,
                            fn=fn.__qualname__,
                            key=key_summary,
                        )
                        _export_compiled_artifact(compiled_fn, o_path, export_func_name)
                        _debug_log(
                            "export_end",
                            sha=sha,
                            path=o_path,
                            fn=fn.__qualname__,
                            key=key_summary,
                        )
            except Exception as e:
                print(f"quack cache: export failed for key {sha}: {e}")

        return _noop_kernel if COMPILE_ONLY else compiled_entry

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def cache_info():
        return CacheInfo(hits=hits, misses=misses, maxsize=None, currsize=len(cache))

    wrapper.cache = cache
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper
