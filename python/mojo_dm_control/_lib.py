"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_DM_CONTROL_LIB") or os.path.join(
    ROOT, "dist", "libmojo-dm-control.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "mc_euler2quat": ([F, F, F, I], None),
    "mc_mj_quatprod": ([I, I, I, I], None),
    "mc_mj_quatneg": ([I, I, I], None),
    "mc_mj_quat2vel": ([I, I, I, F], None),
    "mc_mj_quatdiff": ([I, I, I, I], None),
    "mc_euler_to_rmat": ([I, I, I, I, I, I, I], None),
    "mc_axis_rotation": ([I, I, I, I, I], None),
    "mc_rmat_to_euler": ([I, I, I], None),
    "mc_quat_conj": ([I, I, I], None),
    "mc_quat_inv": ([I, I, I], None),
    "mc_quat_mul": ([I, I, I, I], None),
    "mc_quat_rotate": ([I, I, I, I], None),
    "mc_quat_to_axisangle": ([I, I], None),
    "mc_quat_to_mat": ([I, I], None),
    "mc_quat_rotate_scalar": ([F, F, F, F, F, F, F, I], None),
    "mc_mat_to_quat": ([I, I], None),
    "mc_axisangle_to_quat": ([I, I, I, F], None),
    "mc_quat_log": ([I, I, I, F], None),
    "mc_quat_dist": ([I, I, I, I, F], None),
    "mc_rotation_matrix_2d": ([I, I, I], None),
    "mc_tolerance_margin": ([I, I, I, F, F, F, F, I], None),
    "mc_in_bounds": ([I, I, I, F, F], None),
    "mc_bounded_quat_dist": ([I, I, I, I], None),
    "mc_squared_difference_sum": ([I, I, I], F),
    "mc_nullspace_products": ([I, I, I, I, I, I, F], None),
    "mc_sym2dense": ([I, I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_DM_CONTROL_MOJO")
    if override:
        return [override]
    found = shutil.which("mojo")
    if found:
        return [found]
    raise BuildError("mojo not found; set MOJO_DM_CONTROL_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/ported.mojo` into `dist/libmojo-dm-control.so` if it is stale."""
    target = os.path.join(SRC, "ported.mojo")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(target):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        target,
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(f"{' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        build()
        handle = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(handle, name)
            fn.argtypes = argtypes
            fn.restype = restype
        _lib = handle
    return _lib


_FUNCTIONS: dict[str, object] = {}


def fn(name: str):
    """The exported symbol `name`, looked up once instead of per call."""
    f = _FUNCTIONS.get(name)
    if f is None:
        f = getattr(lib(), name)
        _FUNCTIONS[name] = f
    return f


def f64(a) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = np.asarray(a, dtype=np.float64)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def addr(a) -> int:
    """The address of `a`'s first element, as the C ABI wants it.

    The pointer this returns does not keep `a` alive. A kernel that writes into
    the buffer it is given therefore cannot be handed a temporary inline --
    ``fn(name)(addr(buf), addr(buf.copy()), ...)`` frees the copy before the
    kernel runs, and the write lands in freed heap. Bind such buffers to a name
    first so they outlive the call.
    """
    return np.asarray(a).ctypes.data


def rows(a, width: int) -> tuple[np.ndarray, np.ndarray]:
    """`(original, flat)` where `flat` is a `(n, width)` view of `a`.

    This is the only place where a batched NumPy argument is reshaped for the
    kernel; the kernels themselves index rows, not the leading dimensions.
    """
    arr = f64(a)
    return arr, arr.reshape(-1, width)


def main() -> int:
    """`python -m mojo_dm_control._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
