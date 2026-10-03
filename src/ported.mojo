"""Mojo port of the compute-bound parts of google-deepmind/dm_control.

The functions below keep dm_control's names, argument order, branch structure and
arithmetic. Where a dm_control function is batched, its Mojo twin loops over a
flat, C-contiguous buffer whose row layout is exactly the trailing axis of the
NumPy argument; that reshaping lives in the ctypes wrapper, never here.

Every exported entry point is a C ABI wrapper. Buffers cross as `Int` addresses
because `@export` rejects parametric functions. Nothing in this file allocates:
all scratch, including temporaries, is supplied by the caller.

Kernels with a branch-free body run at width `W` over a contiguous array with a
scalar tail, and the ones whose cost is transcendental rather than bandwidth
hand contiguous row ranges to `parallelize` once the batch clears
`PARALLEL_MIN`. Neither changes a result: the SIMD transcendentals are
bit-identical to their scalar forms and the workers only split a range the
serial path would have covered in the same order.
"""

from max.algorithm import parallelize
from std.math import acos, acosh, asin, atan2, atanh, cos, cosh, exp, log, sin, sqrt, tan, tanh
from std.sys import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int, AnyOrigin[mut=True]]

# SIMD lane count. A branch-free kernel over a contiguous array uses width-W
# loads with a scalar tail. A (n, k) batch is interleaved, not strided, so it
# gets no vector body: a width-W load at `stride * row + k` returns consecutive
# components of one row, not component k of W rows.
comptime W = simd_width_of[DType.float64]()

# Rows below this run on the calling thread. `parallelize` costs about 5ms of
# worker launch on this host, so the crossover for these kernels is near 2e5
# rows and everything smaller loses by 2-10x.
comptime PARALLEL_MIN = 1 << 18
comptime PARALLEL_WORKERS = 16

# numpy.pi, spelled as a literal so the rotation constants need no import.
comptime PI = 3.141592653589793
comptime TWO_PI = 2.0 * PI
comptime DEG2RAD = PI / 180.0

# Constants used to determine when a rotation is close to a pole.
comptime POLE_LIMIT = 1.0 - 1e-6
comptime TOL = 1e-10

# Branch order of dm_control.utils.rewards._sigmoids.
comptime SIGMOID_GAUSSIAN = 0
comptime SIGMOID_HYPERBOLIC = 1
comptime SIGMOID_LONG_TAIL = 2
comptime SIGMOID_RECIPROCAL = 3
comptime SIGMOID_COSINE = 4
comptime SIGMOID_LINEAR = 5
comptime SIGMOID_QUADRATIC = 6
comptime SIGMOID_TANH_SQUARED = 7

# Euler ordering codes, matching the keys of _eulermap.
comptime EULER_XYZ = 0
comptime EULER_XYX = 1
comptime EULER_ZYX = 2
comptime EULER_XZY = 3
comptime EULER_YZX = 4
comptime EULER_ZXY = 5
comptime EULER_YXZ = 6


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def _clip(number: Float64, low: Float64, high: Float64) -> Float64:
    if number < low:
        return low
    if number > high:
        return high
    return number


def deg2rad(degrees: Float64) -> Float64:
    return degrees * DEG2RAD


# ============================================================ dm_control/mujoco/math.py


def mju_negQuat(quat: Ptr, res: Ptr, n: Int):
    for b in range(n):
        res[4 * b + 0] = quat[4 * b + 0]
        res[4 * b + 1] = -quat[4 * b + 1]
        res[4 * b + 2] = -quat[4 * b + 2]
        res[4 * b + 3] = -quat[4 * b + 3]


def mju_mulQuat(quat1: Ptr, quat2: Ptr, res: Ptr, n: Int):
    """Hamilton product over `n` rows of 4."""
    for b in range(n):
        var w1 = quat1[4 * b + 0]
        var i1 = quat1[4 * b + 1]
        var j1 = quat1[4 * b + 2]
        var k1 = quat1[4 * b + 3]
        var w2 = quat2[4 * b + 0]
        var i2 = quat2[4 * b + 1]
        var j2 = quat2[4 * b + 2]
        var k2 = quat2[4 * b + 3]
        # |w1 -i1 -j1 -k1|   |w2|   |w1w2 - i1i2 - j1j2 - k1k2|
        # |i1  w1 -k1  j1| . |i2| = |w1i2 + i1w2 + j1k2 - k1j2|
        # |j1  k1  w1 -i1|   |j2|   |w1j2 - i1k2 + j1w2 + k1i2|
        # |k1 -j1  i1  w1|   |k2|   |w1k2 + i1j2 - j1i2 + k1w2|
        res[4 * b + 0] = w1 * w2 - i1 * i2 - j1 * j2 - k1 * k2
        res[4 * b + 1] = w1 * i2 + i1 * w2 + j1 * k2 - k1 * j2
        res[4 * b + 2] = w1 * j2 - i1 * k2 + j1 * w2 + k1 * i2
        res[4 * b + 3] = w1 * k2 + i1 * j2 - j1 * i2 + k1 * w2


def mju_quat2Vel(quat: Ptr, res: Ptr, n: Int, dt: Float64):
    """Angular velocity carried by an orientation difference, over `n` rows of 4."""
    for b in range(n):
        var q0 = quat[4 * b + 0]
        var i0 = quat[4 * b + 1]
        var j0 = quat[4 * b + 2]
        var k0 = quat[4 * b + 3]
        var vn = sqrt(i0 * i0 + j0 * j0 + k0 * k0)
        # A zero vector part is the identity rotation, hence no velocity.
        if vn == 0.0:
            res[3 * b + 0] = 0.0
            res[3 * b + 1] = 0.0
            res[3 * b + 2] = 0.0
        else:
            var angle = 2.0 * atan2(vn, q0)
            if angle > PI:
                # MuJoCo takes the rotation on (-pi, pi], so a quaternion with
                # a negative real part describes the opposite, shorter turn.
                angle -= TWO_PI
            res[3 * b + 0] = angle * (i0 / vn) / dt
            res[3 * b + 1] = angle * (j0 / vn) / dt
            res[3 * b + 2] = angle * (k0 / vn) / dt

def mju_quatdiff(source: Ptr, target: Ptr, res: Ptr, n: Int):
    """mju_mulQuat(mju_negQuat(source), target) in one pass, no scratch."""
    for b in range(n):
        var w1 = source[4 * b + 0]
        var i1 = -source[4 * b + 1]
        var j1 = -source[4 * b + 2]
        var k1 = -source[4 * b + 3]
        var w2 = target[4 * b + 0]
        var i2 = target[4 * b + 1]
        var j2 = target[4 * b + 2]
        var k2 = target[4 * b + 3]
        res[4 * b + 0] = w1 * w2 - i1 * i2 - j1 * j2 - k1 * k2
        res[4 * b + 1] = w1 * i2 + i1 * w2 + j1 * k2 - k1 * j2
        res[4 * b + 2] = w1 * j2 - i1 * k2 + j1 * w2 + k1 * i2
        res[4 * b + 3] = w1 * k2 + i1 * j2 - j1 * i2 + k1 * w2


def euler2quat(ax: Float64, ay: Float64, az: Float64, res: Ptr):
    var r1 = az
    var r2 = ay
    var r3 = ax

    var c1 = cos(deg2rad(r1 / 2))
    var s1 = sin(deg2rad(r1 / 2))
    var c2 = cos(deg2rad(r2 / 2))
    var s2 = sin(deg2rad(r2 / 2))
    var c3 = cos(deg2rad(r3 / 2))
    var s3 = sin(deg2rad(r3 / 2))

    res[0] = c1 * c2 * c3 + s1 * s2 * s3
    res[1] = c1 * c2 * s3 - s1 * s2 * c3
    res[2] = c1 * s2 * c3 + s1 * c2 * s3
    res[3] = s1 * c2 * c3 - c1 * s2 * s3


# ==================================================== dm_control/utils/transformations.py


def _axis_entry(
    rmat: Ptr, o: Int, rs: Int, full: Int, axis: Int, c: Float64, s: Float64
):
    """One matrix of the batch, every entry written so the caller can hand over
    an uninitialised buffer."""
    if axis == 0:
        # rmat[:, 0, 0:3] = [[1, 0, 0]]
        rmat.unsafe_store(o + 0, 1.0)
        rmat.unsafe_store(o + 1, 0.0)
        rmat.unsafe_store(o + 2, 0.0)
        # rmat[:, 1, 0:3] = vstack([zeros(n), ct, -st]).T
        rmat.unsafe_store(o + rs + 0, 0.0)
        rmat.unsafe_store(o + rs + 1, c)
        rmat.unsafe_store(o + rs + 2, -s)
        # rmat[:, 2, 0:3] = vstack([zeros(n), st, ct]).T
        rmat.unsafe_store(o + 2 * rs + 0, 0.0)
        rmat.unsafe_store(o + 2 * rs + 1, s)
        rmat.unsafe_store(o + 2 * rs + 2, c)
    elif axis == 1:
        # rmat[:, 0, 0:3] = vstack([ct, zeros(n), st]).T
        rmat.unsafe_store(o + 0, c)
        rmat.unsafe_store(o + 1, 0.0)
        rmat.unsafe_store(o + 2, s)
        # rmat[:, 1, 0:3] = [[0, 1, 0]]
        rmat.unsafe_store(o + rs + 0, 0.0)
        rmat.unsafe_store(o + rs + 1, 1.0)
        rmat.unsafe_store(o + rs + 2, 0.0)
        # rmat[:, 2, 0:3] = vstack([-st, zeros(n), ct]).T
        rmat.unsafe_store(o + 2 * rs + 0, -s)
        rmat.unsafe_store(o + 2 * rs + 1, 0.0)
        rmat.unsafe_store(o + 2 * rs + 2, c)
    else:
        # rmat[:, 0, 0:3] = vstack([ct, -st, zeros(n)]).T
        rmat.unsafe_store(o + 0, c)
        rmat.unsafe_store(o + 1, -s)
        rmat.unsafe_store(o + 2, 0.0)
        # rmat[:, 1, 0:3] = vstack([st, ct, zeros(n)]).T
        rmat.unsafe_store(o + rs + 0, s)
        rmat.unsafe_store(o + rs + 1, c)
        rmat.unsafe_store(o + rs + 2, 0.0)
        # rmat[:, 2, 0:3] = [[0, 0, 1]]
        rmat.unsafe_store(o + 2 * rs + 0, 0.0)
        rmat.unsafe_store(o + 2 * rs + 1, 0.0)
        rmat.unsafe_store(o + 2 * rs + 2, 1.0)
    if full:
        rmat.unsafe_store(o + 3, 0.0)
        rmat.unsafe_store(o + rs + 3, 0.0)
        rmat.unsafe_store(o + 2 * rs + 3, 0.0)
        rmat.unsafe_store(o + 3 * rs + 0, 0.0)
        rmat.unsafe_store(o + 3 * rs + 1, 0.0)
        rmat.unsafe_store(o + 3 * rs + 2, 0.0)
        rmat.unsafe_store(o + 3 * rs + 3, 1.0)


def _axis_range(
    theta: Ptr, rmat: Ptr, lo: Int, hi: Int, full: Int, axis: Int
):
    var ts = 16 if full else 9
    var rs = 4 if full else 3
    var b = lo
    while b + W <= hi:
        var t = theta.unsafe_load[width=W](b)
        var ct = cos(t)
        var st = sin(t)
        for j in range(W):
            _axis_entry(rmat, ts * (b + j), rs, full, axis, ct[j], st[j])
        b += W
    while b < hi:
        var t = theta.unsafe_load(b)
        _axis_entry(rmat, ts * b, rs, full, axis, cos(t), sin(t))
        b += 1


def axis_rotation(theta: Ptr, rmat: Ptr, n: Int, full: Int, axis: Int):
    """`n` axis-rotation matrices, cos/sin evaluated once per angle."""
    if n < PARALLEL_MIN:
        _axis_range(theta, rmat, 0, n, full, axis)
        return
    var chunk = (n + PARALLEL_WORKERS - 1) // PARALLEL_WORKERS

    def work(i: Int) {imm}:
        var lo = i * chunk
        _axis_range(theta, rmat, lo, min(lo + chunk, n), full, axis)

    parallelize(work, PARALLEL_WORKERS, PARALLEL_WORKERS)


# `euler_to_rmat`: `rot(a0) @ rot(a1) @ rot(a2)`, with the three axis codes as
# arguments so the Euler ordering resolves in the Python wrapper once per call
# rather than in a branch inside the row loop.
def _euler_compose_range(
    euler: Ptr,
    rmat: Ptr,
    lo: Int,
    hi: Int,
    n: Int,
    full: Int,
    a0: Int,
    a1: Int,
    a2: Int,
):
    """`rot(a0) @ rot(a1) @ rot(a2)` for rows `lo..hi`, writing `hi - lo` matrices.

    Upstream builds all three axis rotations as `(n, s, s)` arrays and
    contracts them with two batched products, so one Euler vector costs five
    `(n, s, s)` writes and five `(n, s, s)` reads. Both contractions keep the
    rotations in registers and fill each entry from a dot product, so writing
    each entry once here leaves one write and one read per result element.
    With `s == 4` that also drops the seven products with the homogeneous row
    and column that never affect a rotation matrix.

    `euler` is the three angle columns concatenated -- `theta[:, 0]` then
    `theta[:, 1]` then `theta[:, 2]`, each `n` long -- rather than the caller's
    interleaved `(n, 3)`, so a width-`W` block of a column is a width-`W` block
    of rows and the vector body needs no gather. `rmat` is the packed
    `(n, s * s)` result, which the caller unpacks. `n` is the whole batch, so a
    worker narrows `lo..hi` while its angle columns keep the batch stride; `rmat`
    arrives already advanced to that worker's first matrix.
    """
    var ts = 16 if full else 9
    var rs = 4 if full else 3
    var b = lo
    while b + W <= hi:
        # `euler` holds the whole batch indexed by absolute row, and its three
        # angle columns start at `euler`, `euler + n` and `euler + 2n`.
        var v0 = euler.unsafe_load[width=W](b)
        var v1 = euler.unsafe_load[width=W](n + b)
        var v2 = euler.unsafe_load[width=W](2 * n + b)
        # All six transcendentals for the block at once, then one scalar store
        # per entry, which is what the strided destination needs anyway.
        var c0 = cos(v0)
        var s0 = sin(v0)
        var c1 = cos(v1)
        var s1 = sin(v1)
        var c2 = cos(v2)
        var s2 = sin(v2)
        for j in range(W):
            _euler_compose_entry(rmat, ts * (b + j - lo), rs, full, a0, a1, a2,
                                 c0[j], s0[j], c1[j], s1[j], c2[j], s2[j])
        b += W
    while b < hi:
        var e0 = euler.unsafe_load(b)
        var e1 = euler.unsafe_load(n + b)
        var e2 = euler.unsafe_load(2 * n + b)
        _euler_compose_entry(rmat, ts * (b - lo), rs, full, a0, a1, a2,
                             cos(e0), sin(e0), cos(e1), sin(e1), cos(e2),
                             sin(e2))
        b += 1


def _euler_compose_entry(
    rmat: Ptr,
    o: Int,
    rs: Int,
    full: Int,
    a0: Int,
    a1: Int,
    a2: Int,
    c0: Float64,
    s0: Float64,
    c1: Float64,
    s1: Float64,
    c2: Float64,
    s2: Float64,
):
    """One Euler row: three rotations and their product, each entry written once.

    A right multiplication by a rotation is a signed permutation of the columns,
    so `A @ Rx(t)` is `[Ac0, tc * Ac1 + ts * Ac2, -ts * Ac1 + tc * Ac2]` and
    likewise for `Ry` and `Rz`; the column coefficients below are that identity.
    `c0`/`s0` are the cosine and sine of the angle driving rotation `a0`, and so
    on, so the caller's vector body evaluates all six.
    """
    # R0, kept in registers. Its column j is (r0j, r1j, r2j).
    var r00 = 1.0
    var r01 = 0.0
    var r02 = 0.0
    var r10 = 0.0
    var r11 = 1.0
    var r12 = 0.0
    var r20 = 0.0
    var r21 = 0.0
    var r22 = 1.0
    if a0 == 0:
        r11 = c0
        r12 = -s0
        r21 = s0
        r22 = c0
    elif a0 == 1:
        r00 = c0
        r02 = s0
        r20 = -s0
        r22 = c0
    else:
        r00 = c0
        r01 = -s0
        r10 = s0
        r11 = c0

    # M = R0 @ R1.
    var m00 = 0.0
    var m01 = 0.0
    var m02 = 0.0
    var m10 = 0.0
    var m11 = 0.0
    var m12 = 0.0
    var m20 = 0.0
    var m21 = 0.0
    var m22 = 0.0
    if a1 == 0:
        m00 = r00
        m01 = c1 * r01 + s1 * r02
        m02 = -s1 * r01 + c1 * r02
        m10 = r10
        m11 = c1 * r11 + s1 * r12
        m12 = -s1 * r11 + c1 * r12
        m20 = r20
        m21 = c1 * r21 + s1 * r22
        m22 = -s1 * r21 + c1 * r22
    elif a1 == 1:
        m00 = c1 * r00 - s1 * r02
        m01 = r01
        m02 = s1 * r00 + c1 * r02
        m10 = c1 * r10 - s1 * r12
        m11 = r11
        m12 = s1 * r10 + c1 * r12
        m20 = c1 * r20 - s1 * r22
        m21 = r21
        m22 = s1 * r20 + c1 * r22
    else:
        m00 = c1 * r00 + s1 * r01
        m01 = -s1 * r00 + c1 * r01
        m02 = r02
        m10 = c1 * r10 + s1 * r11
        m11 = -s1 * r10 + c1 * r11
        m12 = r12
        m20 = c1 * r20 + s1 * r21
        m21 = -s1 * r20 + c1 * r21
        m22 = r22

    # R = M @ R2.
    var d00 = 0.0
    var d01 = 0.0
    var d02 = 0.0
    var d10 = 0.0
    var d11 = 0.0
    var d12 = 0.0
    var d20 = 0.0
    var d21 = 0.0
    var d22 = 0.0
    if a2 == 0:
        d00 = m00
        d01 = c2 * m01 + s2 * m02
        d02 = -s2 * m01 + c2 * m02
        d10 = m10
        d11 = c2 * m11 + s2 * m12
        d12 = -s2 * m11 + c2 * m12
        d20 = m20
        d21 = c2 * m21 + s2 * m22
        d22 = -s2 * m21 + c2 * m22
    elif a2 == 1:
        d00 = c2 * m00 - s2 * m02
        d01 = m01
        d02 = s2 * m00 + c2 * m02
        d10 = c2 * m10 - s2 * m12
        d11 = m11
        d12 = s2 * m10 + c2 * m12
        d20 = c2 * m20 - s2 * m22
        d21 = m21
        d22 = s2 * m20 + c2 * m22
    else:
        d00 = c2 * m00 + s2 * m01
        d01 = -s2 * m00 + c2 * m01
        d02 = m02
        d10 = c2 * m10 + s2 * m11
        d11 = -s2 * m10 + c2 * m11
        d12 = m12
        d20 = c2 * m20 + s2 * m21
        d21 = -s2 * m20 + c2 * m21
        d22 = m22

    rmat.unsafe_store(o + 0 * rs + 0, d00)
    rmat.unsafe_store(o + 0 * rs + 1, d01)
    rmat.unsafe_store(o + 0 * rs + 2, d02)
    rmat.unsafe_store(o + 1 * rs + 0, d10)
    rmat.unsafe_store(o + 1 * rs + 1, d11)
    rmat.unsafe_store(o + 1 * rs + 2, d12)
    rmat.unsafe_store(o + 2 * rs + 0, d20)
    rmat.unsafe_store(o + 2 * rs + 1, d21)
    rmat.unsafe_store(o + 2 * rs + 2, d22)
    if full:
        # The homogeneous row and column, whose contributions to the 3x3 block
        # are the zeros and ones of the identity.
        rmat.unsafe_store(o + 0 * rs + 3, 0.0)
        rmat.unsafe_store(o + 1 * rs + 3, 0.0)
        rmat.unsafe_store(o + 2 * rs + 3, 0.0)
        rmat.unsafe_store(o + 3 * rs + 0, 0.0)
        rmat.unsafe_store(o + 3 * rs + 1, 0.0)
        rmat.unsafe_store(o + 3 * rs + 2, 0.0)
        rmat.unsafe_store(o + 3 * rs + 3, 1.0)


def euler_to_rmat(euler: Ptr, rmat: Ptr, n: Int, full: Int, a0: Int, a1: Int, a2: Int):
    """`n` rotation matrices from three contiguous angle columns, in one pass."""
    if n < PARALLEL_MIN:
        _euler_compose_range(euler, rmat, 0, n, n, full, a0, a1, a2)
        return
    var chunk = (n + PARALLEL_WORKERS - 1) // PARALLEL_WORKERS
    var ts = 16 if full else 9

    def work(i: Int) {imm}:
        var lo = i * chunk
        _euler_compose_range(euler, rmat + ts * lo, lo, min(lo + chunk, n), n,
                             full, a0, a1, a2)

    parallelize(work, PARALLEL_WORKERS, PARALLEL_WORKERS)


def _rmat_to_euler_xyz(rmat: Ptr, euler: Ptr):
    # | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
    # | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
    # | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
    if rmat[0 * 3 + 2] > POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = PI / 2
        euler[2] = atan2(rmat[1 * 3 + 0], rmat[1 * 3 + 1])
        return
    if rmat[0 * 3 + 2] < -POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = -PI / 2
        euler[2] = atan2(rmat[1 * 3 + 0], rmat[1 * 3 + 1])
        return
    euler[0] = -atan2(rmat[1 * 3 + 2], rmat[2 * 3 + 2])
    euler[1] = asin(rmat[0 * 3 + 2])
    euler[2] = -atan2(rmat[0 * 3 + 1], rmat[0 * 3 + 0])


def _rmat_to_euler_xyx(rmat: Ptr, euler: Ptr):
    # | r00 r01 r02 |   |  cy      sy*sx1               sy*cx1             |
    # | r10 r11 r12 | = |  sy*sx0  cx0*cx1-cy*sx0*sx1  -cy*cx1*sx0-cx0*sx1 |
    # | r20 r21 r22 |   | -sy*cx0  cx1*sx0+cy*cx0*sx1   cy*cx0*cx1-sx0*sx1 |
    if rmat[0 * 3 + 0] < 1.0:
        if rmat[0 * 3 + 0] > -1.0:
            euler[1] = acos(_clip(rmat[0 * 3 + 0], -1.0, 1.0))
            euler[0] = atan2(rmat[1 * 3 + 0], -rmat[2 * 3 + 0])
            euler[2] = atan2(rmat[0 * 3 + 1], rmat[0 * 3 + 2])
            return
        else:
            # Not a unique solution:  x1_angle - x0_angle = atan2(-r12, r11)
            euler[1] = PI
            euler[0] = -atan2(-rmat[1 * 3 + 2], rmat[1 * 3 + 1])
            euler[2] = 0.0
            return
    else:
        # Not a unique solution:  x1_angle + x0_angle = atan2(-r12, r11)
        euler[1] = 0.0
        euler[0] = -atan2(-rmat[1 * 3 + 2], rmat[1 * 3 + 1])
        euler[2] = 0.0
        return


def _rmat_to_euler_zyx(rmat: Ptr, euler: Ptr):
    if rmat[2 * 3 + 0] > POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = -PI / 2
        euler[2] = atan2(rmat[0 * 3 + 1], rmat[0 * 3 + 2])
        return
    if rmat[2 * 3 + 0] < -POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = PI / 2
        euler[2] = atan2(rmat[0 * 3 + 1], rmat[0 * 3 + 2])
        return
    euler[0] = atan2(rmat[1 * 3 + 0], rmat[0 * 3 + 0])
    euler[1] = -asin(rmat[2 * 3 + 0])
    euler[2] = atan2(rmat[2 * 3 + 1], rmat[2 * 3 + 2])


def _rmat_to_euler_xzy(rmat: Ptr, euler: Ptr):
    if rmat[0 * 3 + 1] > POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = -PI / 2
        euler[2] = atan2(rmat[1 * 3 + 2], rmat[1 * 3 + 0])
        return
    if rmat[0 * 3 + 1] < -POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = PI / 2
        euler[2] = atan2(rmat[1 * 3 + 2], rmat[1 * 3 + 0])
        return
    euler[0] = atan2(rmat[2 * 3 + 1], rmat[1 * 3 + 1])
    euler[1] = -asin(rmat[0 * 3 + 1])
    euler[2] = atan2(rmat[0 * 3 + 2], rmat[0 * 3 + 0])


def _rmat_to_euler_yzx(rmat: Ptr, euler: Ptr):
    if rmat[1 * 3 + 0] > POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = PI / 2
        euler[2] = -atan2(rmat[0 * 3 + 2], rmat[0 * 3 + 1])
        return
    if rmat[1 * 3 + 0] < -POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = -PI / 2
        euler[2] = -atan2(rmat[0 * 3 + 2], rmat[0 * 3 + 1])
        return
    euler[0] = -atan2(rmat[2 * 3 + 0], rmat[0 * 3 + 0])
    euler[1] = asin(rmat[1 * 3 + 0])
    euler[2] = -atan2(rmat[1 * 3 + 2], rmat[1 * 3 + 1])


def _rmat_to_euler_zxy(rmat: Ptr, euler: Ptr):
    if rmat[2 * 3 + 1] > POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = PI / 2
        euler[2] = atan2(rmat[0 * 3 + 2], rmat[0 * 3 + 0])
        return
    if rmat[2 * 3 + 1] < -POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = -PI / 2
        euler[2] = atan2(rmat[0 * 3 + 2], rmat[0 * 3 + 0])
        return
    euler[0] = -atan2(rmat[0 * 3 + 1], rmat[1 * 3 + 1])
    euler[1] = asin(rmat[2 * 3 + 1])
    euler[2] = -atan2(rmat[2 * 3 + 0], rmat[2 * 3 + 2])


def _rmat_to_euler_yxz(rmat: Ptr, euler: Ptr):
    if rmat[1 * 3 + 2] > POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = -PI / 2
        euler[2] = -atan2(rmat[0 * 3 + 1], rmat[0 * 3 + 0])
        return
    if rmat[1 * 3 + 2] < -POLE_LIMIT:
        euler[0] = 0.0
        euler[1] = PI / 2
        euler[2] = -atan2(rmat[0 * 3 + 1], rmat[0 * 3 + 0])
        return
    euler[0] = atan2(rmat[0 * 3 + 2], rmat[2 * 3 + 2])
    euler[1] = -asin(rmat[1 * 3 + 2])
    euler[2] = atan2(rmat[1 * 3 + 0], rmat[1 * 3 + 1])


def rmat_to_euler(rmat: Ptr, euler: Ptr, ordering: Int):
    if ordering == EULER_XYZ:
        _rmat_to_euler_xyz(rmat, euler)
    elif ordering == EULER_XYX:
        _rmat_to_euler_xyx(rmat, euler)
    elif ordering == EULER_ZYX:
        _rmat_to_euler_zyx(rmat, euler)
    elif ordering == EULER_XZY:
        _rmat_to_euler_xzy(rmat, euler)
    elif ordering == EULER_YZX:
        _rmat_to_euler_yzx(rmat, euler)
    elif ordering == EULER_ZXY:
        _rmat_to_euler_zxy(rmat, euler)
    else:
        _rmat_to_euler_yxz(rmat, euler)


def quat_conj(quat: Ptr, res: Ptr, n: Int):
    for b in range(n):
        res[4 * b + 0] = quat[4 * b + 0]
        res[4 * b + 1] = -quat[4 * b + 1]
        res[4 * b + 2] = -quat[4 * b + 2]
        res[4 * b + 3] = -quat[4 * b + 3]


def quat_inv(quat: Ptr, res: Ptr, n: Int):
    for b in range(n):
        var q0 = quat[4 * b + 0]
        var q1 = quat[4 * b + 1]
        var q2 = quat[4 * b + 2]
        var q3 = quat[4 * b + 3]
        var s = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
        res[4 * b + 0] = q0 / s
        res[4 * b + 1] = -q1 / s
        res[4 * b + 2] = -q2 / s
        res[4 * b + 3] = -q3 / s


def quat_mul(quat1: Ptr, quat2: Ptr, res: Ptr, n: Int):
    mju_mulQuat(quat1, quat2, res, n)


def quat_rotate(quat: Ptr, vec: Ptr, res: Ptr, n: Int):
    """quat_mul(quat_mul(quat, qvec), quat_conj(quat))[1:] with qvec = [0, vec]."""
    for b in range(n):
        var w = quat[4 * b + 0]
        var i = quat[4 * b + 1]
        var j = quat[4 * b + 2]
        var k = quat[4 * b + 3]
        var v0 = 0.0
        var v1 = vec[3 * b + 0]
        var v2 = vec[3 * b + 1]
        var v3 = vec[3 * b + 2]
        # inner = quat * qvec
        var a0 = w * v0 - i * v1 - j * v2 - k * v3
        var a1 = w * v1 + i * v0 + j * v3 - k * v2
        var a2 = w * v2 - i * v3 + j * v0 + k * v1
        var a3 = w * v3 + i * v2 - j * v1 + k * v0
        # res = (inner * conj(quat))[1:]
        var cw = w
        var ci = -i
        var cj = -j
        var ck = -k
        res[3 * b + 0] = a1 * cw + a0 * ci + a2 * ck - a3 * cj
        res[3 * b + 1] = a2 * cw - a1 * ck + a0 * cj + a3 * ci
        res[3 * b + 2] = a3 * cw + a1 * cj - a2 * ci + a0 * ck


def quat_to_axisangle(quat: Ptr, axisangle: Ptr):
    var angle = 2.0 * acos(_clip(quat[0], -1.0, 1.0))
    if angle < TOL:
        axisangle[0] = 0.0
        axisangle[1] = 0.0
        axisangle[2] = 0.0
        return
    var qn = sin(angle / 2)
    angle = (angle + PI) % TWO_PI - PI
    axisangle[0] = quat[1] / qn * angle
    axisangle[1] = quat[2] / qn * angle
    axisangle[2] = quat[3] / qn * angle


def _identity4(mat: Ptr):
    for i in range(16):
        mat[i] = 0.0
    mat[0 * 4 + 0] = 1.0
    mat[1 * 4 + 1] = 1.0
    mat[2 * 4 + 2] = 1.0
    mat[3 * 4 + 3] = 1.0


def quat_to_mat(quat: Ptr, mat: Ptr):
    var q0 = quat[0]
    var q1 = quat[1]
    var q2 = quat[2]
    var q3 = quat[3]
    var nq = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
    _identity4(mat)
    if nq < TOL:
        return
    var s = sqrt(2.0 / nq)
    q0 *= s
    q1 *= s
    q2 *= s
    q3 *= s
    mat[0 * 4 + 0] = 1.0 - q2 * q2 - q3 * q3
    mat[0 * 4 + 1] = q1 * q2 - q3 * q0
    mat[0 * 4 + 2] = q1 * q3 + q2 * q0
    mat[0 * 4 + 3] = 0.0
    mat[1 * 4 + 0] = q1 * q2 + q3 * q0
    mat[1 * 4 + 1] = 1.0 - q1 * q1 - q3 * q3
    mat[1 * 4 + 2] = q2 * q3 - q1 * q0
    mat[1 * 4 + 3] = 0.0
    mat[2 * 4 + 0] = q1 * q3 - q2 * q0
    mat[2 * 4 + 1] = q2 * q3 + q1 * q0
    mat[2 * 4 + 2] = 1.0 - q1 * q1 - q2 * q2
    mat[2 * 4 + 3] = 0.0


def mat_to_quat(mat: Ptr, quat: Ptr):
    var t = mat[0 * 4 + 0] + mat[1 * 4 + 1] + mat[2 * 4 + 2] + mat[3 * 4 + 3]
    if t > mat[3 * 4 + 3]:
        quat[0] = t
        quat[3] = mat[1 * 4 + 0] - mat[0 * 4 + 1]
        quat[2] = mat[0 * 4 + 2] - mat[2 * 4 + 0]
        quat[1] = mat[2 * 4 + 1] - mat[1 * 4 + 2]
    else:
        var i = 0
        var j = 1
        var k = 2
        if mat[1 * 4 + 1] > mat[0 * 4 + 0]:
            i = 1
            j = 2
            k = 0
        if mat[2 * 4 + 2] > mat[i * 4 + i]:
            i = 2
            j = 0
            k = 1
        t = mat[i * 4 + i] - (mat[j * 4 + j] + mat[k * 4 + k]) + mat[3 * 4 + 3]
        quat[i + 1] = t
        quat[j + 1] = mat[i * 4 + j] + mat[j * 4 + i]
        quat[k + 1] = mat[k * 4 + i] + mat[i * 4 + k]
        quat[0] = mat[k * 4 + j] - mat[j * 4 + k]
    var f = 0.5 / sqrt(t * mat[3 * 4 + 3])
    quat[0] *= f
    quat[1] *= f
    quat[2] *= f
    quat[3] *= f


def axisangle_to_quat(axisangle: Ptr, quat: Ptr, n: Int, tol: Float64):
    for b in range(n):
        var a0 = axisangle[3 * b + 0]
        var a1 = axisangle[3 * b + 1]
        var a2 = axisangle[3 * b + 2]
        var angle = sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        var ax = 1.0
        var ay = 0.0
        var az = 0.0
        if angle > tol:
            ax = a0 / angle
            ay = a1 / angle
            az = a2 / angle
        else:
            angle = 0.0
        var sine = sin(angle / 2)
        var cosine = cos(angle / 2)
        quat[4 * b + 0] = cosine
        quat[4 * b + 1] = ax * sine
        quat[4 * b + 2] = ay * sine
        quat[4 * b + 3] = az * sine


def _quat_log_range(quat: Ptr, res: Ptr, lo: Int, hi: Int, tol: Float64):
    for b in range(lo, hi):
        var q0 = quat[4 * b + 0] + tol
        var q1 = quat[4 * b + 1] + tol
        var q2 = quat[4 * b + 2] + tol
        var q3 = quat[4 * b + 3] + tol
        var q_norm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        var v0 = quat[4 * b + 1] + tol
        var v1 = quat[4 * b + 2] + tol
        var v2 = quat[4 * b + 3] + tol
        var v_norm = sqrt(v0 * v0 + v1 * v1 + v2 * v2)
        # Clip to 2*tol because we subtract it here.
        var scaled = acos(_clip(quat[4 * b + 0] - tol, -1.0, 1.0))
        res[4 * b + 0] = log(q_norm)
        res[4 * b + 1] = quat[4 * b + 1] / v_norm * scaled / q_norm
        res[4 * b + 2] = quat[4 * b + 2] / v_norm * scaled / q_norm
        res[4 * b + 3] = quat[4 * b + 3] / v_norm * scaled / q_norm


def quat_log(quat: Ptr, res: Ptr, n: Int, tol: Float64):
    if n < PARALLEL_MIN:
        _quat_log_range(quat, res, 0, n, tol)
        return
    var chunk = (n + PARALLEL_WORKERS - 1) // PARALLEL_WORKERS

    def work(i: Int) {imm}:
        var lo = i * chunk
        _quat_log_range(quat, res, lo, min(lo + chunk, n), tol)

    parallelize(work, PARALLEL_WORKERS, PARALLEL_WORKERS)


def quat_dist(source: Ptr, target: Ptr, res: Ptr, n: Int, tol: Float64):
    """quat_mul(source, quat_inv(target)), normalized, then the norm of its log."""
    for b in range(n):
        var t0 = target[4 * b + 0]
        var t1 = target[4 * b + 1]
        var t2 = target[4 * b + 2]
        var t3 = target[4 * b + 3]
        var s = t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3
        var i0 = t0 / s
        var i1 = -t1 / s
        var i2 = -t2 / s
        var i3 = -t3 / s
        var w = source[4 * b + 0]
        var i = source[4 * b + 1]
        var j = source[4 * b + 2]
        var k = source[4 * b + 3]
        var p0 = w * i0 - i * i1 - j * i2 - k * i3
        var p1 = w * i1 + i * i0 + j * i3 - k * i2
        var p2 = w * i2 - i * i3 + j * i0 + k * i1
        var p3 = w * i3 + i * i2 - j * i1 + k * i0
        var pn = sqrt(p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3)
        p0 /= pn
        p1 /= pn
        p2 /= pn
        p3 /= pn
        var q0 = p0 + tol
        var q1 = p1 + tol
        var q2 = p2 + tol
        var q3 = p3 + tol
        var q_norm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        var v0 = p1 + tol
        var v1 = p2 + tol
        var v2 = p3 + tol
        var v_norm = sqrt(v0 * v0 + v1 * v1 + v2 * v2)
        var scaled = acos(_clip(p0 - tol, -1.0, 1.0))
        var l0 = log(q_norm)
        var l1 = p1 / v_norm * scaled / q_norm
        var l2 = p2 / v_norm * scaled / q_norm
        var l3 = p3 / v_norm * scaled / q_norm
        res[b] = sqrt(l0 * l0 + l1 * l1 + l2 * l2 + l3 * l3)


def _rotation_2d_range(theta: Ptr, res: Ptr, lo: Int, hi: Int):
    var b = lo
    while b + W <= hi:
        var t = theta.unsafe_load[width=W](b)
        var ct = cos(t)
        var st = sin(t)
        for j in range(W):
            var o = 4 * (b + j)
            res.unsafe_store(o + 0, ct[j])
            res.unsafe_store(o + 1, -st[j])
            res.unsafe_store(o + 2, st[j])
            res.unsafe_store(o + 3, ct[j])
        b += W
    while b < hi:
        var t = theta.unsafe_load(b)
        var ct = cos(t)
        var st = sin(t)
        var o = 4 * b
        res.unsafe_store(o + 0, ct)
        res.unsafe_store(o + 1, -st)
        res.unsafe_store(o + 2, st)
        res.unsafe_store(o + 3, ct)
        b += 1


def rotation_matrix_2d(theta: Ptr, res: Ptr, n: Int):
    if n < PARALLEL_MIN:
        _rotation_2d_range(theta, res, 0, n)
        return
    var chunk = (n + PARALLEL_WORKERS - 1) // PARALLEL_WORKERS

    def work(i: Int) {imm}:
        var lo = i * chunk
        _rotation_2d_range(theta, res, lo, min(lo + chunk, n))

    parallelize(work, PARALLEL_WORKERS, PARALLEL_WORKERS)


# ============================================================ dm_control/utils/rewards.py


def _sigmoid_scale(value_at_1: Float64, sigmoid: Int) -> Float64:
    if sigmoid == SIGMOID_GAUSSIAN:
        return sqrt(-2 * log(value_at_1))
    if sigmoid == SIGMOID_HYPERBOLIC:
        return acosh(1 / value_at_1)
    if sigmoid == SIGMOID_LONG_TAIL:
        return sqrt(1 / value_at_1 - 1)
    if sigmoid == SIGMOID_RECIPROCAL:
        return 1 / value_at_1 - 1
    if sigmoid == SIGMOID_COSINE:
        return acos(2 * value_at_1 - 1) / PI
    if sigmoid == SIGMOID_LINEAR:
        return 1 - value_at_1
    if sigmoid == SIGMOID_QUADRATIC:
        return sqrt(1 - value_at_1)
    return atanh(sqrt(1 - value_at_1))


def tolerance_margin(
    x: Ptr,
    res: Ptr,
    n: Int,
    lower: Float64,
    upper: Float64,
    margin: Float64,
    value_at_1: Float64,
    sigmoid: Int,
):
    """The whole `tolerance` with a positive margin, in one pass.

    Upstream's `d = lower - x if x < lower else x - upper` is exactly
    `max(lower - x, x - upper)`, and that maximum is non-positive precisely
    where `lower <= x <= upper`. Clamping it at zero therefore turns the
    `in_bounds` select into a `max`, and all eight sigmoids are 1 at argument
    zero, so the composed `where(in_bounds, 1.0, ...)` needs no branch either.
    What is left is one read of `x` and one write of `res`, and a sigmoid that
    vectorises.
    """
    comptime V = SIMD[DType.float64, W]
    var scale = _sigmoid_scale(value_at_1, sigmoid)
    var lo = V(lower)
    var hi = V(upper)
    var m = V(margin)
    var s = V(scale)
    var zero = V(0.0)
    var one = V(1.0)
    var i = 0
    while i + W <= n:
        var v = x.unsafe_load[width=W](i)
        var arg = max(max(lo - v, v - hi), zero) / m * s
        res.unsafe_store(i, _sigmoid(arg, one, sigmoid))
        i += W
    while i < n:
        var v = x.unsafe_load(i)
        var arg = max(max(lower - v, v - upper), 0.0) / margin * scale
        res.unsafe_store(i, _sigmoid_scalar(arg, 1.0, sigmoid))
        i += 1


def _sigmoid(
    arg: SIMD[DType.float64, W], one: SIMD[DType.float64, W], sigmoid: Int
) -> SIMD[DType.float64, W]:
    """The sigmoid at a non-negative argument; the `|arg| < 1` gate of the
    cosine, linear and quadratic types is a `min` at 1."""
    var narrowed = min(arg, one)
    if sigmoid == SIGMOID_GAUSSIAN:
        return exp(-0.5 * (arg * arg))
    if sigmoid == SIGMOID_HYPERBOLIC:
        return 1 / cosh(arg)
    if sigmoid == SIGMOID_LONG_TAIL:
        return 1 / (arg * arg + 1)
    if sigmoid == SIGMOID_RECIPROCAL:
        return 1 / (arg + 1)
    if sigmoid == SIGMOID_COSINE:
        return (1 + cos(PI * narrowed)) / 2
    if sigmoid == SIGMOID_LINEAR:
        return 1 - narrowed
    if sigmoid == SIGMOID_QUADRATIC:
        return 1 - narrowed * narrowed
    var t = tanh(arg)
    return 1 - t * t


def _sigmoid_scalar(arg: Float64, one: Float64, sigmoid: Int) -> Float64:
    var narrowed = min(arg, one)
    if sigmoid == SIGMOID_GAUSSIAN:
        return exp(-0.5 * (arg * arg))
    if sigmoid == SIGMOID_HYPERBOLIC:
        return 1 / cosh(arg)
    if sigmoid == SIGMOID_LONG_TAIL:
        return 1 / (arg * arg + 1)
    if sigmoid == SIGMOID_RECIPROCAL:
        return 1 / (arg + 1)
    if sigmoid == SIGMOID_COSINE:
        return (1 + cos(PI * narrowed)) / 2
    if sigmoid == SIGMOID_LINEAR:
        return 1 - narrowed
    if sigmoid == SIGMOID_QUADRATIC:
        return 1 - narrowed * narrowed
    var t = tanh(arg)
    return 1 - t * t


def in_bounds(x: Ptr, res: Ptr, n: Int, lower: Float64, upper: Float64):
    for i in range(n):
        res[i] = 1.0 if lower <= x[i] and x[i] <= upper else 0.0


# ================================ dm_control/locomotion/tasks/reference_pose/rewards.py


def _bounded_quat_dist_range(source: Ptr, target: Ptr, res: Ptr, lo: Int, hi: Int):
    for b in range(lo, hi):
        var s0 = source[4 * b + 0]
        var s1 = source[4 * b + 1]
        var s2 = source[4 * b + 2]
        var s3 = source[4 * b + 3]
        var t0 = target[4 * b + 0]
        var t1 = target[4 * b + 1]
        var t2 = target[4 * b + 2]
        var t3 = target[4 * b + 3]
        var sn = sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        var tn = sqrt(t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3)
        s0 /= sn
        s1 /= sn
        s2 /= sn
        s3 /= sn
        t0 /= tn
        t1 /= tn
        t2 /= tn
        t3 /= tn
        source[4 * b + 0] = s0
        source[4 * b + 1] = s1
        source[4 * b + 2] = s2
        source[4 * b + 3] = s3
        target[4 * b + 0] = t0
        target[4 * b + 1] = t1
        target[4 * b + 2] = t2
        target[4 * b + 3] = t3
        # "Distance" in interval [-1, 1].
        var dist = 2 * (s0 * t0 + s1 * t1 + s2 * t2 + s3 * t3) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        res.unsafe_store(b, 0.5 * acos(min(1.0, dist)))



def bounded_quat_dist(source: Ptr, target: Ptr, res: Ptr, n: Int):
    """Normalises `source` and `target` in place, like upstream's `source /= ...`."""
    if n < PARALLEL_MIN:
        _bounded_quat_dist_range(source, target, res, 0, n)
        return
    var chunk = (n + PARALLEL_WORKERS - 1) // PARALLEL_WORKERS

    def work(i: Int) {imm}:
        var lo = i * chunk
        _bounded_quat_dist_range(source, target, res, lo, min(lo + chunk, n))

    parallelize(work, PARALLEL_WORKERS, PARALLEL_WORKERS)


def squared_difference_sum(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        acc += d * d
    return acc


# ================================================= dm_control/utils/inverse_kinematics.py


def nullspace_products(
    jac: Ptr,
    delta: Ptr,
    hess: Ptr,
    joint_delta: Ptr,
    ndelta: Int,
    nv: Int,
    scale: Float64,
):
    """The three nullspace_method products in one crossing of the boundary:
    `hess = jac.T @ jac`, `joint_delta = jac.T @ delta`, `hess += scale * I`.
    A `scale` of 0 leaves the regulariser off, as upstream's branch does."""
    for i in range(nv):
        var acc = 0.0
        for d in range(ndelta):
            acc += jac[d * nv + i] * delta[d]
        joint_delta.unsafe_store(i, acc)
        for j in range(nv):
            var g = 0.0
            for d in range(ndelta):
                g += jac[d * nv + i] * jac[d * nv + j]
            hess.unsafe_store(i * nv + j, g)
        hess.unsafe_store(i * nv + i, hess.unsafe_load(i * nv + i) + scale)


# ============================================================== dm_control/suite/lqr_solver.py


def sym2dense(m: Ptr, res: Ptr, n: Int, rownnz: IPtr, rowadr: IPtr, colind: IPtr):
    """MuJoCo's mju_sym2dense: expand the symmetric sparse mass matrix."""
    for t in range(n * n):
        res[t] = 0.0
    for r in range(n):
        for k in range(rownnz[r]):
            var e = rowadr[r] + k
            var c = colind[e]
            res[r * n + c] = m[e]
            res[c * n + r] = m[e]


# ===================================================================== C ABI wrappers


@export("mc_euler2quat")
def mc_euler2quat(ax: Float64, ay: Float64, az: Float64, dst: Int) abi("C"):
    euler2quat(ax, ay, az, p(dst))


@export("mc_mj_quatprod")
def mc_mj_quatprod(q: Int, r: Int, dst: Int, n: Int) abi("C"):
    mju_mulQuat(p(q), p(r), p(dst), n)


@export("mc_mj_quatneg")
def mc_mj_quatneg(q: Int, dst: Int, n: Int) abi("C"):
    mju_negQuat(p(q), p(dst), n)


@export("mc_mj_quat2vel")
def mc_mj_quat2vel(q: Int, dst: Int, n: Int, dt: Float64) abi("C"):
    mju_quat2Vel(p(q), p(dst), n, dt)


@export("mc_mj_quatdiff")
def mc_mj_quatdiff(source: Int, target: Int, dst: Int, n: Int) abi("C"):
    mju_quatdiff(p(source), p(target), p(dst), n)


@export("mc_euler_to_rmat")
def mc_euler_to_rmat(
    euler: Int, rmat: Int, n: Int, full: Int, a0: Int, a1: Int, a2: Int
) abi("C"):
    euler_to_rmat(p(euler), p(rmat), n, full, a0, a1, a2)


@export("mc_axis_rotation")
def mc_axis_rotation(theta: Int, rmat: Int, n: Int, full: Int, axis: Int) abi("C"):
    axis_rotation(p(theta), p(rmat), n, full, axis)


@export("mc_rmat_to_euler")
def mc_rmat_to_euler(rmat: Int, euler: Int, ordering: Int) abi("C"):
    rmat_to_euler(p(rmat), p(euler), ordering)


@export("mc_quat_conj")
def mc_quat_conj(quat: Int, dst: Int, n: Int) abi("C"):
    quat_conj(p(quat), p(dst), n)


@export("mc_quat_inv")
def mc_quat_inv(quat: Int, dst: Int, n: Int) abi("C"):
    quat_inv(p(quat), p(dst), n)


@export("mc_quat_mul")
def mc_quat_mul(q1: Int, q2: Int, dst: Int, n: Int) abi("C"):
    quat_mul(p(q1), p(q2), p(dst), n)


@export("mc_quat_rotate")
def mc_quat_rotate(quat: Int, vec: Int, dst: Int, n: Int) abi("C"):
    quat_rotate(p(quat), p(vec), p(dst), n)



@export("mc_quat_rotate_scalar")
def mc_quat_rotate_scalar(
    w: Float64, i: Float64, j: Float64, k: Float64,
    v1: Float64, v2: Float64, v3: Float64, dst: Int
) abi("C"):
    var res = p(dst)
    # inner = quat * qvec, with qvec = [0, v1, v2, v3].
    var a0 = w * 0.0 - i * v1 - j * v2 - k * v3
    var a1 = w * v1 + i * 0.0 + j * v3 - k * v2
    var a2 = w * v2 - i * v3 + j * 0.0 + k * v1
    var a3 = w * v3 + i * v2 - j * v1 + k * 0.0
    # res = (inner * conj(quat))[1:]
    res.unsafe_store(0, a1 * w - a0 * i - a2 * k + a3 * j)
    res.unsafe_store(1, a2 * w + a1 * k - a0 * j - a3 * i)
    res.unsafe_store(2, a3 * w - a1 * j + a2 * i - a0 * k)


@export("mc_quat_to_axisangle")
def mc_quat_to_axisangle(quat: Int, axisangle: Int) abi("C"):
    quat_to_axisangle(p(quat), p(axisangle))


@export("mc_quat_to_mat")
def mc_quat_to_mat(quat: Int, mat: Int) abi("C"):
    quat_to_mat(p(quat), p(mat))


@export("mc_mat_to_quat")
def mc_mat_to_quat(mat: Int, quat: Int) abi("C"):
    mat_to_quat(p(mat), p(quat))


@export("mc_axisangle_to_quat")
def mc_axisangle_to_quat(axisangle: Int, quat: Int, n: Int, tol: Float64) abi("C"):
    axisangle_to_quat(p(axisangle), p(quat), n, tol)


@export("mc_quat_log")
def mc_quat_log(quat: Int, dst: Int, n: Int, tol: Float64) abi("C"):
    quat_log(p(quat), p(dst), n, tol)


@export("mc_quat_dist")
def mc_quat_dist(source: Int, target: Int, dst: Int, n: Int, tol: Float64) abi("C"):
    quat_dist(p(source), p(target), p(dst), n, tol)


@export("mc_rotation_matrix_2d")
def mc_rotation_matrix_2d(theta: Int, res: Int, n: Int) abi("C"):
    rotation_matrix_2d(p(theta), p(res), n)


@export("mc_tolerance_margin")
def mc_tolerance_margin(
    x: Int, dst: Int, n: Int, lower: Float64, upper: Float64,
    margin: Float64, value_at_1: Float64, sigmoid: Int
) abi("C"):
    tolerance_margin(p(x), p(dst), n, lower, upper, margin, value_at_1, sigmoid)


@export("mc_in_bounds")
def mc_in_bounds(x: Int, dst: Int, n: Int, lower: Float64, upper: Float64) abi("C"):
    in_bounds(p(x), p(dst), n, lower, upper)


@export("mc_bounded_quat_dist")
def mc_bounded_quat_dist(source: Int, target: Int, dst: Int, n: Int) abi("C"):
    bounded_quat_dist(p(source), p(target), p(dst), n)


@export("mc_squared_difference_sum")
def mc_squared_difference_sum(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return squared_difference_sum(p(a), p(b), n)


@export("mc_nullspace_products")
def mc_nullspace_products(
    jac: Int, delta: Int, hess: Int, joint_delta: Int,
    ndelta: Int, nv: Int, scale: Float64
) abi("C"):
    nullspace_products(
        p(jac), p(delta), p(hess), p(joint_delta), ndelta, nv, scale
    )


@export("mc_sym2dense")
def mc_sym2dense(
    m: Int, res: Int, n: Int, rownnz: Int, rowadr: Int, colind: Int
) abi("C"):
    sym2dense(p(m), p(res), n, ip(rownnz), ip(rowadr), ip(colind))
