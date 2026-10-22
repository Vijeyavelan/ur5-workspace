"""
UR5 (CB3) kinematics: standard DH forward kinematics and the closed-form
inverse kinematics that returns all 8 arm configurations ("branches").

DH parameters are Universal Robots' published values for the CB3 UR5.
Frames: base = UR controller "base" DH frame (URDF 'base_link_inertia'),
tool = flange (URDF 'tool0'): z out of the flange.

Run this file to self-test (random round trips).
"""
import numpy as np

D1, A2, A3, D4, D5, D6 = 0.089159, -0.42500, -0.39225, 0.10915, 0.09465, 0.0823
DH_D = np.array([D1, 0.0, 0.0, D4, D5, D6])
DH_A = np.array([0.0, A2, A3, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# branch index b = 4*i_shoulder + 2*i_wrist + i_elbow
BRANCH_LABELS = [
    f"shoulder{'AB'[s]}-wrist{'+-'[w]}-elbow{'+-'[e]}"
    for s in range(2) for w in range(2) for e in range(2)
]


def _dh(theta, i):
    """Batch DH transform for joint i. theta: (...,) -> (...,4,4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
    T = np.zeros(theta.shape + (4, 4))
    T[..., 0, 0] = ct
    T[..., 0, 1] = -st * ca
    T[..., 0, 2] = st * sa
    T[..., 0, 3] = DH_A[i] * ct
    T[..., 1, 0] = st
    T[..., 1, 1] = ct * ca
    T[..., 1, 2] = -ct * sa
    T[..., 1, 3] = DH_A[i] * st
    T[..., 2, 1] = sa
    T[..., 2, 2] = ca
    T[..., 2, 3] = DH_D[i]
    T[..., 3, 3] = 1.0
    return T


def fk(q, return_all=False):
    """q: (...,6) -> flange pose (...,4,4) in base frame."""
    q = np.asarray(q, dtype=float)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    frames = []
    for i in range(6):
        T = T @ _dh(q[..., i], i)
        frames.append(T)
    return (T, frames) if return_all else T


def ik(T06, eps=1e-9):
    """
    Closed-form IK. T06: (N,4,4) flange poses in base frame.
    Returns Q (N,8,6) in (-pi, pi], valid (N,8) bool.
    Branch index b = 4*i_shoulder + 2*i_wrist + i_elbow.
    """
    T06 = np.asarray(T06, dtype=float)
    N = T06.shape[0]
    Q = np.full((N, 8, 6), np.nan)
    valid = np.zeros((N, 8), dtype=bool)

    p06 = T06[:, :3, 3]
    p05 = p06 - D6 * T06[:, :3, 2]
    r = np.hypot(p05[:, 0], p05[:, 1])
    ok1 = r > abs(D4) + eps
    psi = np.arctan2(p05[:, 1], p05[:, 0])
    phi = np.arccos(np.clip(D4 / np.maximum(r, eps), -1, 1))

    # inverse of T06 (rotation part transposed)
    R = T06[:, :3, :3]
    Rt = np.transpose(R, (0, 2, 1))
    pinv = -np.einsum("nij,nj->ni", Rt, p06)
    # columns of T60
    X60 = Rt[:, :, 0]
    Y60 = Rt[:, :, 1]

    for i_s, sgn_s in enumerate((1.0, -1.0)):
        th1 = psi + sgn_s * phi + np.pi / 2
        s1, c1 = np.sin(th1), np.cos(th1)
        arg5 = (p06[:, 0] * s1 - p06[:, 1] * c1 - D4) / D6
        ok5 = np.abs(arg5) <= 1 + eps
        acos5 = np.arccos(np.clip(arg5, -1, 1))
        for i_w, sgn_w in enumerate((1.0, -1.0)):
            th5 = sgn_w * acos5
            s5 = np.sin(th5)
            s5safe = np.where(np.abs(s5) < 1e-9, 1e-9, s5)
            th6 = np.arctan2(
                (-X60[:, 1] * s1 + Y60[:, 1] * c1) / s5safe,
                (X60[:, 0] * s1 - Y60[:, 0] * c1) / s5safe,
            )
            T01 = _dh(th1, 0)
            T45 = _dh(th5, 4)
            T56 = _dh(th6, 5)
            T46 = T45 @ T56
            T46inv = np.linalg.inv(T46)
            T14 = np.linalg.inv(T01) @ T06 @ T46inv
            # joints 2-3 form a planar 2-link arm in frame 1's x-y plane:
            # P14_xy = (a2 c2 + a3 c23, a2 s2 + a3 s23)
            P14x, P14y = T14[:, 0, 3], T14[:, 1, 3]
            P14xy = np.hypot(P14x, P14y)
            arg3 = (P14xy ** 2 - A2 ** 2 - A3 ** 2) / (2 * A2 * A3)
            ok3 = np.abs(arg3) <= 1 + eps
            acos3 = np.arccos(np.clip(arg3, -1, 1))
            for i_e, sgn_e in enumerate((1.0, -1.0)):
                th3 = sgn_e * acos3
                th2 = np.arctan2(P14y, P14x) - np.arctan2(
                    A3 * np.sin(th3), A2 + A3 * np.cos(th3)
                )
                T12 = _dh(th2, 1)
                T23 = _dh(th3, 2)
                T34 = np.linalg.inv(T12 @ T23) @ T14
                th4 = np.arctan2(T34[:, 1, 0], T34[:, 0, 0])
                b = 4 * i_s + 2 * i_w + i_e
                q = np.stack([th1, th2, th3, th4, th5, th6], axis=-1)
                Q[:, b, :] = np.angle(np.exp(1j * q))  # wrap to (-pi, pi]
                valid[:, b] = ok1 & ok5 & ok3
    # numerical guard: keep only solutions whose FK reproduces the target
    Tchk = fk(np.nan_to_num(Q))
    perr = np.linalg.norm(Tchk[..., :3, 3] - T06[:, None, :3, 3], axis=-1)
    rerr = np.linalg.norm(Tchk[..., :3, :3] - T06[:, None, :3, :3], axis=(-1, -2))
    valid &= (perr < 1e-6) & (rerr < 1e-6)
    return Q, valid


def singularity_terms(q):
    """
    The UR Jacobian determinant factors as  s3 * s5 * a2*a3 * (a2 c2 + a3 c23 + d5 s234).
    Each factor is one singularity:
      elbow    |sin q3|           -> 0 when the arm is fully stretched/folded
      wrist    |sin q5|           -> 0 when wrist 1 and wrist 3 axes line up
      shoulder |a2 c2+a3 c23+d5 s234| (m) -> 0 when the wrist passes over the base axis
    """
    q = np.asarray(q)
    q2, q3, q4, q5 = q[..., 1], q[..., 2], q[..., 3], q[..., 4]
    elbow = np.abs(np.sin(q3))
    wrist = np.abs(np.sin(q5))
    shoulder = np.abs(A2 * np.cos(q2) + A3 * np.cos(q2 + q3) + D5 * np.sin(q2 + q3 + q4))
    return elbow, wrist, shoulder


def jacobian(q):
    """Geometric Jacobian (6x6) at the flange, base frame. q: (6,)"""
    T, frames = fk(np.asarray(q)[None], return_all=True)
    pe = T[0, :3, 3]
    J = np.zeros((6, 6))
    z, p = np.array([0, 0, 1.0]), np.zeros(3)
    for i in range(6):
        J[:3, i] = np.cross(z, pe - p)
        J[3:, i] = z
        z, p = frames[i][0, :3, 2], frames[i][0, :3, 3]
    return J


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q = rng.uniform(-np.pi, np.pi, size=(2000, 6))
    T = fk(q)
    Q, valid = ik(T)
    # every random config must be recovered as one of the 8 branches
    d = np.abs(np.angle(np.exp(1j * (Q - q[:, None, :]))))
    hit = (np.nan_to_num(d, nan=9).max(-1) < 1e-6) & valid
    print("round-trip recovered:", hit.any(1).mean())
    print("valid branches per pose (mean):", valid.sum(1).mean())
    # check the determinant factorization against the numeric Jacobian
    errs = []
    for qi in q[:200]:
        e, w, s = singularity_terms(qi)
        det_formula = np.sin(qi[2]) * np.sin(qi[4]) * A2 * A3 * (
            A2 * np.cos(qi[1]) + A3 * np.cos(qi[1] + qi[2]) + D5 * np.sin(qi[1] + qi[2] + qi[3]))
        errs.append(abs(abs(np.linalg.det(jacobian(qi))) - abs(det_formula)))
    print("max |det J| formula error:", max(errs))
