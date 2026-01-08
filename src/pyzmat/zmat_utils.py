import numpy as np
from ase import Atoms
import copy


from ase.neighborlist import natural_cutoffs, neighbor_list

    
from numba import njit

# -------------------------
# 3D vector helpers
# -------------------------

@njit(cache=True, fastmath=True)
def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@njit(cache=True, fastmath=True)
def norm3(a):
    return np.sqrt(dot3(a, a))

@njit(cache=True, fastmath=True)
def cross3(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=np.float64)

# -------------------------
# Numba version of form_orthonormal_frame_fast
# -------------------------

@njit(cache=True, fastmath=True)
def form_orthonormal_frame_numba(xyz, j, k, l):
    """
    xyz: (N,3) float64
    j,k,l: int
    Returns:
        tvec1, tvec2, r_kj, r_kl, tcrp, ncrp, norm_rkj, norm_rkl, norm_tcrp
    Notes:
        No Python exceptions here. Caller must pre-validate norms if needed.
    """
    r_j = xyz[j]
    r_k = xyz[k]
    r_l = xyz[l]

    r_kj = r_j - r_k
    r_kl = r_l - r_k

    norm_rkj = norm3(r_kj)
    norm_rkl = norm3(r_kl)

    # Avoid divide-by-zero; caller should precheck.
    tvec1 = r_kj / norm_rkj
    tvec2 = r_kl / norm_rkl

    tcrp = cross3(tvec1, tvec2)
    norm_tcrp = norm3(tcrp)

    tcrp = tcrp / norm_tcrp
    ncrp = cross3(tcrp, tvec1)

    return tvec1, tvec2, r_kj, r_kl, tcrp, ncrp, norm_rkj, norm_rkl, norm_tcrp


# -------------------------
# Numba curvature-tensor kernel
# -------------------------

@njit(cache=True, fastmath=True)
#@njit(cache=True, fastmath=False, error_model="python")
def get_curvature_tensor_kernel_numba(xyz, B, conn, bl, ang, dih, valid_frame):
    """
    xyz: (N,3) float64
    B:   (m,3N) float64  with m = 3N-6
    conn:(N,3) int64 (or int32), conn[i,0]=j, conn[i,1]=k, conn[i,2]=l
    bl:  (N,) float64 bond lengths (only meaningful for i>=2)
    ang: (N,) float64 bond angles in radians (only meaningful for i>=2 or i>=3 depending on your convention)
    dih: (N,) float64 dihedral angles in radians (only meaningful for i>=3)
    valid_frame: (N,) uint8/bool; if False for i, skip (or leave zeros)

    Returns:
        K: (3N,m,m) float64
    """
    N = xyz.shape[0]
    m = 3 * N - 6

    K = np.zeros((3 * N, m, m), dtype=np.float64)

    if N > 2:
        bond_length = bl[2]
        bond_angle = ang[2]
        sin_a = np.sin(bond_angle)
        cos_a = np.cos(bond_angle)

        ref = conn[2, 0]
        if ref == 0:
            K[6, 1, 2] = sin_a
            K[6, 2, 1] = sin_a
            K[6, 2, 2] = -bond_length * cos_a
            K[7, 1, 2] = cos_a
            K[7, 2, 1] = cos_a
            K[7, 2, 2] = -bond_length * sin_a
        elif ref == 1:
            K[6, 1, 2] = sin_a
            K[6, 2, 1] = sin_a
            K[6, 2, 2] = bond_length * cos_a
            K[7, 1, 2] = -cos_a
            K[7, 2, 1] = -cos_a
            K[7, 2, 2] = bond_length * sin_a

    for i in range(3, N):
        if not valid_frame[i]:
            continue

        j = conn[i, 0]
        k = conn[i, 1]
        l = conn[i, 2]

        j3 = 3 * j
        k3 = 3 * k
        l3 = 3 * l

        ij = 3 * i - 6
        ijk = 3 * i - 5
        ijkl = 3 * i - 4

        bond_length = bl[i]
        bond_angle = ang[i]
        dihedral_angle = dih[i]

        sin_a = np.sin(bond_angle)
        cos_a = np.cos(bond_angle)
        sin_d = np.sin(dihedral_angle)
        cos_d = np.cos(dihedral_angle)

        tvec1, tvec2, r_kj, r_kl, tcrp, ncrp, psi_kj, psi_kl, psi_kjxkl = form_orthonormal_frame_numba(xyz, j, k, l)

        eps = 1e-14  # choose sensibly
        if psi_kj <= eps or psi_kl <= eps or psi_kjxkl <= eps:
            valid_frame[i] = 0  # if it's mutable, otherwise just `continue`
            continue

        if psi_kj <= eps:
            raise ZeroDivisionError(i)  # raises with i embedded

        inv_kj = 1.0 / psi_kj
        inv_kl = 1.0 / psi_kl
        inv_kj2 = inv_kj * inv_kj
        inv_kl2 = inv_kl * inv_kl
        inv_cr = 1.0 / psi_kjxkl
        inv_cr2 = inv_cr * inv_cr

        for s in range(m):
            # Ds_kj = (Bj[s] - Bk[s]) component-wise
            Ds_kj0 = B[s, j3 + 0] - B[s, k3 + 0]
            Ds_kj1 = B[s, j3 + 1] - B[s, k3 + 1]
            Ds_kj2 = B[s, j3 + 2] - B[s, k3 + 2]

            Ds_kl0 = B[s, l3 + 0] - B[s, k3 + 0]
            Ds_kl1 = B[s, l3 + 1] - B[s, k3 + 1]
            Ds_kl2 = B[s, l3 + 2] - B[s, k3 + 2]

            Dspsi_kj = tvec1[0] * Ds_kj0 + tvec1[1] * Ds_kj1 + tvec1[2] * Ds_kj2
            Dspsi_kl = tvec2[0] * Ds_kl0 + tvec2[1] * Ds_kl1 + tvec2[2] * Ds_kl2

            # Dstvec1 = psi_kj**(-2) * (Ds_kj * psi_kj - Dspsi_kj * r_kj)
            Dstvec10 = inv_kj2 * (Ds_kj0 * psi_kj - Dspsi_kj * r_kj[0])
            Dstvec11 = inv_kj2 * (Ds_kj1 * psi_kj - Dspsi_kj * r_kj[1])
            Dstvec12 = inv_kj2 * (Ds_kj2 * psi_kj - Dspsi_kj * r_kj[2])

            Dstvec20 = inv_kl2 * (Ds_kl0 * psi_kl - Dspsi_kl * r_kl[0])
            Dstvec21 = inv_kl2 * (Ds_kl1 * psi_kl - Dspsi_kl * r_kl[1])
            Dstvec22 = inv_kl2 * (Ds_kl2 * psi_kl - Dspsi_kl * r_kl[2])

            # cross(Dstvec1, tvec2)
            Ds_c10 = Dstvec11 * tvec2[2] - Dstvec12 * tvec2[1]
            Ds_c11 = Dstvec12 * tvec2[0] - Dstvec10 * tvec2[2]
            Ds_c12 = Dstvec10 * tvec2[1] - Dstvec11 * tvec2[0]

            # cross(tvec1, Dstvec2)
            Ds_c20 = tvec1[1] * Dstvec22 - tvec1[2] * Dstvec21
            Ds_c21 = tvec1[2] * Dstvec20 - tvec1[0] * Dstvec22
            Ds_c22 = tvec1[0] * Dstvec21 - tvec1[1] * Dstvec20

            # u = c1 + c2
            Ds_u0 = Ds_c10 + Ds_c20
            Ds_u1 = Ds_c11 + Ds_c21
            Ds_u2 = Ds_c12 + Ds_c22

            Dspsi_kjxkl = tcrp[0] * Ds_u0 + tcrp[1] * Ds_u1 + tcrp[2] * Ds_u2

            # Dstcrp
            Dstcrp0 = (Ds_u0 - Dspsi_kjxkl * tcrp[0]) * inv_cr
            Dstcrp1 = (Ds_u1 - Dspsi_kjxkl * tcrp[1]) * inv_cr
            Dstcrp2 = (Ds_u2 - Dspsi_kjxkl * tcrp[2]) * inv_cr

            # cross(Dttcrp, tvec1)
            Ds_a0 = Dstcrp1 * tvec1[2] - Dstcrp2 * tvec1[1]
            Ds_a1 = Dstcrp2 * tvec1[0] - Dstcrp0 * tvec1[2]
            Ds_a2 = Dstcrp0 * tvec1[1] - Dstcrp1 * tvec1[0]

            # cross(tcrp, Dttvec1)
            Ds_b0 = tcrp[1] * Dstvec12 - tcrp[2] * Dstvec11
            Ds_b1 = tcrp[2] * Dstvec10 - tcrp[0] * Dstvec12
            Ds_b2 = tcrp[0] * Dstvec11 - tcrp[1] * Dstvec10

            # Dsncrp
            Dsncrp0 = Ds_a0 + Ds_b0
            Dsncrp1 = Ds_a1 + Ds_b1
            Dsncrp2 = Ds_a2 + Ds_b2

            for t in range(s, m):
                # Compute t-side intermediates (Dt_kj, Dt_kl, etc.) similarly.
                # Dt_kj = (Bj[t] - Bk[t]) component-wise
                Dt_kj0 = B[t, j3 + 0] - B[t, k3 + 0]
                Dt_kj1 = B[t, j3 + 1] - B[t, k3 + 1]
                Dt_kj2 = B[t, j3 + 2] - B[t, k3 + 2]

                Dt_kl0 = B[t, l3 + 0] - B[t, k3 + 0]
                Dt_kl1 = B[t, l3 + 1] - B[t, k3 + 1]
                Dt_kl2 = B[t, l3 + 2] - B[t, k3 + 2]

                Dtpsi_kj = tvec1[0] * Dt_kj0 + tvec1[1] * Dt_kj1 + tvec1[2] * Dt_kj2
                Dtpsi_kl = tvec2[0] * Dt_kl0 + tvec2[1] * Dt_kl1 + tvec2[2] * Dt_kl2
                
                # Dttvec1 = psi_kj**(-2) * (Dt_kj * psi_kj - Dtpsi_kj * r_kj)
                Dttvec10 = inv_kj2 * (Dt_kj0 * psi_kj - Dtpsi_kj * r_kj[0])
                Dttvec11 = inv_kj2 * (Dt_kj1 * psi_kj - Dtpsi_kj * r_kj[1])
                Dttvec12 = inv_kj2 * (Dt_kj2 * psi_kj - Dtpsi_kj * r_kj[2])

                # Dttvec2 = psi_kl**(-2) * (Dt_kl * psi_kl - Dtpsi_kl * r_kl)
                Dttvec20 = inv_kl2 * (Dt_kl0 * psi_kl - Dtpsi_kl * r_kl[0])
                Dttvec21 = inv_kl2 * (Dt_kl1 * psi_kl - Dtpsi_kl * r_kl[1])
                Dttvec22 = inv_kl2 * (Dt_kl2 * psi_kl - Dtpsi_kl * r_kl[2])

                # cross(Dttvec1, tvec2)
                Dt_c10 = Dttvec11 * tvec2[2] - Dttvec12 * tvec2[1]
                Dt_c11 = Dttvec12 * tvec2[0] - Dttvec10 * tvec2[2]
                Dt_c12 = Dttvec10 * tvec2[1] - Dttvec11 * tvec2[0]

                # cross(tvec1, Dttvec2)
                Dt_c20 = tvec1[1] * Dttvec22 - tvec1[2] * Dttvec21
                Dt_c21 = tvec1[2] * Dttvec20 - tvec1[0] * Dttvec22
                Dt_c22 = tvec1[0] * Dttvec21 - tvec1[1] * Dttvec20

                # u = c1 + c2
                Dt_u0 = Dt_c10 + Dt_c20
                Dt_u1 = Dt_c11 + Dt_c21
                Dt_u2 = Dt_c12 + Dt_c22

                Dtpsi_kjxkl = tcrp[0] * Dt_u0 + tcrp[1] * Dt_u1 + tcrp[2] * Dt_u2

                # Dttcrp
                Dttcrp0 = (Dt_u0 - Dtpsi_kjxkl * tcrp[0]) * inv_cr
                Dttcrp1 = (Dt_u1 - Dtpsi_kjxkl * tcrp[1]) * inv_cr
                Dttcrp2 = (Dt_u2 - Dtpsi_kjxkl * tcrp[2]) * inv_cr

                # cross(Dttcrp, tvec1)
                Dt_a0 = Dttcrp1 * tvec1[2] - Dttcrp2 * tvec1[1]
                Dt_a1 = Dttcrp2 * tvec1[0] - Dttcrp0 * tvec1[2]
                Dt_a2 = Dttcrp0 * tvec1[1] - Dttcrp1 * tvec1[0]

                # cross(tcrp, Dttvec1)
                Dt_b0 = tcrp[1] * Dttvec12 - tcrp[2] * Dttvec11
                Dt_b1 = tcrp[2] * Dttvec10 - tcrp[0] * Dttvec12
                Dt_b2 = tcrp[0] * Dttvec11 - tcrp[1] * Dttvec10

                Dtncrp0 = Dt_a0 + Dt_b0
                Dtncrp1 = Dt_a1 + Dt_b1
                Dtncrp2 = Dt_a2 + Dt_b2

                # DtDs_j = K[j3:j3+3, s, t]
                DtDs_j0 = K[j3 + 0, s, t]
                DtDs_j1 = K[j3 + 1, s, t]
                DtDs_j2 = K[j3 + 2, s, t]

                DtDs_k0 = K[k3 + 0, s, t]
                DtDs_k1 = K[k3 + 1, s, t]
                DtDs_k2 = K[k3 + 2, s, t]

                DtDs_l0 = K[l3 + 0, s, t]
                DtDs_l1 = K[l3 + 1, s, t]
                DtDs_l2 = K[l3 + 2, s, t]

                DtDs_kj0 = DtDs_j0 - DtDs_k0
                DtDs_kj1 = DtDs_j1 - DtDs_k1
                DtDs_kj2 = DtDs_j2 - DtDs_k2

                DtDs_kl0 = DtDs_l0 - DtDs_k0
                DtDs_kl1 = DtDs_l1 - DtDs_k1
                DtDs_kl2 = DtDs_l2 - DtDs_k2

                # DtDspsi_kj = np.dot(Dttvec1, Ds_kj) + np.dot(tvec1, DtDs_kj)
                DtDspsi_kj = (Dttvec10 * Ds_kj0 + Dttvec11 * Ds_kj1 + Dttvec12 * Ds_kj2) + \
                            (tvec1[0] * DtDs_kj0 + tvec1[1] * DtDs_kj1 + tvec1[2] * DtDs_kj2)

                DtDspsi_kl = (Dttvec20 * Ds_kl0 + Dttvec21 * Ds_kl1 + Dttvec22 * Ds_kl2) + \
                            (tvec2[0] * DtDs_kl0 + tvec2[1] * DtDs_kl1 + tvec2[2] * DtDs_kl2)
                
                # for DtDsvec terms
                # precompute scalars
                psi_kj2 = psi_kj * psi_kj
                two_psi_kj_Dtpsi_kj = 2.0 * psi_kj * Dtpsi_kj
                inv_kj4 = inv_kj2 * inv_kj2  # 1/psi^4

                # A components
                A0 = DtDs_kj0 * psi_kj + Ds_kj0 * Dtpsi_kj - DtDspsi_kj * r_kj[0] - Dspsi_kj * Dt_kj0
                A1 = DtDs_kj1 * psi_kj + Ds_kj1 * Dtpsi_kj - DtDspsi_kj * r_kj[1] - Dspsi_kj * Dt_kj1
                A2 = DtDs_kj2 * psi_kj + Ds_kj2 * Dtpsi_kj - DtDspsi_kj * r_kj[2] - Dspsi_kj * Dt_kj2

                # B components
                B0 = Ds_kj0 * psi_kj - Dspsi_kj * r_kj[0]
                B1 = Ds_kj1 * psi_kj - Dspsi_kj * r_kj[1]
                B2 = Ds_kj2 * psi_kj - Dspsi_kj * r_kj[2]

                DtDstvec10 = (A0 * psi_kj2 - B0 * two_psi_kj_Dtpsi_kj) * inv_kj4
                DtDstvec11 = (A1 * psi_kj2 - B1 * two_psi_kj_Dtpsi_kj) * inv_kj4
                DtDstvec12 = (A2 * psi_kj2 - B2 * two_psi_kj_Dtpsi_kj) * inv_kj4

                psi_kl2 = psi_kl * psi_kl
                two_psi_kl_Dtpsi_kl = 2.0 * psi_kl * Dtpsi_kl
                inv_kl4 = inv_kl2 * inv_kl2

                A0 = DtDs_kl0 * psi_kl + Ds_kl0 * Dtpsi_kl - DtDspsi_kl * r_kl[0] - Dspsi_kl * Dt_kl0
                A1 = DtDs_kl1 * psi_kl + Ds_kl1 * Dtpsi_kl - DtDspsi_kl * r_kl[1] - Dspsi_kl * Dt_kl1
                A2 = DtDs_kl2 * psi_kl + Ds_kl2 * Dtpsi_kl - DtDspsi_kl * r_kl[2] - Dspsi_kl * Dt_kl2

                B0 = Ds_kl0 * psi_kl - Dspsi_kl * r_kl[0]
                B1 = Ds_kl1 * psi_kl - Dspsi_kl * r_kl[1]
                B2 = Ds_kl2 * psi_kl - Dspsi_kl * r_kl[2]

                DtDstvec20 = (A0 * psi_kl2 - B0 * two_psi_kl_Dtpsi_kl) * inv_kl4
                DtDstvec21 = (A1 * psi_kl2 - B1 * two_psi_kl_Dtpsi_kl) * inv_kl4
                DtDstvec22 = (A2 * psi_kl2 - B2 * two_psi_kl_Dtpsi_kl) * inv_kl4

                # DtDspsi_kjxkl
                # cross(Dstvec1, tvec2)
                w10 = Dstvec11 * tvec2[2] - Dstvec12 * tvec2[1]
                w11 = Dstvec12 * tvec2[0] - Dstvec10 * tvec2[2]
                w12 = Dstvec10 * tvec2[1] - Dstvec11 * tvec2[0]

                # cross(tvec1, Dstvec2)
                w20 = tvec1[1] * Dstvec22 - tvec1[2] * Dstvec21
                w21 = tvec1[2] * Dstvec20 - tvec1[0] * Dstvec22
                w22 = tvec1[0] * Dstvec21 - tvec1[1] * Dstvec20

                w0 = w10 + w20
                w1 = w11 + w21
                w2 = w12 + w22

                termA = Dttcrp0 * w0 + Dttcrp1 * w1 + Dttcrp2 * w2
                # v = cross(DtDstvec1, tvec2)
                v0 = DtDstvec11 * tvec2[2] - DtDstvec12 * tvec2[1]
                v1 = DtDstvec12 * tvec2[0] - DtDstvec10 * tvec2[2]
                v2 = DtDstvec10 * tvec2[1] - DtDstvec11 * tvec2[0]

                # + cross(Dstvec1, Dttvec2)
                v0 += Dstvec11 * Dttvec22 - Dstvec12 * Dttvec21
                v1 += Dstvec12 * Dttvec20 - Dstvec10 * Dttvec22
                v2 += Dstvec10 * Dttvec21 - Dstvec11 * Dttvec20

                # + cross(Dttvec1, Dstvec2)
                v0 += Dttvec11 * Dstvec22 - Dttvec12 * Dstvec21
                v1 += Dttvec12 * Dstvec20 - Dttvec10 * Dstvec22
                v2 += Dttvec10 * Dstvec21 - Dttvec11 * Dstvec20

                # + cross(tvec1, DtDstvec2)
                v0 += tvec1[1] * DtDstvec22 - tvec1[2] * DtDstvec21
                v1 += tvec1[2] * DtDstvec20 - tvec1[0] * DtDstvec22
                v2 += tvec1[0] * DtDstvec21 - tvec1[1] * DtDstvec20

                termB = tcrp[0] * v0 + tcrp[1] * v1 + tcrp[2] * v2

                DtDspsi_kjxkl = termA + termB

                # A = v - DtDspsi_kjxkl*tcrp - Dspsi_kjxkl*Dttcrp
                A0 = v0 - DtDspsi_kjxkl * tcrp[0] - Dspsi_kjxkl * Dttcrp0
                A1 = v1 - DtDspsi_kjxkl * tcrp[1] - Dspsi_kjxkl * Dttcrp1
                A2 = v2 - DtDspsi_kjxkl * tcrp[2] - Dspsi_kjxkl * Dttcrp2

                # B = w - Dspsi_kjxkl*tcrp
                B0 = w0 - Dspsi_kjxkl * tcrp[0]
                B1 = w1 - Dspsi_kjxkl * tcrp[1]
                B2 = w2 - Dspsi_kjxkl * tcrp[2]

                # DtDstcrp = ((A*psi - B*Dtpsi) / psi^2)
                # using inv_cr2 = 1/psi^2
                DtDstcrp0 = (A0 * psi_kjxkl - B0 * Dtpsi_kjxkl) * inv_cr2
                DtDstcrp1 = (A1 * psi_kjxkl - B1 * Dtpsi_kjxkl) * inv_cr2
                DtDstcrp2 = (A2 * psi_kjxkl - B2 * Dtpsi_kjxkl) * inv_cr2

                # cross(DtDstcrp, tvec1)
                c10 = DtDstcrp1 * tvec1[2] - DtDstcrp2 * tvec1[1]
                c11 = DtDstcrp2 * tvec1[0] - DtDstcrp0 * tvec1[2]
                c12 = DtDstcrp0 * tvec1[1] - DtDstcrp1 * tvec1[0]

                # cross(Dstcrp, Dttvec1)
                c20 = Dstcrp1 * Dttvec12 - Dstcrp2 * Dttvec11
                c21 = Dstcrp2 * Dttvec10 - Dstcrp0 * Dttvec12
                c22 = Dstcrp0 * Dttvec11 - Dstcrp1 * Dttvec10

                # cross(Dttcrp, Dstvec1)
                c30 = Dttcrp1 * Dstvec12 - Dttcrp2 * Dstvec11
                c31 = Dttcrp2 * Dstvec10 - Dttcrp0 * Dstvec12
                c32 = Dttcrp0 * Dstvec11 - Dttcrp1 * Dstvec10

                # cross(tcrp, DtDstvec1)
                c40 = tcrp[1] * DtDstvec12 - tcrp[2] * DtDstvec11
                c41 = tcrp[2] * DtDstvec10 - tcrp[0] * DtDstvec12
                c42 = tcrp[0] * DtDstvec11 - tcrp[1] * DtDstvec10

                DtDsncrp0 = c10 + c20 + c30 + c40
                DtDsncrp1 = c11 + c21 + c31 + c41
                DtDsncrp2 = c12 + c22 + c32 + c42

                d_ij_t = 1.0 if t == ij else 0.0
                d_ijk_t = 1.0 if t == ijk else 0.0
                d_ijkl_t = 1.0 if t == ijkl else 0.0

                d_ij_s = 1.0 if s == ij else 0.0
                d_ijk_s = 1.0 if s == ijk else 0.0
                d_ijkl_s = 1.0 if s == ijkl else 0.0

                # Read already-computed reference blocks 
                Kj0 = K[j3 + 0, s, t]
                Kj1 = K[j3 + 1, s, t]
                Kj2 = K[j3 + 2, s, t]

                Ki0 = (Kj0
                        + DtDsncrp0 * (bond_length * sin_a * cos_d)
                        + Dsncrp0 * (d_ij_t * sin_a * cos_d
                                    + bond_length * d_ijk_t * cos_a * cos_d
                                    - bond_length * d_ijkl_t * sin_a * sin_d)
                        + Dtncrp0 * (d_ij_s * sin_a * cos_d)
                        + ncrp[0] * (d_ij_s * d_ijk_t * cos_a * cos_d
                                        - d_ij_s * d_ijkl_t * sin_a * sin_d)
                        + (Dtncrp0 * bond_length + ncrp[0] * d_ij_t) 
                        * (d_ijk_s * cos_a * cos_d
                            - d_ijkl_s * sin_a * sin_d)
                        + ncrp[0] * bond_length * (- d_ijk_s * d_ijk_t * sin_a * cos_d
                                                    - d_ijk_s * d_ijkl_t * cos_a * sin_d
                                                    - d_ijkl_s * d_ijk_t * cos_a * sin_d
                                                    - d_ijkl_s * d_ijkl_t * sin_a * cos_d)
                        + DtDstcrp0 * (bond_length * sin_a * sin_d)
                        + Dstcrp0 * (d_ij_t * sin_a * sin_d 
                                        + bond_length * d_ijk_t * cos_a * sin_d 
                                        + bond_length * d_ijkl_t * sin_a * cos_d)
                        + tcrp[0] * (d_ij_s * d_ijk_t * cos_a * sin_d 
                                        + d_ij_s * d_ijkl_t * sin_a * cos_d)
                        + Dttcrp0 * (d_ij_s * sin_a * cos_d)
                        + (Dttcrp0 * bond_length + tcrp[0] * d_ij_t)
                        * (d_ijk_s * cos_a * sin_d 
                            + d_ijkl_s * sin_a * cos_d)
                        + tcrp[0] * bond_length * (- d_ijk_s * d_ijk_t * sin_a * sin_d 
                                                    + d_ijk_s * d_ijkl_t * cos_a * cos_d 
                                                    + d_ijkl_s * d_ijk_t * cos_a * cos_d 
                                                    - d_ijkl_s * d_ijkl_t * sin_a * sin_d)
                        - DtDstvec10 * bond_length * cos_a
                        - Dstvec10 * (d_ij_t * cos_a - bond_length * d_ijk_t * sin_a)
                        - Dttvec10 * (d_ij_s * cos_a - bond_length * d_ijk_s * sin_a)
                        - tvec1[0] * (- d_ij_s * d_ijk_t * sin_a 
                                        - d_ij_t * d_ijk_s * sin_a 
                                        - bond_length * d_ijk_s * d_ijk_t * cos_a)
                        )
                Ki1 = (Kj1
                        + DtDsncrp1 * (bond_length * sin_a * cos_d)
                        + Dsncrp1 * (d_ij_t * sin_a * cos_d
                                    + bond_length * d_ijk_t * cos_a * cos_d
                                    - bond_length * d_ijkl_t * sin_a * sin_d)
                        + Dtncrp1 * (d_ij_s * sin_a * cos_d)
                        + ncrp[1] * (d_ij_s * d_ijk_t * cos_a * cos_d
                                        - d_ij_s * d_ijkl_t * sin_a * sin_d)
                        + (Dtncrp1 * bond_length + ncrp[1] * d_ij_t) 
                        * (d_ijk_s * cos_a * cos_d
                            - d_ijkl_s * sin_a * sin_d)
                        + ncrp[1] * bond_length * (- d_ijk_s * d_ijk_t * sin_a * cos_d
                                                    - d_ijk_s * d_ijkl_t * cos_a * sin_d
                                                    - d_ijkl_s * d_ijk_t * cos_a * sin_d
                                                    - d_ijkl_s * d_ijkl_t * sin_a * cos_d)
                        + DtDstcrp1 * (bond_length * sin_a * sin_d)
                        + Dstcrp1 * (d_ij_t * sin_a * sin_d 
                                        + bond_length * d_ijk_t * cos_a * sin_d 
                                        + bond_length * d_ijkl_t * sin_a * cos_d)
                        + tcrp[1] * (d_ij_s * d_ijk_t * cos_a * sin_d 
                                        + d_ij_s * d_ijkl_t * sin_a * cos_d)
                        + Dttcrp1 * (d_ij_s * sin_a * cos_d)
                        + (Dttcrp1 * bond_length + tcrp[1] * d_ij_t)
                        * (d_ijk_s * cos_a * sin_d 
                            + d_ijkl_s * sin_a * cos_d)
                        + tcrp[1] * bond_length * (- d_ijk_s * d_ijk_t * sin_a * sin_d 
                                                    + d_ijk_s * d_ijkl_t * cos_a * cos_d 
                                                    + d_ijkl_s * d_ijk_t * cos_a * cos_d 
                                                    - d_ijkl_s * d_ijkl_t * sin_a * sin_d)
                        - DtDstvec11 * bond_length * cos_a
                        - Dstvec11 * (d_ij_t * cos_a - bond_length * d_ijk_t * sin_a)
                        - Dttvec11 * (d_ij_s * cos_a - bond_length * d_ijk_s * sin_a)
                        - tvec1[1] * (- d_ij_s * d_ijk_t * sin_a 
                                        - d_ij_t * d_ijk_s * sin_a 
                                        - bond_length * d_ijk_s * d_ijk_t * cos_a)
                        )
                Ki2 = (Kj2
                        + DtDsncrp2 * (bond_length * sin_a * cos_d)
                        + Dsncrp2 * (d_ij_t * sin_a * cos_d
                                    + bond_length * d_ijk_t * cos_a * cos_d
                                    - bond_length * d_ijkl_t * sin_a * sin_d)
                        + Dtncrp2 * (d_ij_s * sin_a * cos_d)
                        + ncrp[2] * (d_ij_s * d_ijk_t * cos_a * cos_d
                                        - d_ij_s * d_ijkl_t * sin_a * sin_d)
                        + (Dtncrp2 * bond_length + ncrp[2] * d_ij_t) 
                        * (d_ijk_s * cos_a * cos_d
                            - d_ijkl_s * sin_a * sin_d)
                        + ncrp[2] * bond_length * (- d_ijk_s * d_ijk_t * sin_a * cos_d
                                                    - d_ijk_s * d_ijkl_t * cos_a * sin_d
                                                    - d_ijkl_s * d_ijk_t * cos_a * sin_d
                                                    - d_ijkl_s * d_ijkl_t * sin_a * cos_d)
                        + DtDstcrp2 * (bond_length * sin_a * sin_d)
                        + Dstcrp2 * (d_ij_t * sin_a * sin_d 
                                        + bond_length * d_ijk_t * cos_a * sin_d 
                                        + bond_length * d_ijkl_t * sin_a * cos_d)
                        + tcrp[2] * (d_ij_s * d_ijk_t * cos_a * sin_d 
                                        + d_ij_s * d_ijkl_t * sin_a * cos_d)
                        + Dttcrp2 * (d_ij_s * sin_a * cos_d)
                        + (Dttcrp2 * bond_length + tcrp[2] * d_ij_t)
                        * (d_ijk_s * cos_a * sin_d 
                            + d_ijkl_s * sin_a * cos_d)
                        + tcrp[2] * bond_length * (- d_ijk_s * d_ijk_t * sin_a * sin_d 
                                                    + d_ijk_s * d_ijkl_t * cos_a * cos_d 
                                                    + d_ijkl_s * d_ijk_t * cos_a * cos_d 
                                                    - d_ijkl_s * d_ijkl_t * sin_a * sin_d)
                        - DtDstvec12 * bond_length * cos_a
                        - Dstvec12 * (d_ij_t * cos_a - bond_length * d_ijk_t * sin_a)
                        - Dttvec12 * (d_ij_s * cos_a - bond_length * d_ijk_s * sin_a)
                        - tvec1[2] * (- d_ij_s * d_ijk_t * sin_a 
                                        - d_ij_t * d_ijk_s * sin_a 
                                        - bond_length * d_ijk_s * d_ijk_t * cos_a)
                        )

                K[3 * i + 0, s, t] = Ki0
                K[3 * i + 1, s, t] = Ki1
                K[3 * i + 2, s, t] = Ki2

                # mirror symmetry
                K[3 * i + 0, t, s] = Ki0
                K[3 * i + 1, t, s] = Ki1
                K[3 * i + 2, t, s] = Ki2

    return K


class ZmatUtils:
    """
    Contains functions concerning the conversion of structures between cartesian and internal coordinates, as well as the first and second derivatives of cartesian coordinates w.r.t. internal coordinates.
    """

    @staticmethod
    def kronecker_delta(i, j):
        """
        The Kronecker δ function returns 1 if its two inputs i and j are equal and 0 otherwise.
        """
        if not isinstance(i, int) or not isinstance(j, int):
            raise TypeError("Inputs to kronecker_delta must be integers.")
        return 1 if i == j else 0

    @staticmethod
    def get_zmat_def(atoms, cutoff_scale = 1.2):
        """
        Generate a possibly sensible zmat definition from Cartesian coordinates. 
        Based on: O. Weser, B. Hein-Janke, R. A. Mata, J. Comput. Chem. 2023, 44(5), 710. https://doi.org/10.1002/jcc.27029
        Very under tested: use at own risk.
    
        Parameters: 
        atoms: ASE Atoms object
        cutoff_scale: (Optional) A scale factor for ASE covalent cutoffs
    
        Returns:
        zmat_def: List of tuples defining the connectivities of a Z-matrix from an unordered* list of Cartesian coordinates, 
        of the form [(i', j', k', l'), ...] where i' is the index of the atom being defined, j' is the index of the bond reference, 
        k' is the index of the angle reference, and l' is the index of the dihedral reference in the Cartesian list.

        *: i.e. when the order of the Cartesian list does not match the order in which the Z-matrix is defined.  
        """
    
        # Function to find the neighbor with the maximum valency
        def max_valent(neigh):
            return max(neigh, key = lambda x: len(conn_graph[x]))
        
        # Get covalent cutoffs for each atom in the list
        cutoffs = natural_cutoffs(atoms)
        cutoffs = [c * cutoff_scale for c in cutoffs]
    
        # Get a list of all bonded neighbors i and j
        i_list, j_list = neighbor_list('ij', atoms, cutoff=cutoffs,self_interaction=False)
        natms = len(atoms)
    
        # Define connectivity graph
        conn_graph = {i: set() for i in range(0,natms)}       # Initialise dictionary of sets
        for i, j in zip(i_list, j_list):
            if (i != j):                                    # Redundant since ASE's neighbor_list avoids self interaction, but just in case
                conn_graph[i].add(j)
                conn_graph[j].add(i)
    
        # The i-th entry in conn_graph is thus the set of atoms j bonded to i
    
        # Find atom closest to centroid that is NOT HYDROGEN
        coords = atoms.get_positions()
        centroid = coords.mean(axis = 0)
        elements = atoms.get_chemical_symbols()
        indices = np.arange(0, len(elements))
        distances = np.linalg.norm(coords - centroid, axis = 1)
    
        zipped = list(zip(elements, indices, distances))
        zipped_sorted = sorted(zipped, key = lambda x: x[2])
        elements_sorted, indices_sorted, distances_sorted = zip(*zipped_sorted)
    
        for i in range(len(distances_sorted)):
            if (elements_sorted[i] != 'H'):
                origin = indices_sorted[i]
                break
        
        # Initialize dictionaries (All indices are based on the input list of atoms)
        # Zmatrix being constructed
        zmatrix = {   
                origin: {'b': None, 'a': None, 'd': None}           
                }
        # Atoms already visited
        visited = {origin}
    
        # Children to the current parent: origin
        parent = {nbr: origin for nbr in conn_graph[origin]}
    
        # Neighbors to the current set of children, EXCLUDING visited sites
        work = {nbr: conn_graph[nbr] - visited for nbr in conn_graph[origin]}
    
        # Main loop
        while work:     # So long as work is not empty, there is work to be done
            new_work = {}
    
            # To expand the Zmatrix frontier, we prioritize neighbor's with high valency
            for i in sorted(work, key = lambda x: len(conn_graph[x]), reverse = True):
    
                if (i in visited):  # If neighbor has been visited then skip
                    continue
                
                # BOND: Current parent
                b = parent[i]         
    
                # Case: b in first three entries of zmatrix 
                # BIT: I think this is relevant to the 'else' case when checking len(zmat)
                keys=list(zmatrix.keys())
                if (b in keys[:3]):
                    if (len(zmatrix) == 1):       # Only 1 atom exist in Zmat, i is the second atom
                        a = None
                        d = None
                        # Only bond; no angle or torsion
    
                    elif (len(zmatrix) == 2):     # Only 2 atoms exist in Zmat, i is the third atom
                        a = max_valent(conn_graph[b] & set(zmatrix.keys()))       
                        d = None
                        # BIT: 
                        # Technically at this point there is only 1 choice for the angle atom
                        # The remaining atom bonded to b and existing in the Zmat
                        # Should not need to check max_valency in this case
    
                    else:   # At least 3 atoms exist in the Zmatrix
                        # Make the angle the parent of the bond (should be the case)
                        # Else, make it the highest valency neighbor-to-b already defined
                        if (parent.get(b) is not None): 
                            a = parent[b]         
                        else:
                            a = max_valent(conn_graph[b] & set(zmatrix.keys()))
    
                        # Make the dihedral the parent of the angle, if already defined and not b/a
                        # Else, make it the highest valency neighbor-to-a or b, that have already been defined but are not b or a
                        if ((parent.get(a) is not None)and(parent[a] not in {b, a})):
                            d = parent[a]
                        else:
                            neighbors_a = (conn_graph[a] & set(zmatrix.keys())) - {b, a}
                            if (neighbors_a):
                                d = max_valent(neighbors_a)   
                            else:
                                d = max_valent((conn_graph[b] & set(zmatrix.keys())) - {b, a})
                else:
                    # fallback
                    a = zmatrix[b]['b']
                    d = zmatrix[b]['a']
    
                # Add to Zmatrix
                zmatrix[i] = {'b': b, 'a': a, 'd': d}
    
                # Update the list of visited atoms
                visited.add(i)
    
                # Expand frontier
                for j in sorted(work[i], key = lambda x: len(conn_graph[x]), reverse = True):
                    if (j not in visited):
                        new_work[j] = conn_graph[j] - visited
                        parent[j] = i
    
            # Update work
            work = new_work
    
        # MAIN LOOP END
    
        # Construct list in insertion order
        zmat_def = []
        for atom, refs in zmatrix.items():
            zmat_def.append((atom, refs['b'], refs['a'], refs['d']))
    
        return zmat_def
    
    @staticmethod
    def atoms_2_zmat_init(atoms, zmat_def):
        """
        Convert an unordered ASE Atoms object (in Cartesian coordinates) into a Z-matrix representation.

        Parameters:
        atoms: ASE Atoms object
        zmat_def: List of tuples defining the connectivities of the Z-matrix from an unordered list of Cartesian coordinates, 
        of the form [(i', j', k', l'), ...] where i' is the index of the atom being defined, j' is the index of the bond reference, 
        k' is the index of the angle reference, and l' is the index of the dihedral reference in the Cartesian list. 

        Returns:
        zmat: List of lists defining the values of the Z-matrix of the form [(symbol, bond_length, bond_angle, dih_angle), ...]
        zmat_conn: List of tuples defining the connectivities of the Z-matrix of the form [(symbol, j, k, l), ...] where j, k, and l are
        the indeces of the bond reference, angle reference, and dihedral reference respectively in the Z-matrix. 
        """

        # 
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object.")
        if not isinstance(zmat_def, list):
            raise TypeError("zmat_def must be a list of tuples.")
        for entry in zmat_def:
            if not (isinstance(entry, tuple) and len(entry) == 4):
                raise ValueError("Each element in zmat_def must be a tuple of four elements (atom_id, bond_id, angle_id, dihedral_id).")
        
        symbols = atoms.get_chemical_symbols()
        if max([t[0] for t in zmat_def]) >= len(atoms) or min([t[0] for t in zmat_def]) < 0:
            raise IndexError("Atom index in zmat_def is out of range of the atoms object.")
        zmat = []
        id_swap = {None: None}
    
        for i, (atom_id, bond_id, angle_id, dihedral_id) in enumerate(zmat_def):
            symbol = symbols[atom_id]
            bond_length = None if bond_id is None else atoms.get_distance(atom_id, bond_id)
            bond_angle = None if angle_id is None else atoms.get_angle(atom_id, bond_id, angle_id)
            dihedral_angle = None if dihedral_id is None else atoms.get_dihedral(atom_id, bond_id, angle_id, dihedral_id)
            zmat.append([symbol, bond_length, bond_angle, dihedral_angle])
            id_swap[atom_id] = i
    
        zmat_conn = []
        for (atom_id, bond_id, angle_id, dihedral_id) in zmat_def:
            symbol = symbols[atom_id]
            bond = id_swap[bond_id] if bond_id is not None else None
            angle = id_swap[angle_id] if angle_id is not None else None
            dihedral = id_swap[dihedral_id] if dihedral_id is not None else None
            zmat_conn.append((symbol, bond, angle, dihedral))
    
        return zmat, zmat_conn

    @staticmethod
    def atoms_2_zmat(atoms, zmat_conn):
        """
        Convert an ASE Atoms object to a Z-matrix, given the connectivities.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object.")
        if not isinstance(zmat_conn, list):
            raise TypeError("zmat_conn must be a list of tuples.")
        if len(zmat_conn) != len(atoms):
            raise ValueError("Length of zmat_conn must match the number of atoms in the atoms object.")
    
        symbols = atoms.get_chemical_symbols()
        zmat = []
        
        for i, conn in enumerate(zmat_conn):
            if not (isinstance(conn, tuple) and len(conn) == 4):
                raise ValueError("Each connectivity in zmat_conn must be a tuple of four elements (symbol, bond, angle, dihedral).")
            bond_ref = conn[1]
            angle_ref = conn[2]
            dihedral_ref = conn[3]
            bond_length = None if bond_ref is None else atoms.get_distance(i, bond_ref)
            bond_angle = None if angle_ref is None else atoms.get_angle(i, bond_ref, angle_ref)
            dihedral_angle = None if dihedral_ref is None else atoms.get_dihedral(i, bond_ref, angle_ref, dihedral_ref)
            zmat.append([symbols[i], bond_length, bond_angle, dihedral_angle])
    
        return zmat

    

    @staticmethod
    def form_orthonormal_frame(xyz, j, k, l):
        """
        Form a set of three orthonormal unit vectors and other useful quantities.
        """
        if not isinstance(xyz, (list, np.ndarray)):
            raise TypeError("xyz must be a list or numpy array of coordinates.")
        if max(j, k, l) >= len(xyz) or min(j, k, l) < 0:
            raise IndexError("Indices j, k, l must be within the range of xyz.")
    
        r_j = xyz[j]
        r_k = xyz[k]
        r_l = xyz[l]
    
        r_kj = r_j - r_k
        r_kl = r_l - r_k
    
        norm_rkj = np.linalg.norm(r_kj)
        norm_rkl = np.linalg.norm(r_kl)
        if norm_rkj == 0:
            raise ValueError("Cannot form orthonormal frame: reference atoms j and k coincide (zero bond length).")
        if norm_rkl == 0:
            raise ValueError("Cannot form orthonormal frame: reference atoms k and l coincide (zero bond length).")
    
        tvec1 = r_kj / norm_rkj
        tvec2 = r_kl / norm_rkl
    
        tcrp = np.cross(tvec1, tvec2)
        norm_tcrp = np.linalg.norm(tcrp)
        if norm_tcrp == 0:
            raise ValueError("Cannot form orthonormal frame: vectors tvec1 and tvec2 are parallel.")
        tcrp /= norm_tcrp
        ncrp = np.cross(tcrp, tvec1)
        
        return r_j, r_k, r_l, tvec1, tvec2, r_kj, r_kl, tcrp, ncrp
    

    @staticmethod
    def form_orthonormal_frame_fast(xyz, j, k, l):
        """
        Form a set of three orthonormal unit vectors and other useful quantities.
        """

        r_j = xyz[j]
        r_k = xyz[k]
        r_l = xyz[l]
    
        r_kj = r_j - r_k
        r_kl = r_l - r_k
    
        norm_rkj = np.linalg.norm(r_kj)
        norm_rkl = np.linalg.norm(r_kl)
        if norm_rkj == 0:
            raise ValueError("Cannot form orthonormal frame: reference atoms j and k coincide (zero bond length).")
        if norm_rkl == 0:
            raise ValueError("Cannot form orthonormal frame: reference atoms k and l coincide (zero bond length).")
    
        tvec1 = r_kj / norm_rkj
        tvec2 = r_kl / norm_rkl
    
        tcrp = np.cross(tvec1, tvec2)
        norm_tcrp = np.linalg.norm(tcrp)
        if norm_tcrp == 0:
            raise ValueError("Cannot form orthonormal frame: vectors tvec1 and tvec2 are parallel.")
        tcrp /= norm_tcrp
        ncrp = np.cross(tcrp, tvec1)
        
        return r_j, r_k, r_l, tvec1, tvec2, r_kj, r_kl, tcrp, ncrp
    



    @staticmethod
    def zmat_2_atoms(zmat, zmat_conn):
        """
        Convert a Z-matrix to an ASE Atoms object.
        """
        if not isinstance(zmat, list) or not all(isinstance(row, (list, tuple)) and len(row) >= 4 for row in zmat):
            raise TypeError("zmat must be a list of tuples/lists with at least 4 elements each.")
        if not isinstance(zmat_conn, list) or not all(isinstance(row, tuple) and len(row) == 4 for row in zmat_conn):
            raise TypeError("zmat_conn must be a list of tuples of four elements each (symbol, bond, angle, dihedral).")
        if len(zmat) != len(zmat_conn):
            raise ValueError("Length of zmat and zmat_conn must be equal.")
        N = len(zmat)
        if N < 1:
            raise ValueError("Z-matrix must contain at least one atom.")
    
        xyz = np.zeros((N, 3))
        symbols = [row[0] for row in zmat]
    
        xyz[0] = [0.0, 0.0, 0.0]
    
        if N > 1:
            if zmat[1][1] is None:
                raise ValueError("Bond length for the second atom is missing.")
            xyz[1] = [zmat[1][1], 0.0, 0.0]
    
        if N > 2:
            if zmat[2][1] is None or zmat[2][2] is None:
                raise ValueError("Bond length or bond angle for the third atom is missing.")
            bond_length = zmat[2][1]
            bond_angle = np.radians(zmat[2][2])
            ref = zmat_conn[2][1]
            if ref == 0:
                x = bond_length * np.cos(bond_angle)
                y = bond_length * np.sin(bond_angle)
                xyz[2] = [x, y, 0.0]
            elif ref == 1:
                x = zmat[1][1] - bond_length * np.cos(bond_angle)
                y = - bond_length * np.sin(bond_angle)
                xyz[2] = [x, y, 0.0]
            else:
                raise ValueError("Invalid connectivity for the third atom; reference must be 0 or 1.")
    
        if N > 3:
            for i in range(3, N):
                if zmat[i][1] is None or zmat[i][2] is None or zmat[i][3] is None:
                    raise ValueError(f"Internal coordinate(s) for atom {i+1} are missing.")
                bond_length = zmat[i][1]
                bond_angle = np.radians(zmat[i][2])
                dihedral_angle = np.radians(zmat[i][3])
                try:
                    _, j, k, l = zmat_conn[i]
                except Exception:
                    raise ValueError(f"Connectivity for atom {i+1} is invalid.")
                r_j, r_k, r_l, tvec1, tvec2, r_kj, r_kl, tcrp, ncrp = ZmatUtils.form_orthonormal_frame(xyz, j, k, l)
                x = bond_length * np.sin(bond_angle) * np.cos(dihedral_angle)
                y = bond_length * np.sin(bond_angle) * np.sin(dihedral_angle)
                z = bond_length * np.cos(bond_angle)
                local_coords = x * ncrp + y * tcrp - z * tvec1
                xyz[i] = r_j + local_coords
    
        atoms = Atoms(symbols = symbols, positions=xyz)
        return atoms

    @staticmethod
    def get_B_matrix(zmat, zmat_conn):
        """
        Calculate the Wilson B-matrix analytically given a Z-matrix and its connectivities.
        """
        if len(zmat) != len(zmat_conn):
            raise ValueError("Length of zmat and zmat_conn must be equal.")
            
        atoms = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
        xyz = atoms.get_positions()
        N = len(atoms)
        if N < 3:
            raise ValueError("At least 3 atoms are required to compute the B-matrix.")
        
        B = np.zeros([3 * N - 6, 3 * N])
        B[0, 3] = 1
    
        bond_length = zmat[2][1]
        bond_angle = np.radians(zmat[2][2])
    
        if zmat_conn[2][1] == 0:
            B[1, 6] = np.cos(bond_angle)
            B[2, 6] = - bond_length * np.sin(bond_angle)
            B[1, 7] = np.sin(bond_angle)
            B[2, 7] = bond_length * np.cos(bond_angle)
        elif zmat_conn[2][1] == 1:
            B[0, 6] = 1
            B[1, 6] = - np.cos(bond_angle)
            B[2, 6] = bond_length * np.sin(bond_angle)
            B[1, 7] = - np.sin(bond_angle)
            B[2, 7] = - bond_length * np.cos(bond_angle)
        else:
            raise ValueError("Invalid connectivity for atom 3 in zmat_conn; expected reference index 0 or 1.")
    
        if N > 3:
            for i in range(3, N):
                try:
                    _, j, k, l = zmat_conn[i]
                except Exception:
                    raise ValueError(f"Invalid connectivity for atom {i+1}.")
                
                bond_length = zmat[i][1]
                bond_angle = np.radians(zmat[i][2])
                dihedral_angle = np.radians(zmat[i][3])
                r_j, r_k, r_l, tvec1, tvec2, r_kj, r_kl, tcrp, ncrp = ZmatUtils.form_orthonormal_frame(xyz, j, k, l)
                psi_kl = np.linalg.norm(r_kl)
                psi_kj = np.linalg.norm(r_kj)
                psi_kjxkl = np.linalg.norm(np.cross(tvec1, tvec2))
                for s in range(3 * N - 6):
                    D_j = np.array([B[s, 3 * j], B[s, 3 * j + 1], B[s, 3 * j + 2]])
                    D_k = np.array([B[s, 3 * k], B[s, 3 * k + 1], B[s, 3 * k + 2]])
                    D_l = np.array([B[s, 3 * l], B[s, 3 * l + 1], B[s, 3 * l + 2]])
                    D_kj = D_j - D_k
                    D_kl = D_l - D_k
                    Dpsi_kj = np.dot(tvec1, D_kj)
                    Dpsi_kl = np.dot(tvec2, D_kl)
                    Dtvec1 = psi_kj ** (-2) * (D_kj * psi_kj - Dpsi_kj * r_kj)
                    Dtvec2 = psi_kl ** (-2) * (D_kl * psi_kl - Dpsi_kl * r_kl)
                    Dpsi_kjxkl = np.dot(tcrp, np.cross(Dtvec1, tvec2) + np.cross(tvec1, Dtvec2))
                    Dtcrp = ((np.cross(Dtvec1, tvec2) + np.cross(tvec1, Dtvec2)) - Dpsi_kjxkl * tcrp) * psi_kjxkl ** (-1)
                    Dncrp = np.cross(Dtcrp, tvec1) + np.cross(tcrp, Dtvec1)
                    for c in range(3):
                        B[s, 3 * i + c] = (
                            B[s, 3 * j + c]
                            + Dncrp[c] * bond_length * np.sin(bond_angle) * np.cos(dihedral_angle)
                            + ncrp[c] * ZmatUtils.kronecker_delta(3 * i - 6, s) * np.sin(bond_angle) * np.cos(dihedral_angle)
                            + ncrp[c] * (bond_length * (ZmatUtils.kronecker_delta(3 * i - 5, s) * np.cos(bond_angle) * np.cos(dihedral_angle) - ZmatUtils.kronecker_delta(3 * i - 4, s) * np.sin(bond_angle) * np.sin(dihedral_angle)))
                            + Dtcrp[c] * bond_length * np.sin(bond_angle) * np.sin(dihedral_angle)
                            + tcrp[c] * ZmatUtils.kronecker_delta(3 * i - 6, s) * np.sin(bond_angle) * np.sin(dihedral_angle)
                            + tcrp[c] * (bond_length * (ZmatUtils.kronecker_delta(3 * i - 5, s) * np.cos(bond_angle) * np.sin(dihedral_angle) + ZmatUtils.kronecker_delta(3 * i - 4, s) * np.sin(bond_angle) * np.cos(dihedral_angle)))
                            - Dtvec1[c] * bond_length * np.cos(bond_angle)
                            - tvec1[c] * (ZmatUtils.kronecker_delta(3 * i - 6, s) * np.cos(bond_angle) - bond_length * ZmatUtils.kronecker_delta(3 * i - 5, s) * np.sin(bond_angle))
                        )
        return B


    @staticmethod
    def get_curvature_tensor(zmat, zmat_conn):
        N = len(zmat)
        K = np.zeros([3 * N, 3 * N - 6, 3 * N - 6])
        atoms = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
        xyz = atoms.get_positions()
        B = ZmatUtils.get_B_matrix(zmat, zmat_conn)
        for i in range(N):
            if i == 0:
                continue
            elif i == 1:
                continue
            elif i == 2:
                bond_length = zmat[2][1]
                bond_angle = np.radians(zmat[2][2])
                if zmat_conn[2][1] == 0:
                    K[6, 1, 2] = -np.sin(bond_angle)
                    K[6, 2, 1] = -np.sin(bond_angle)
                    K[6, 2, 2] = -bond_length * np.cos(bond_angle)
                    K[7, 1, 2] = np.cos(bond_angle)
                    K[7, 2, 1] = np.cos(bond_angle)
                    K[7, 2, 2] = -bond_length * np.sin(bond_angle)
                elif zmat_conn[2][1] == 1:
                    K[6, 1, 2] = np.sin(bond_angle)
                    K[6, 2, 1] = np.sin(bond_angle)
                    K[6, 2, 2] = bond_length * np.cos(bond_angle)
                    K[7, 1, 2] = -np.cos(bond_angle)
                    K[7, 2, 1] = -np.cos(bond_angle)
                    K[7, 2, 2] = bond_length * np.sin(bond_angle)
                else:
                    raise ValueError("Invalid connectivity for atom 3 in zmat_conn; expected reference index 0 or 1.")
            elif N > 3:
                try:
                    _, j, k, l = zmat_conn[i]
                except Exception:
                    raise ValueError(f"Invalid connectivity for atom {i+1}.")
                bond_length = zmat[i][1]
                bond_angle = np.radians(zmat[i][2])
                dihedral_angle = np.radians(zmat[i][3])
                r_j, r_k, r_l, tvec1, tvec2, r_kj, r_kl, tcrp, ncrp = ZmatUtils.form_orthonormal_frame(xyz, j, k, l)
                psi_kl = np.linalg.norm(r_kl)
                psi_kj = np.linalg.norm(r_kj)
                psi_kjxkl = np.linalg.norm(np.cross(tvec1, tvec2))
                for s in range(3 * N - 6):
                    Ds_j = np.array([B[s, 3 * j], B[s, 3 * j + 1], B[s, 3 * j + 2]])
                    Ds_k = np.array([B[s, 3 * k], B[s, 3 * k + 1], B[s, 3 * k + 2]])
                    Ds_l = np.array([B[s, 3 * l], B[s, 3 * l + 1], B[s, 3 * l + 2]])
                    Ds_kj = Ds_j - Ds_k
                    Ds_kl = Ds_l - Ds_k
    
                    Dspsi_kj = np.dot(tvec1, Ds_kj)
                    Dspsi_kl = np.dot(tvec2, Ds_kl)
                    Dstvec1 = psi_kj ** (-2) * (Ds_kj * psi_kj - Dspsi_kj * r_kj)
                    Dstvec2 = psi_kl ** (-2) * (Ds_kl * psi_kl - Dspsi_kl * r_kl)
                    Dspsi_kjxkl = np.dot(tcrp, np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2))
                    Dstcrp = ((np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2)) - Dspsi_kjxkl * tcrp) * psi_kjxkl ** (-1)
                    Dsncrp = np.cross(Dstcrp, tvec1) + np.cross(tcrp, Dstvec1)
                    for t in range(s, 3 * N - 6):
                        Dt_j = np.array([B[t, 3 * j], B[t, 3 * j + 1], B[t, 3 * j + 2]])
                        Dt_k = np.array([B[t, 3 * k], B[t, 3 * k + 1], B[t, 3 * k + 2]])
                        Dt_l = np.array([B[t, 3 * l], B[t, 3 * l + 1], B[t, 3 * l + 2]])
                        Dt_kj = Dt_j - Dt_k
                        Dt_kl = Dt_l - Dt_k
    
                        Dtpsi_kj = np.dot(tvec1, Dt_kj)
                        Dtpsi_kl = np.dot(tvec2, Dt_kl)
                        Dttvec1 = psi_kj ** (-2) * (Dt_kj * psi_kj - Dtpsi_kj * r_kj)
                        Dttvec2 = psi_kl ** (-2) * (Dt_kl * psi_kl - Dtpsi_kl * r_kl)
                        Dtpsi_kjxkl = np.dot(tcrp, np.cross(Dttvec1, tvec2) + np.cross(tvec1, Dttvec2))
                        Dttcrp = ((np.cross(Dttvec1, tvec2) + np.cross(tvec1, Dttvec2)) - Dtpsi_kjxkl * tcrp) * psi_kjxkl ** (-1)
                        Dtncrp = np.cross(Dttcrp, tvec1) + np.cross(tcrp, Dttvec1)
    
                        DtDs_j = np.array([K[3 * j, s, t], K[3 * j + 1, s, t], K[3 * j + 2, s, t]])
                        DtDs_k = np.array([K[3 * k, s, t], K[3 * k + 1, s, t], K[3 * k + 2, s, t]])
                        DtDs_l = np.array([K[3 * l, s, t], K[3 * l + 1, s, t], K[3 * l + 2, s, t]])
    
                        DtDs_kj = DtDs_j - DtDs_k
                        DtDs_kl = DtDs_l - DtDs_k
                        
                        DtDspsi_kj = np.dot(Dttvec1, Ds_kj) + np.dot(tvec1, DtDs_kj)
                        DtDspsi_kl = np.dot(Dttvec2, Ds_kl) + np.dot(tvec2, DtDs_kl)
    
                        DtDstvec1 = (
                            ((DtDs_kj * psi_kj + Ds_kj * Dtpsi_kj - DtDspsi_kj * r_kj - Dspsi_kj * Dt_kj) * psi_kj ** 2
                            - (Ds_kj * psi_kj - Dspsi_kj * r_kj) * (2 * psi_kj * Dtpsi_kj)) * psi_kj ** (-4)
                        )
    
                        DtDstvec2 = (
                            ((DtDs_kl * psi_kl + Ds_kl * Dtpsi_kl - DtDspsi_kl * r_kl - Dspsi_kl * Dt_kl) * psi_kl ** 2
                            - (Ds_kl * psi_kl - Dspsi_kl * r_kl) * (2 * psi_kl * Dtpsi_kl)) * psi_kl ** (-4)
                        )
    
                        DtDspsi_kjxkl = (
                            np.dot(Dttcrp, (np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2)))
                            + np.dot(tcrp, (np.cross(DtDstvec1, tvec2) + np.cross(Dstvec1, Dttvec2) 
                                            + np.cross(Dttvec1, Dstvec2) + np.cross(tvec1, DtDstvec2)))
                        )
    
                        DtDstcrp = (
                            ((np.cross(DtDstvec1, tvec2) + np.cross(Dstvec1,  Dttvec2)
                              + np.cross(Dttvec1,  Dstvec2) + np.cross(tvec1, DtDstvec2)
                              - DtDspsi_kjxkl * tcrp - Dspsi_kjxkl * Dttcrp) * psi_kjxkl
                             - (np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2)
                                - Dspsi_kjxkl * tcrp) * Dtpsi_kjxkl)
                            * psi_kjxkl**(-2)            # exponent −1  (never −2)
                        )
    
                        DtDsncrp = (
                            np.cross(DtDstcrp, tvec1) + np.cross(Dstcrp, Dttvec1) 
                            + np.cross(Dttcrp, Dstvec1) + np.cross(tcrp, DtDstvec1)
                        )
                    
                        for c in range(3):
                            ij = 3 * i - 6
                            ijk = 3 * i - 5
                            ijkl = 3 * i - 4
                            K[3 * i + c, s, t] = (
                                K[3 * j + c, s, t]
                                + DtDsncrp[c] * (bond_length * np.sin(bond_angle) * np.cos(dihedral_angle))
                                + Dsncrp[c] * (ZmatUtils.kronecker_delta(ij, t) * np.sin(bond_angle) * np.cos(dihedral_angle)
                                            + bond_length * ZmatUtils.kronecker_delta(ijk, t) * np.cos(bond_angle) * np.cos(dihedral_angle)
                                            - bond_length * ZmatUtils.kronecker_delta(ijkl, t) * np.sin(bond_angle) * np.sin(dihedral_angle))
                                + Dtncrp[c] * (ZmatUtils.kronecker_delta(ij, s) * np.sin(bond_angle) * np.cos(dihedral_angle))
                                + ncrp[c] * (ZmatUtils.kronecker_delta(ij, s) * ZmatUtils.kronecker_delta(ijk, t) * np.cos(bond_angle) * np.cos(dihedral_angle)
                                             - ZmatUtils.kronecker_delta(ij, s) * ZmatUtils.kronecker_delta(ijkl, t) * np.sin(bond_angle) * np.sin(dihedral_angle))
                                + (Dtncrp[c] * bond_length + ncrp[c] * ZmatUtils.kronecker_delta(ij, t)) 
                                * (ZmatUtils.kronecker_delta(ijk, s) * np.cos(bond_angle) * np.cos(dihedral_angle)
                                   - ZmatUtils.kronecker_delta(ijkl, s) * np.sin(bond_angle) * np.sin(dihedral_angle))
                                + ncrp[c] * bond_length * (- ZmatUtils.kronecker_delta(ijk, s) * ZmatUtils.kronecker_delta(ijk, t) * np.sin(bond_angle) * np.cos(dihedral_angle)
                                                           - ZmatUtils.kronecker_delta(ijk, s) * ZmatUtils.kronecker_delta(ijkl, t) * np.cos(bond_angle) * np.sin(dihedral_angle)
                                                           - ZmatUtils.kronecker_delta(ijkl, s) * ZmatUtils.kronecker_delta(ijk, t) * np.cos(bond_angle) * np.sin(dihedral_angle)
                                                           - ZmatUtils.kronecker_delta(ijkl, s) * ZmatUtils.kronecker_delta(ijkl, t) * np.sin(bond_angle) * np.cos(dihedral_angle))
                                + DtDstcrp[c] * (bond_length * np.sin(bond_angle) * np.sin(dihedral_angle))
                                + Dstcrp[c] * (ZmatUtils.kronecker_delta(ij, t) * np.sin(bond_angle) * np.sin(dihedral_angle) 
                                               + bond_length * ZmatUtils.kronecker_delta(ijk, t) * np.cos(bond_angle) * np.sin(dihedral_angle) 
                                               + bond_length * ZmatUtils.kronecker_delta(ijkl, t) * np.sin(bond_angle) * np.cos(dihedral_angle))
                                + tcrp[c] * (ZmatUtils.kronecker_delta(ij, s) * ZmatUtils.kronecker_delta(ijk, t) * np.cos(bond_angle) * np.sin(dihedral_angle) 
                                             + ZmatUtils.kronecker_delta(ij, s) * ZmatUtils.kronecker_delta(ijkl, t) * np.sin(bond_angle) * np.cos(dihedral_angle))
                                + Dttcrp[c] * (ZmatUtils.kronecker_delta(ij, s) * np.sin(bond_angle) * np.cos(dihedral_angle))
                                + (Dttcrp[c] * bond_length + tcrp[c] * ZmatUtils.kronecker_delta(ij, t))
                                * (ZmatUtils.kronecker_delta(ijk, s) * np.cos(bond_angle) * np.sin(dihedral_angle) 
                                   + ZmatUtils.kronecker_delta(ijkl, s) * np.sin(bond_angle) * np.cos(dihedral_angle))
                                + tcrp[c] * bond_length * (- ZmatUtils.kronecker_delta(ijk, s) * ZmatUtils.kronecker_delta(ijk, t) * np.sin(bond_angle) * np.sin(dihedral_angle) 
                                                           + ZmatUtils.kronecker_delta(ijk, s) * ZmatUtils.kronecker_delta(ijkl, t) * np.cos(bond_angle) * np.cos(dihedral_angle) 
                                                           + ZmatUtils.kronecker_delta(ijkl, s) * ZmatUtils.kronecker_delta(ijk, t) * np.cos(bond_angle) * np.cos(dihedral_angle) 
                                                           - ZmatUtils.kronecker_delta(ijkl, s) * ZmatUtils.kronecker_delta(ijkl, t) * np.sin(bond_angle) * np.sin(dihedral_angle))
                                - DtDstvec1[c] * bond_length * np.cos(bond_angle)
                                - Dstvec1[c] * (ZmatUtils.kronecker_delta(ij, t) * np.cos(bond_angle) - bond_length * ZmatUtils.kronecker_delta(ijk, t) * np.sin(bond_angle))
                                - Dttvec1[c] * (ZmatUtils.kronecker_delta(ij, s) * np.cos(bond_angle) - bond_length * ZmatUtils.kronecker_delta(ijk, s) * np.sin(bond_angle))
                                - tvec1[c] * (- ZmatUtils.kronecker_delta(ij, s) * ZmatUtils.kronecker_delta(ijk, t) * np.sin(bond_angle) 
                                              - ZmatUtils.kronecker_delta(ij, t) * ZmatUtils.kronecker_delta(ijk, s) * np.sin(bond_angle) 
                                              - bond_length * ZmatUtils.kronecker_delta(ijk, s) * ZmatUtils.kronecker_delta(ijk, t) * np.cos(bond_angle))
                            )
    
                            K[3 * i + c, t, s] = K[3 * i + c, s, t]               
        return K

    @staticmethod
    def get_curvature_tensor_fast(zmat, zmat_conn):
        N = len(zmat)
        K = np.zeros([3 * N, 3 * N - 6, 3 * N - 6])
        atoms = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
        xyz = atoms.get_positions()
        B = ZmatUtils.get_B_matrix(zmat, zmat_conn)
        for i in range(N):
            if i == 0:
                continue
            elif i == 1:
                continue
            elif i == 2:
                bond_length = zmat[2][1]
                bond_angle = np.radians(zmat[2][2])

                sin_a = np.sin(bond_angle)
                cos_a = np.cos(bond_angle)
                if zmat_conn[2][1] == 0:
                    K[6, 1, 2] = sin_a
                    K[6, 2, 1] = sin_a
                    K[6, 2, 2] = -bond_length * cos_a
                    K[7, 1, 2] = cos_a
                    K[7, 2, 1] = cos_a
                    K[7, 2, 2] = -bond_length * sin_a
                elif zmat_conn[2][1] == 1:
                    K[6, 1, 2] = sin_a
                    K[6, 2, 1] = sin_a
                    K[6, 2, 2] = bond_length * cos_a
                    K[7, 1, 2] = -cos_a
                    K[7, 2, 1] = -cos_a
                    K[7, 2, 2] = bond_length * sin_a
                else:
                    raise ValueError("Invalid connectivity for atom 3 in zmat_conn; expected reference index 0 or 1.")
            elif N > 3:
                try:
                    _, j, k, l = zmat_conn[i]
                except Exception:
                    raise ValueError(f"Invalid connectivity for atom {i+1}.")
                j3 = 3 * j; k3 = 3 * k; l3 = 3 * l
                ij = 3 * i - 6
                ijk = 3 * i - 5
                ijkl = 3 * i - 4

                bond_length = zmat[i][1]
                bond_angle = np.radians(zmat[i][2])
                dihedral_angle = np.radians(zmat[i][3])

                sin_a = np.sin(bond_angle)
                cos_a = np.cos(bond_angle)
                sin_d = np.sin(dihedral_angle)
                cos_d = np.cos(dihedral_angle)

                r_j, r_k, r_l, tvec1, tvec2, r_kj, r_kl, tcrp, ncrp = ZmatUtils.form_orthonormal_frame_fast(xyz, j, k, l)
                psi_kl = np.linalg.norm(r_kl)
                psi_kj = np.linalg.norm(r_kj)
                psi_kjxkl = np.linalg.norm(np.cross(tvec1, tvec2))

                Bj = B[:, j3:j3 + 3]
                Bk = B[:, k3:k3 + 3]
                Bl = B[:, l3:l3 + 3]

                for s in range(3 * N - 6):

                    Ds_kj = Bj[s] - Bk[s]
                    Ds_kl = Bl[s] - Bk[s]
    
                    Dspsi_kj = np.dot(tvec1, Ds_kj)
                    Dspsi_kl = np.dot(tvec2, Ds_kl)
                    Dstvec1 = psi_kj ** (-2) * (Ds_kj * psi_kj - Dspsi_kj * r_kj)
                    Dstvec2 = psi_kl ** (-2) * (Ds_kl * psi_kl - Dspsi_kl * r_kl)
                    Dspsi_kjxkl = np.dot(tcrp, np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2))
                    Dstcrp = ((np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2)) - Dspsi_kjxkl * tcrp) * psi_kjxkl ** (-1)
                    Dsncrp = np.cross(Dstcrp, tvec1) + np.cross(tcrp, Dstvec1)

                    for t in range(s, 3 * N - 6):
                        
                        Dt_kj = Bj[t] - Bk[t]
                        Dt_kl = Bl[t] - Bk[t]
    
                        Dtpsi_kj = np.dot(tvec1, Dt_kj)
                        Dtpsi_kl = np.dot(tvec2, Dt_kl)
                        Dttvec1 = psi_kj ** (-2) * (Dt_kj * psi_kj - Dtpsi_kj * r_kj)
                        Dttvec2 = psi_kl ** (-2) * (Dt_kl * psi_kl - Dtpsi_kl * r_kl)
                        Dtpsi_kjxkl = np.dot(tcrp, np.cross(Dttvec1, tvec2) + np.cross(tvec1, Dttvec2))
                        Dttcrp = ((np.cross(Dttvec1, tvec2) + np.cross(tvec1, Dttvec2)) - Dtpsi_kjxkl * tcrp) * psi_kjxkl ** (-1)
                        Dtncrp = np.cross(Dttcrp, tvec1) + np.cross(tcrp, Dttvec1)
    
                        DtDs_j = K[j3:j3 + 3, s, t]
                        DtDs_k = K[k3:k3 + 3, s, t]
                        DtDs_l = K[l3:l3 + 3, s, t]
    
                        DtDs_kj = DtDs_j - DtDs_k
                        DtDs_kl = DtDs_l - DtDs_k
                        
                        DtDspsi_kj = np.dot(Dttvec1, Ds_kj) + np.dot(tvec1, DtDs_kj)
                        DtDspsi_kl = np.dot(Dttvec2, Ds_kl) + np.dot(tvec2, DtDs_kl)
    
                        DtDstvec1 = (
                            ((DtDs_kj * psi_kj + Ds_kj * Dtpsi_kj - DtDspsi_kj * r_kj - Dspsi_kj * Dt_kj) * psi_kj ** 2
                            - (Ds_kj * psi_kj - Dspsi_kj * r_kj) * (2 * psi_kj * Dtpsi_kj)) * psi_kj ** (-4)
                        )
    
                        DtDstvec2 = (
                            ((DtDs_kl * psi_kl + Ds_kl * Dtpsi_kl - DtDspsi_kl * r_kl - Dspsi_kl * Dt_kl) * psi_kl ** 2
                            - (Ds_kl * psi_kl - Dspsi_kl * r_kl) * (2 * psi_kl * Dtpsi_kl)) * psi_kl ** (-4)
                        )
    
                        DtDspsi_kjxkl = (
                            np.dot(Dttcrp, (np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2)))
                            + np.dot(tcrp, (np.cross(DtDstvec1, tvec2) + np.cross(Dstvec1, Dttvec2) 
                                            + np.cross(Dttvec1, Dstvec2) + np.cross(tvec1, DtDstvec2)))
                        )
    
                        DtDstcrp = (
                            ((np.cross(DtDstvec1, tvec2) + np.cross(Dstvec1,  Dttvec2)
                              + np.cross(Dttvec1,  Dstvec2) + np.cross(tvec1, DtDstvec2)
                              - DtDspsi_kjxkl * tcrp - Dspsi_kjxkl * Dttcrp) * psi_kjxkl
                             - (np.cross(Dstvec1, tvec2) + np.cross(tvec1, Dstvec2)
                                - Dspsi_kjxkl * tcrp) * Dtpsi_kjxkl)
                            * psi_kjxkl**(-2)            
                        )
    
                        DtDsncrp = (
                            np.cross(DtDstcrp, tvec1) + np.cross(Dstcrp, Dttvec1) 
                            + np.cross(Dttcrp, Dstvec1) + np.cross(tcrp, DtDstvec1)
                        )

                        d_ij_t = ZmatUtils.kronecker_delta(ij, t)
                        d_ijk_t = ZmatUtils.kronecker_delta(ijk, t)
                        d_ijkl_t = ZmatUtils.kronecker_delta(ijkl, t)
                        d_ij_s = ZmatUtils.kronecker_delta(ij, s)
                        d_ijk_s = ZmatUtils.kronecker_delta(ijk, s)
                        d_ijkl_s = ZmatUtils.kronecker_delta(ijkl, s)

                        Ki = K[3 * i:3 * i + 3, s, t]
                        Kj = K[3 * j:3 * j + 3, s, t]

                        Ki[:] = (
                                Kj[:]
                                + DtDsncrp * (bond_length * sin_a * cos_d)
                                + Dsncrp * (d_ij_t * sin_a * cos_d
                                            + bond_length * d_ijk_t * cos_a * cos_d
                                            - bond_length * d_ijkl_t * sin_a * sin_d)
                                + Dtncrp * (d_ij_s * sin_a * cos_d)
                                + ncrp * (d_ij_s * d_ijk_t * cos_a * cos_d
                                             - d_ij_s * d_ijkl_t * sin_a * sin_d)
                                + (Dtncrp * bond_length + ncrp * d_ij_t) 
                                * (d_ijk_s * cos_a * cos_d
                                   - d_ijkl_s * sin_a * sin_d)
                                + ncrp * bond_length * (- d_ijk_s * d_ijk_t * sin_a * cos_d
                                                           - d_ijk_s * d_ijkl_t * cos_a * sin_d
                                                           - d_ijkl_s * d_ijk_t * cos_a * sin_d
                                                           - d_ijkl_s * d_ijkl_t * sin_a * cos_d)
                                + DtDstcrp * (bond_length * sin_a * sin_d)
                                + Dstcrp * (d_ij_t * sin_a * sin_d 
                                               + bond_length * d_ijk_t * cos_a * sin_d 
                                               + bond_length * d_ijkl_t * sin_a * cos_d)
                                + tcrp * (d_ij_s * d_ijk_t * cos_a * sin_d 
                                             + d_ij_s * d_ijkl_t * sin_a * cos_d)
                                + Dttcrp * (d_ij_s * sin_a * cos_d)
                                + (Dttcrp * bond_length + tcrp * d_ij_t)
                                * (d_ijk_s * cos_a * sin_d 
                                   + d_ijkl_s * sin_a * cos_d)
                                + tcrp * bond_length * (- d_ijk_s * d_ijk_t * sin_a * sin_d 
                                                           + d_ijk_s * d_ijkl_t * cos_a * cos_d 
                                                           + d_ijkl_s * d_ijk_t * cos_a * cos_d 
                                                           - d_ijkl_s * d_ijkl_t * sin_a * sin_d)
                                - DtDstvec1 * bond_length * cos_a
                                - Dstvec1 * (d_ij_t * cos_a - bond_length * d_ijk_t * sin_a)
                                - Dttvec1 * (d_ij_s * cos_a - bond_length * d_ijk_s * sin_a)
                                - tvec1 * (- d_ij_s * d_ijk_t * sin_a 
                                              - d_ij_t * d_ijk_s * sin_a 
                                              - bond_length * d_ijk_s * d_ijk_t * cos_a)
                            )

                        K[3 * i:3 * i + 3, t, s] = K[3 * i:3 * i + 3, s, t]              
        return K     
                    

    @staticmethod
    def get_single_fd_grad(zmat, zmat_conn, q_i, q_j, r_i, r_j, delta):
        """
        Calculate the gradient dr/dq via the central difference method.
        """
        if delta == 0:
            raise ValueError("delta must be non-zero.")
        if q_i < 0 or q_i >= len(zmat):
            raise IndexError("q_i is out of range for the zmat.")
        if q_j not in [1, 2, 3]:
            raise ValueError("q_j must be 1 (bond), 2 (angle) or 3 (dihedral).")
        if r_i < 0 or r_i >= len(zmat):
            raise IndexError("r_i is out of range for the zmat.")
        if r_j not in [0, 1, 2]:
            raise ValueError("r_j must be 0 (x), 1 (y) or 2 (z).")
    
        zmat_fwd = copy.deepcopy(zmat)
        zmat_fwd[q_i][q_j] = zmat[q_i][q_j] + delta
        atoms_fwd = ZmatUtils.zmat_2_atoms(zmat_fwd, zmat_conn)
        try:
            r_fwd = atoms_fwd[r_i].position[r_j]
        except IndexError:
            raise IndexError("r_i is out of range in the forward-perturbed atoms object.")
    
        zmat_bwd = copy.deepcopy(zmat)
        zmat_bwd[q_i][q_j] = zmat[q_i][q_j] - delta
        atoms_bwd = ZmatUtils.zmat_2_atoms(zmat_bwd, zmat_conn)
        try:
            r_bwd = atoms_bwd[r_i].position[r_j]
        except IndexError:
            raise IndexError("r_i is out of range in the backward-perturbed atoms object.")
    
        fd_grad = (r_fwd - r_bwd) / (2 * delta)

        if q_j in [2, 3]:
            fd_grad *= 180.0 / np.pi
        return fd_grad

    @staticmethod
    def get_fd_B_matrix(zmat, zmat_conn, db, da, dt):
        """
        Calculate the Wilson B-matrix via finite difference given a Z-matrix and its connectivities.
        """
        if db == 0 or da == 0 or dt == 0:
            raise ValueError("Finite difference step sizes db, da, and dt must all be non-zero.")
        if len(zmat) == 0:
            raise ValueError("zmat cannot be empty.")
        if len(zmat) != len(zmat_conn):
            raise ValueError("Length of zmat and zmat_conn must be equal.")
    
        deltas = [db, da, dt]
        N = len(zmat)
        B = np.zeros((3 * N - 6, 3 * N), dtype=float)
        row_cnt = 0
    
        for q_i, row in enumerate(zmat):
            for q_j in range(1, 4):
                if row[q_j] is None:
                    continue
                col_cnt = 0
                for r_i in range(N):
                    for r_j in range(3):
                        grad = ZmatUtils.get_single_fd_grad(zmat, zmat_conn, q_i, q_j, r_i, r_j, deltas[q_j - 1])
                        B[row_cnt, col_cnt] = grad
                        col_cnt += 1
                row_cnt += 1
        return B

    
    @staticmethod
    def _displace(zmat, atom, kind, delta):
        """Return a copy of the Z-matrix with q(atom,kind) shifted by ±delta."""
        z = copy.deepcopy(zmat)
        z[atom][kind] += delta
        return z
    

    @staticmethod
    def get_fd_curvature_tensor(zmat, zmat_conn, db, da, dt):
        """
        K[i, s, t] = ∂²r_i / ∂q_s ∂q_t   (shape 3N × M × M) using
        finite differences of the B-matrix.
        """
        # -----------------------------------------------------------------
        # 1) internal-coordinate bookkeeping
        # -----------------------------------------------------------------
        valid   = [(i, j) for i, row in enumerate(zmat)
                            for j in range(1, len(row)) if row[j] is not None]
        M       = len(valid)
        idx_of  = {k: m for m, k in enumerate(valid)}   # (atom,kind) → 0…M−1
    
        deltas  = [db, da, dt]                          # step sizes
        scale   = np.array([1.0 if j == 1 else 180.0 / np.pi  # factor per column
                            for (_, j) in valid])
    
        atoms   = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
        n_cart  = 3 * len(atoms)
        K       = np.zeros((n_cart, M, M))
    
        # -----------------------------------------------------------------
        # 2) iterate over “second-derivative” index t (columns)
        # -----------------------------------------------------------------
        for t, (atom_t, kind_t) in enumerate(valid):
    
            delta_t = deltas[kind_t - 1]
    
            # B(q_t + delta) and B(q_t − delta)
            bp = ZmatUtils.get_B_matrix(ZmatUtils._displace(zmat, atom_t, kind_t, + delta_t),
                                        zmat_conn)
            bm = ZmatUtils.get_B_matrix(ZmatUtils._displace(zmat, atom_t, kind_t, - delta_t),
                                        zmat_conn)
    
            # central finite difference dB/dq_t
            dB = (bp - bm) / (2.0 * delta_t)            # shape (M, 3N)
            if kind_t in (2, 3):                        # angle or torsion column
                dB *= 180.0 / np.pi
    
            # -----------------------------------------------------------------
            # 3) write the whole column into the tensor and enforce symmetry
            # -----------------------------------------------------------------
            K[:, :, t] = dB.T                           # (3N, M) → tensor col
            K[:, t, :] = dB.T                           # mirror to keep K symmetric
    
        return K

    @staticmethod
    def kabsch(P, Q):
        """
        Compute the optimal rotation matrix to align P to Q using the Kabsch algorithm.
    
        Parameters:
            P, Q : np.ndarray of shape (N, 3)
                The coordinates of N atoms in each structure (assumed ordered consistently).
    
        Returns:
            R : np.ndarray of shape (3, 3)
                The rotation matrix.
            P_com : np.ndarray
                Center-of-mass of P (arithmetic mean).
            Q_com : np.ndarray
                Center-of-mass of Q (arithmetic mean).
        """
        # Compute centers (using arithmetic mean here; for mass-weighted, use atoms.get_center_of_mass())
        P_com = P.mean(axis=0)
        Q_com = Q.mean(axis=0)
        P_centered = P - P_com
        Q_centered = Q - Q_com
    
        # Compute covariance matrix
        C = np.dot(P_centered.T, Q_centered)
        
        # SVD of the covariance matrix
        U, S, Vt = np.linalg.svd(C)
        
        # Ensure a right-handed coordinate system
        d = np.linalg.det(np.dot(U, Vt))
        if d < 0:
            U[:, -1] *= -1
    
        R = np.dot(U, Vt)
        return R, P_com, Q_com

    @staticmethod
    def calculate_rmsd(atoms1, atoms2):
        """
        Calculate the RMSD between two ASE Atoms objects after optimal alignment.
        
        The structures are first aligned by subtracting their centers of mass,
        then rotated using the optimal rotation (via Kabsch algorithm) to minimize RMSD.
        
        Parameters:
            atoms1, atoms2 : ase.Atoms
                The two molecular structures to compare.
        
        Returns:
            rmsd : float
                The root-mean-square deviation between the aligned structures.
        """
        # Extract positions from ASE Atoms objects
        pos1 = atoms1.get_positions()
        pos2 = atoms2.get_positions()
        
        # Ensure both structures have the same number of atoms
        if pos1.shape != pos2.shape:
            raise ValueError("Both Atoms objects must have the same number of atoms.")
        
        # Get optimal rotation matrix and center-of-mass values
        R, com1, com2 = ZmatUtils.kabsch(pos1, pos2)
        
        # Align atoms1 to atoms2:
        pos1_aligned = np.dot(pos1 - com1, R) + com2
        
        # Calculate RMSD
        diff = pos1_aligned - pos2
        rmsd = np.sqrt(np.sum(diff**2) / pos1.shape[0])
        return rmsd

    @staticmethod
    def calculate_rmsd_zmat(zmat1, zmat2, zmat_conn):
        if not (len(zmat1) == len(zmat2)):
            raise ValueError("Shape mismatch between input molecules.")
        elif not (len(zmat1) == len(zmat_conn)):
            raise ValueError("Shape mismatch between molecule 1 and connectivity.")
        elif not (len(zmat2) == len(zmat_conn)):
            raise ValueError("Shape mismatch between molecule 2 and connectivity.")

        atoms1 = ZmatUtils.zmat_2_atoms(zmat1, zmat_conn)
        atoms2 = ZmatUtils.zmat_2_atoms(zmat2, zmat_conn)

        return ZmatUtils.calculate_rmsd(atoms1, atoms2)
    
    @staticmethod
    def validate_frames_python(xyz, conn, eps=1e-12):
        """
        Pre-check for zero bond lengths / collinearity so the Numba kernel doesn't explode.
        conn: (N,3) int array, where conn[i,0:2] = (j,k,l)
        Returns a boolean mask valid[i] for i>=3.
        """
        N = xyz.shape[0]
        valid = np.ones(N, dtype=bool)
        for i in range(3, N):
            j, k, l = conn[i, 0], conn[i, 1], conn[i, 2]
            r_kj = xyz[j] - xyz[k]
            r_kl = xyz[l] - xyz[k]
            nkj = np.linalg.norm(r_kj)
            nkl = np.linalg.norm(r_kl)
            if nkj < eps or nkl < eps:
                valid[i] = False
                continue
            t1 = r_kj / nkj
            t2 = r_kl / nkl
            cr = np.cross(t1, t2)
            if np.linalg.norm(cr) < eps:
                valid[i] = False
        return valid

    
    @staticmethod
    def get_curvature_tensor_numba(zmat, zmat_conn):
        """
        You supply:
            xyz: (N,3)
            B:   (3N-6,3N)
            conn: (N,4) int (constructed from zmat_conn)
            bl, ang, dih: (N,) float arrays
        """
        def extract_zmat_params(zmat):
            """
            Extract bond lengths, angles, dihedrals from a Z-matrix.

            Returns
            -------
            bl  : (N,) float64   bond lengths
            ang : (N,) float64   bond angles in radians
            dih : (N,) float64   dihedrals in radians
            """
            N = len(zmat)

            bl  = np.zeros(N, dtype=np.float64)
            ang = np.zeros(N, dtype=np.float64)
            dih = np.zeros(N, dtype=np.float64)

            for i in range(N):
                row = zmat[i]

                # bond length
                if len(row) > 1 and row[1] is not None:
                    bl[i] = float(row[1])

                # bond angle (convert once, here)
                if len(row) > 2 and row[2] is not None:
                    ang[i] = np.deg2rad(float(row[2]))

                # dihedral (convert once, here)
                if len(row) > 3 and row[3] is not None:
                    dih[i] = np.deg2rad(float(row[3]))

            return bl, ang, dih
        
        bl, ang, dih = extract_zmat_params(zmat)

        atoms = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
        xyz = atoms.get_positions()
        B = ZmatUtils.get_B_matrix(zmat, zmat_conn)
        
        conn = np.zeros((len(zmat_conn), 3), dtype=np.int64)

        for i, (_, j, k, l) in enumerate(zmat_conn):
            conn[i, 0] = j if j is not None else -1
            conn[i, 1] = k if k is not None else -1
            conn[i, 2] = l if l is not None else -1

        xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        valid = ZmatUtils.validate_frames_python(xyz, conn)

        return get_curvature_tensor_kernel_numba(xyz, B, conn, bl, ang, dih, valid)
    
    
