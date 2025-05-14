import numpy as np
from ase import Atoms

class ZmatUtils:
    """
    Contains functions concerning the conversion of structures between cartesian and internal coordinates, as well as the first and second derivatives of cartesian coordinates w.r.t. internal coordinates.
    """

    @staticmethod
    def kronecker_delta(i, j):
        """
        The Kronecker δ function returns 1 if its two inputs are equal and 0 otherwise.
        """
        if not isinstance(i, int) or not isinstance(j, int):
            raise TypeError("Inputs to kronecker_delta must be integers.")
        return 1 if i == j else 0

    @staticmethod
    def atoms_2_zmat_init(atoms, zmat_def):
        """
        Convert an ASE Atoms object (in Cartesian coordinates) into a Z-matrix representation.
        """
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
        #K[np.abs(K) < 1e-9] = 0                
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

        return calculate_rmsd(atoms1, atoms2)
