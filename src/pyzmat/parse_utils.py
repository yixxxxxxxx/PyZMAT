# src/zmatopt/parse_utils.py

import ast
import re
import numpy as np
from ase import Atoms
from ase.units import Bohr, Ha

from .constraints import Constraints
from .zmat_utils import ZmatUtils


class ParseUtils:

    @staticmethod
    def get_zmat_def(out_file):
        """
        Read a Python‐literal Z-matrix definition from `out_file` and
        convert 1-based indices to 0-based, preserving empty slots.
        """
        zmat_def = []
        with open(out_file, 'r') as file:
            content = file.read()
            data = ast.literal_eval(content)
            for row in data:
                # empty strings stay None, integers convert to int-1
                zmat_row = tuple(int(i) - 1 if i != '' else None for i in row)
                zmat_def.append(zmat_row)
        return zmat_def

    @staticmethod
    def parse_zmat(zmat_str):
        """
        Parse a plain-text Z-matrix (e.g. from Gaussian_to_CP_input_V5.py)
        using full labels (C1, H2, …) to build connectivity, then strip digits.
        Returns a list of (symbol, bond_ref, angle_ref, dihedral_ref).
        """
        if not isinstance(zmat_str, str):
            raise TypeError("zmat_str must be a string.")

        lines = zmat_str.strip().splitlines()
        if not lines:
            raise ValueError("The provided Z-matrix string is empty.")
        if lines[0].lower().startswith("z-matrix"):
            lines = lines[1:]

        label_mapping = {}
        zmat_conn = []

        # First pass: assign each full label to its line index
        for i, line in enumerate(lines):
            tokens = line.split()
            if not tokens:
                continue
            atom_label = tokens[0]
            label_mapping[atom_label] = i
            # gather up to three references
            refs = []
            for tok in tokens[1:4]:
                if tok:
                    if tok not in label_mapping:
                        raise ValueError(f"Reference label '{tok}' in line {i+1} not defined previously.")
                    refs.append(label_mapping[tok])
                else:
                    refs.append(None)
            # pad to length 3
            while len(refs) < 3:
                refs.append(None)
            zmat_conn.append((atom_label, *refs))

        # strip trailing digits from symbols
        def strip_num(s):
            return re.sub(r'\d+$', '', s)

        final = [(strip_num(sym), b, a, d) for sym, b, a, d in zmat_conn]
        return final

    @staticmethod
    def parse_gaussian_input_old(com_file):
        """
        Legacy parser for Gaussian input (.com) files containing a Z-matrix
        with Variables: and Constants: blocks. Returns (zmat, zmat_conn).
        """
        with open(com_file, "r") as f:
            lines = f.readlines()

        zmatrix_lines = []
        variables = {}
        constants = {}

        reading_z = reading_v = reading_c = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not reading_z and line.startswith("0 "):
                reading_z = True
                continue
            if reading_z and line.lower().startswith("variables:"):
                reading_z, reading_v = False, True
                continue
            if reading_v and line.lower().startswith("constants:"):
                reading_v, reading_c = False, True
                continue

            if reading_z:
                zmatrix_lines.append(line)
            elif reading_v:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        variables[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
            elif reading_c:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        constants[parts[0]] = float(parts[1])
                    except ValueError:
                        pass

        # merge params
        params = {**variables, **constants}

        zmat = []
        zmat_conn = []
        for i, line in enumerate(zmatrix_lines):
            tokens = line.split()
            atom = re.match(r"([A-Za-z]+)", tokens[0]).group(1)
            bond = angle = dihed = None
            conn = [None, None, None]

            if len(tokens) == 3:
                ref, param = int(tokens[1]) - 1, tokens[2]
                bond = params.get(param)
                conn = [ref, None, None]
            elif len(tokens) == 5:
                b_ref, b_param, a_ref, a_param = int(tokens[1]) - 1, tokens[2], int(tokens[3]) - 1, tokens[4]
                bond = params.get(b_param)
                angle = params.get(a_param)
                conn = [b_ref, a_ref, None]
            elif len(tokens) == 7:
                b_ref, b_param, a_ref, a_param, d_ref, d_param = (
                    int(tokens[1]) - 1, tokens[2], int(tokens[3]) - 1,
                    tokens[4], int(tokens[5]) - 1, tokens[6]
                )
                bond = params.get(b_param)
                angle = params.get(a_param)
                dihed = params.get(d_param)
                conn = [b_ref, a_ref, d_ref]

            zmat.append([atom, bond, angle, dihed])
            zmat_conn.append((atom, *conn))

        return zmat, zmat_conn

    @staticmethod
    def parse_gaussian_input(com_file):
        """
        New parser that also collects Constants into a Constraints object.
        Returns (zmat, zmat_conn, constraints).
        """
        with open(com_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        zmatrix_lines = []
        vars_dict = {}
        consts_dict = {}
        state = "search"

        for line in lines:
            low = line.lower()
            if state == "search" and line.startswith("0 "):
                state = "zmat"
                continue
            if state == "zmat" and low.startswith("variables:"):
                state = "vars"
                continue
            if state == "vars" and low.startswith("constants:"):
                state = "consts"
                continue

            if state == "zmat":
                zmatrix_lines.append(line)
            elif state == "vars":
                key, *rest = line.split()
                try:
                    vars_dict[key] = float(rest[0])
                except (IndexError, ValueError):
                    pass
            elif state == "consts":
                key, *rest = line.split()
                try:
                    consts_dict[key] = float(rest[0])
                except (IndexError, ValueError):
                    pass

        # combine params
        params = {**vars_dict, **consts_dict}

        # parse geometry & build constraints lists
        zmat = []
        zmat_conn = []
        bonds_c, ang_c, dih_c = [], [], []

        for i, line in enumerate(zmatrix_lines):
            tokens = line.split()
            atom = re.match(r"([A-Za-z]+)", tokens[0]).group(1)
            bond = angle = dihed = None
            b_ref = a_ref = d_ref = None

            if len(tokens) == 3:
                b_ref, bond = int(tokens[1]) - 1, params.get(tokens[2])
            elif len(tokens) == 5:
                b_ref = int(tokens[1]) - 1
                bond = params.get(tokens[2])
                a_ref = int(tokens[3]) - 1
                angle = params.get(tokens[4])
            elif len(tokens) == 7:
                b_ref = int(tokens[1]) - 1
                bond = params.get(tokens[2])
                a_ref = int(tokens[3]) - 1
                angle = params.get(tokens[4])
                d_ref = int(tokens[5]) - 1
                dihed = params.get(tokens[6])

            zmat.append([atom, bond, angle, dihed])
            zmat_conn.append((atom, b_ref, a_ref, d_ref))

        # build Constraints from constants only
        for key, val in consts_dict.items():
            lk = key.lower()
            if lk.startswith("bnd"):
                bonds_c.append((int(lk[3:]) - 1, val))
            elif lk.startswith("ang"):
                ang_c.append((int(lk[3:]) - 1, val))
            elif lk.startswith("dih"):
                dih_c.append((int(lk[3:]) - 1, val))

        constraints = Constraints(bonds=bonds_c, angles=ang_c, dihedrals=dih_c)
        return zmat, zmat_conn, constraints

    @staticmethod
    def parse_gaussian_fchk(filename, zmat_conn):
        """
        Read a Gaussian .fchk file and extract:
          - optimized Cartesian coords to ASE Atoms
          - internal forces (converted to eV/A or eV/deg)
          - total energy (eV)
        Returns (zmat, forces, energy).
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        coords = []
        forces = []
        energy = None
        read_c = read_f = False
        Ncoords = Nforces = None

        # first pass: energy
        for L in lines:
            if L.startswith("Total Energy"):
                toks = L.split()
                energy = float(toks[3]) * Ha
                break

        # parse coords & forces
        for L in lines:
            if "Current cartesian coordinates" in L:
                Ncoords = int(L.split("N=")[1])
                read_c = True
                continue
            if "Internal Forces" in L:
                Nforces = int(L.split("N=")[1])
                read_f = True
                continue

            if read_c:
                if not L.strip():
                    if len(coords) >= Ncoords:
                        read_c = False
                    continue
                coords.extend(float(x) for x in L.split())
                if len(coords) >= Ncoords:
                    read_c = False

            if read_f:
                if not L.strip():
                    if len(forces) >= Nforces:
                        read_f = False
                    continue
                forces.extend(float(x) for x in L.split())
                if len(forces) >= Nforces:
                    read_f = False

        if len(coords) < Ncoords:
            raise ValueError("Incomplete coordinate block in FCHK.")
        positions = np.array(coords[:Ncoords]).reshape(-1, 3) * Bohr

        # convert forces: Ha/Bohr → eV/Å for x,y,z; Ha/rad → eV/deg otherwise
        forces_si = []
        for i, f in enumerate(forces):
            if i % 3 == 0:
                forces_si.append(f * (Ha / Bohr))
            else:
                forces_si.append(f * (Ha * np.pi / 180))

        # build Atoms and back‐convert to Z‐matrix
        symbols = [entry[0] for entry in zmat_conn]
        atoms = Atoms(symbols=symbols, positions=positions)
        zmat = ZmatUtils.atoms_2_zmat(atoms, zmat_conn)

        return zmat, forces_si, energy

    @staticmethod
    def parse_gaussian_log(filename, check_calc=False):
        """
        Parse a Gaussian .log optimized run, extracting:
          - symbolic Z‐matrix → zmat_conn and param names
          - Variables/Constants → param values
          - Optimized Parameters → updated internal coords & forces
          - final SCF Done energy
        Returns (zmat, zmat_conn, forces, energy[, normal_termination]).
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        # 1) Symbolic Z-matrix
        zmat_conn = []
        param_names = []
        in_z = False
        skip = False
        for L in lines:
            if "Symbolic Z-matrix:" in L:
                in_z = True
                skip = True
                continue
            if in_z:
                if skip:
                    skip = False
                    continue
                if L.strip().lower().startswith("variables:"):
                    break
                if not L.strip():
                    continue
                toks = L.split()
                sym = re.match(r"([A-Za-z]+)", toks[0]).group(1)
                if len(toks) == 1:
                    zmat_conn.append((sym, None, None, None))
                    param_names.append((None, None, None))
                elif len(toks) == 3:
                    b, bp = int(toks[1]) - 1, toks[2]
                    zmat_conn.append((sym, b, None, None))
                    param_names.append((bp, None, None))
                elif len(toks) == 5:
                    b, bp, a, ap = int(toks[1]) - 1, toks[2], int(toks[3]) - 1, toks[4]
                    zmat_conn.append((sym, b, a, None))
                    param_names.append((bp, ap, None))
                else:
                    b, bp, a, ap, d, dp = (
                        int(toks[1]) - 1, toks[2], int(toks[3]) - 1,
                        toks[4], int(toks[5]) - 1, toks[6]
                    )
                    zmat_conn.append((sym, b, a, d))
                    param_names.append((bp, ap, dp))

        # 2) Variables & Constants
        var_dict = {}
        in_v = in_c = False
        for L in lines:
            l = L.strip().lower()
            if l.startswith("variables:"):
                in_v, in_c = True, False
                continue
            if l.startswith("constants:"):
                in_v, in_c = False, True
                continue
            if in_v or in_c:
                parts = L.split()
                if len(parts) >= 2:
                    try:
                        var_dict[parts[0]] = float(parts[1])
                    except ValueError:
                        pass

        # 3) Optimized Parameters & forces
        opt_block = False
        opt_params = {}
        forces_list = []
        for L in lines:
            if "Optimized Parameters" in L:
                opt_block = True
                continue
            if opt_block:
                if not L.strip().startswith("!"):
                    continue
                clean = L.strip("! \n")
                parts = clean.split()
                if len(parts) >= 2:
                    key = parts[0]
                    try:
                        val = float(parts[1])
                        opt_params[key] = val
                        # attempts to read force after “-DE/DX =”
                        if "DE/DX" in clean:
                            forces_list.append(float(parts[-1]))
                    except ValueError:
                        pass

        var_dict.update(opt_params)

        # build final zmat values
        zmat = []
        for i, ((sym, b, a, d), (bp, ap, dp)) in enumerate(zip(zmat_conn, param_names)):
            if i == 0:
                zmat.append([sym, None, None, None])
            elif i == 1:
                zmat.append([sym, var_dict.get(bp), None, None])
            elif i == 2:
                zmat.append([sym, var_dict.get(bp), var_dict.get(ap), None])
            else:
                zmat.append([sym,
                             var_dict.get(bp),
                             var_dict.get(ap),
                             var_dict.get(dp)])

        # 4) Final SCF energy & termination flag
        energy = None
        normal_term = False
        for L in reversed(lines):
            if L.strip().startswith("Normal termination"):
                normal_term = True
            if L.strip().startswith("SCF Done:"):
                parts = L.split()
                try:
                    energy = float(parts[4]) * Ha
                except Exception:
                    pass
                if energy is not None:
                    break

        forces = [f * Ha for f in forces_list]

        if len(zmat) == 0:
            print('Z-Matrix detection unsuccessful.')
            print('Was this calculation restarted from a checkpoint file? If yes, please consider using parse_gaussian_fchk(filename, zmat_conn).')
            print('(Hint: zmat_conn can be obtained using parse_gaussian_input(com_file).)')
        if check_calc:
            return zmat, zmat_conn, forces, energy, normal_term
        else:
            return zmat, zmat_conn, forces, energy
