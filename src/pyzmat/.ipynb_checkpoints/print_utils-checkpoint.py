from ase import Atoms
from .constraints import Constraints
import numpy as np

class PrintUtils:
    @staticmethod
    def print_xyz(atoms, comment = '', fmt = '%22.15f', filename = None):
        comment = comment.rstrip()
        if '\n' in comment:
            raise ValueError('Comment line should not have line breaks.')
        natoms = len(atoms)
        print(f'{natoms}')
        print(comment)
        counter_dict = {}
        for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
            if s not in counter_dict:
                counter_dict[s] = 1
            else:
                counter_dict[s] += 1
            s_counted = s + str(counter_dict[s])
            print('%-2s %s %s %s' % (s_counted, fmt % x, fmt % y, fmt % z))

    @staticmethod
    def print_zmat(zmat, zmat_conn, constraints = None):

        # If no constraints are provided, set to an object with empty lists.
        if constraints is None:
            class DummyConstraints:
                def __init__(self):
                    self.bonds = []
                    self.angles = []
                    self.dihedrals = []
            constraints = DummyConstraints()
            
        # Here, each zmat_conn entry is a 4-tuple: (symbol, bond_ref, angle_ref, dihedral_ref)
        for i, (zmat_row, conn_row) in enumerate(zip(zmat, zmat_conn)):
            symbol = zmat_row[0]
            # Extract connectivity indices from the new tuple: skip the first element (symbol)
            c1 = conn_row[1] + 1 if conn_row[1] is not None else ''
            c2 = conn_row[2] + 1 if conn_row[2] is not None else ''
            c3 = conn_row[3] + 1 if conn_row[3] is not None else ''
            if i == 0:
                print(f"{symbol}")
            elif i == 1:
                print(f"{symbol}    {c1}    bnd{i + 1}")
            elif i == 2:
                print(f"{symbol}    {c1}    bnd{i + 1}    {c2}    ang{i + 1}")
            else:
                print(f"{symbol}    {c1}    bnd{i + 1}    {c2}    ang{i + 1}    {c3}    dih{i + 1}")
        print("Variables:")
        # For each atom (from index 1 onward), print the DOFs that are not constrained.
        for i in range(1, len(zmat)):
            atom = zmat[i]
            if not any(idx == i for idx, _ in constraints.bonds):
                if atom[1] is not None:
                    print(f"bnd{i+1}     {atom[1]:.6f}")
            if i >= 2 and not any(idx == i for idx, _ in constraints.angles):
                if atom[2] is not None:
                    print(f"ang{i+1}     {atom[2]:.6f}")
            if i >= 3 and not any(idx == i for idx, _ in constraints.dihedrals):
                if atom[3] is not None:
                    dih = atom[3]
                    if dih > 180:
                        dih = dih - 360
                    print(f"dih{i+1}     {dih:.6f}")
        if constraints.bonds or constraints.angles or constraints.dihedrals:
            print("Constants:")
            for idx, val in constraints.bonds:
                cur_val = val if val is not None else zmat[idx][1]
                print(f"bnd{idx+1}     {cur_val:.6f}")
            for idx, val in constraints.angles:
                cur_val = val if val is not None else zmat[idx][2]
                print(f"ang{idx+1}     {cur_val:.6f}")
            for idx, val in constraints.dihedrals:
                cur_val = val if val is not None else zmat[idx][3]
                if cur_val > 180:
                    cur_val = cur_val - 360
                print(f"dih{idx+1}     {cur_val:.6f}")

    @staticmethod
    def print_forces(forces, zmat, fmt='%22.15f'):
        blank = ' ' * 22
        counter_dict = {}
        for i, row in enumerate(zmat):
            s = row[0]
            if s not in counter_dict:
                counter_dict[s] = 1
            else:
                counter_dict[s] += 1
            s_counted = s + str(counter_dict[s])
            
            if i == 0:
                print('%-2s %s %s %s' % (s_counted, blank, blank, blank))
            elif i == 1:
                print('%-2s %s %s %s' % (s_counted, fmt % forces[0], blank, blank))
            elif i == 2:
                print('%-2s %s %s %s' % (s_counted, fmt % forces[1], fmt % forces[2], blank))
            else:
                force_bond = forces[3 * i - 6]
                force_angle = forces[3 * i - 5]
                force_torsion = forces[3 * i - 4]
                print('%-2s %s %s %s' % (s_counted, fmt % force_bond, fmt % force_angle, fmt % force_torsion))

    @staticmethod
    def print_hessian(hessian, zmat, constraints=None, block_size=5):
        """
        Print the lower‐triangular part of a Hessian matrix, but reorder so that
        any DOFs listed in constraints appear at the end.  Variable names are
        built internally from `zmat`.  Columns are delimited by two spaces,
        and the first digit of every number (not the sign) lines up in the same
        column.
        """
        H = np.asarray(hessian)
        m = H.shape[0]
        if H.shape[1] != m:
            raise ValueError("Hessian must be square")
    
        # default empty constraints
        if constraints is None:
            class _C:
                bonds = []
                angles = []
                dihedrals = []
            constraints = _C()
    
        # 1) build the original variable‐name list in zmat order
        orig_names = []
        n = len(zmat)
        for i in range(1, n):
            if zmat[i][1] is not None:
                orig_names.append(f"bnd{i+1}")
            if i >= 2 and zmat[i][2] is not None:
                orig_names.append(f"ang{i+1}")
            if i >= 3 and zmat[i][3] is not None:
                orig_names.append(f"dih{i+1}")
    
        if len(orig_names) != m:
            raise ValueError(f"zmat yields {len(orig_names)} DOFs, but Hessian is {m}×{m}")
    
        # 2) collect the constant DOFs
        const_names = [f"bnd{idx+1}" for idx, _ in constraints.bonds] + \
                      [f"ang{idx+1}" for idx, _ in constraints.angles] + \
                      [f"dih{idx+1}" for idx, _ in constraints.dihedrals]
    
        # 3) split into non‐const then const
        nonconst = [nm for nm in orig_names if nm not in const_names]
        new_order = nonconst + const_names
    
        # 4) permute H to match the new order
        idx_map = [orig_names.index(nm) for nm in new_order]
        H2 = H[np.ix_(idx_map, idx_map)]
    
        # 5) determine field widths so digits align at col 1
        example = f"{0.0:.8E}"         # e.g. "0.00000000E+00" (14 chars)
        digit_width = len(example)     # 14
        fw = digit_width + 1           # reserve 1 char for sign or leading space -> 15
    
        # 6) print in blocks of columns
        for block_start in range(0, m, block_size):
            block_end = min(block_start + block_size, m)
    
            # header row
            print(" " * (fw + 1), end=" ")
            for j in range(block_start, block_end):
                print(f"{new_order[j]:{fw}s}", end=" ")
            print()
    
            # data rows
            for i in range(block_start, m):
                # row label
                print(f"{new_order[i]:{fw}s}", end=" ")
                for j in range(block_start, block_end):
                    val = H2[i, j]
                    # build abs‐value string
                    sig = "-" if val < 0 else " "
                    body = f"{abs(val):.8E}"       # always starts with a digit
                    entry = (sig + body).ljust(fw)  # pad on right
                    print(entry, end=" ")
                print()