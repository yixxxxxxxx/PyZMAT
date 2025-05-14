from ase import Atoms
from constraints import Constraints

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