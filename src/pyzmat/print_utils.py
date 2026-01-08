from ase import Atoms
from .constraints import Constraints
import numpy as np
import io
from typing import List, Tuple, Optional, Dict

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
				print(f"{symbol}	{c1}	bnd{i + 1}")
			elif i == 2:
				print(f"{symbol}	{c1}	bnd{i + 1}	{c2}	ang{i + 1}")
			else:
				print(f"{symbol}	{c1}	bnd{i + 1}	{c2}	ang{i + 1}	{c3}	dih{i + 1}")
		print("Variables:")
		# For each atom (from index 1 onward), print the DOFs that are not constrained.
		for i in range(1, len(zmat)):
			atom = zmat[i]
			if not any(idx == i for idx, _ in constraints.bonds):
				if atom[1] is not None:
					print(f"bnd{i+1}	 {atom[1]:.6f}")
			if i >= 2 and not any(idx == i for idx, _ in constraints.angles):
				if atom[2] is not None:
					print(f"ang{i+1}	 {atom[2]:.6f}")
			if i >= 3 and not any(idx == i for idx, _ in constraints.dihedrals):
				if atom[3] is not None:
					dih = atom[3]
					if dih > 180:
						dih = dih - 360
					print(f"dih{i+1}	 {dih:.6f}")
		if constraints.bonds or constraints.angles or constraints.dihedrals:
			print("Constants:")
			for idx, val in constraints.bonds:
				cur_val = val if val is not None else zmat[idx][1]
				print(f"bnd{idx+1}	 {cur_val:.6f}")
			for idx, val in constraints.angles:
				cur_val = val if val is not None else zmat[idx][2]
				print(f"ang{idx+1}	 {cur_val:.6f}")
			for idx, val in constraints.dihedrals:
				cur_val = val if val is not None else zmat[idx][3]
				if cur_val > 180:
					cur_val = cur_val - 360
				print(f"dih{idx+1}	 {cur_val:.6f}")

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
		example = f"{0.0:.8E}"		 # e.g. "0.00000000E+00" (14 chars)
		digit_width = len(example)	 # 14
		fw = digit_width + 1		   # reserve 1 char for sign or leading space -> 15
	
		# 6) print in blocks of columns
		for block_start in range(0, m, block_size):
			block_end = min(block_start + block_size, m)
	
			# header row
			print(" " * (fw - 6 + 1), end=" ")
			for j in range(block_start, block_end):
				print(f"{new_order[j]:{fw}s}", end=" ")
			print()
	
			# data rows
			for i in range(block_start, m):
				# row label
				print(f"{new_order[i]:{fw - 9}s}", end=" ")
				for j in range(block_start, block_end):
					if j <= i:
						val = H2[i, j]
						# build abs‐value string
						sig = "-" if val < 0 else " "
						body = f"{abs(val):.8E}"	   # always starts with a digit
						entry = (sig + body).ljust(fw)  # pad on right
						print(entry, end=" ")
				print()



	@staticmethod
	def print_orca_input(
		zmat: List[List[object]],
		zmat_conn: List[Tuple[object, Optional[int], Optional[int], Optional[int]]],
		constraints = None,
		*,
		level_of_theory: str = "PBE0",
		basis_set: str = "6-311G(d,p)",
		maxcore_mb: int = 4000,
		nproc: int = 8,
		scf_maxiter: int = 256,
		scf_conv: str = "STRONG",
		geom_maxiter: int = 128,
		geom_conv: str = "TIGHT",
		tol_maxg: float = 4.5e-4,
		tol_rmsg: float = 3.0e-4,
		tol_maxd: float = 1.8e-3,
		tol_rmsd: float = 1.2e-3,
		use_symmetry: bool = False,
		scs: bool = True,
		charge: int = 0,
		multiplicity: int = 1,
		title_lines: Optional[List[str]] = None,
	) -> str:
		"""
		Build an ORCA input file (as a string) from zmat, zmat_conn, and constraints.

		Parameters
		----------
		zmat
			[[symbol, bond, angle, dihedral], ...]; numeric entries can be None.
		zmat_conn
			[(symbol, b_ref, a_ref, d_ref), ...] with 0-based integer refs or None.
		constraints
			{"bonds":[(row_idx, None)], "angles":[(row_idx, None)], "dihedrals":[(row_idx, None)]}
			Indices 0-based (your convention). We print ORCA constraints 0-indexed.
		Other keyword args configure header blocks; defaults match your earlier example.

		Returns
		-------
		str
			Complete ORCA input file content.
		"""

		def fnum(x: Optional[float]) -> str:
			# Format numbers compactly but stably for ORCA
			return f"{x:.6f}"
		


		# If no constraints are provided, set to an object with empty lists.
		if constraints is None:
			class DummyConstraints:
				def __init__(self):
					self.bonds = []
					self.angles = []
					self.dihedrals = []
			constraints = DummyConstraints()

		if geom_maxiter == 0:
			if basis_set.startswith('def2'):
				if 'RI-MP2' in level_of_theory.upper():
					ri_aux = basis_set + "/C"
					route_line: str = "! " + level_of_theory + " " + basis_set + " " + ri_aux +" RIJCOSX def2/J OPT D3BJ DEFGRID3 ENGRAD"
				else:	
					route_line: str = "! " + level_of_theory + " " + basis_set + " def2/J RIJCOSX D3BJ DEFGRID3 ENGRAD"				
			else:
				route_line: str = "! " + level_of_theory + " " + basis_set + " AutoAux RIJCOSX D3BJ DEFGRID3 ENGRAD"
		else:
			if basis_set.startswith('def2'):
				if 'RI-MP2' in level_of_theory.upper():
					ri_aux = basis_set + "/C"
					route_line: str = "! " + level_of_theory + " " + basis_set + " " + ri_aux +" RIJCOSX def2/J OPT D3BJ DEFGRID3"
				else:
					route_line: str = "! " + level_of_theory + " " + basis_set + " def2/J RIJCOSX OPT D3BJ DEFGRID3"
			else:
				route_line: str = "! " + level_of_theory + " " + basis_set + " AutoAux RIJCOSX D3BJ DEFGRID3 ENGRAD"
		# ---------- Build header ----------
		lines: List[str] = []
		lines.append("# Calculation type")
		lines.append(route_line)
		lines.append("")
		lines.append("# Calculation specifications")
		if 'MP2' in level_of_theory.upper() and scs:
			lines.append("%MP2")
			lines.append("   DoSCS     TRUE")
			lines.append("END")
		lines.append(f"%MaxCore {maxcore_mb}")
		lines.append("%PAL")
		lines.append(f"   NPROC {nproc}")
		lines.append("END")
		lines.append("%SYMMETRY")
		lines.append(f"   USESYMMETRY {'TRUE' if use_symmetry else 'FALSE'}")
		lines.append("END")
		lines.append("%SCF")
		lines.append(f"   MAXITER {scf_maxiter}")
		lines.append(f"   CONVERGENCE {scf_conv}")
		lines.append("END")
		if geom_maxiter != 0:
			lines.append("%GEOM")
			lines.append(f"   MAXITER     {geom_maxiter}")
			lines.append(f"   CONVERGENCE     {geom_conv}")
			lines.append(f"   TolMaxG     {tol_maxg:.6f}")
			lines.append(f"   TolRMSG     {tol_rmsg:.6f}")
			lines.append(f"   TolMaxD     {tol_maxd:.6f}")
			lines.append(f"   TolRMSD     {tol_rmsd:.6f}")
	
			# ---------- Constraints (0-indexed as per your convention) ----------
			def add_constraint_block():
				# Gather constraint lines, deriving j/k/l from zmat_conn
				c_lines: List[str] = []
	
				# Bonds
				for (row, _target) in constraints.bonds:
					if not (0 <= row < len(zmat_conn)):
						continue
					_, jb, _, _ = zmat_conn[row]
					if jb is None:
						continue
					# ORCA constraints we output 0-indexed (your rule)
					c_lines.append(f"   {{ B {row} {jb} C }}")
	
				# Angles
				for (row, _target) in constraints.angles:
					if not (0 <= row < len(zmat_conn)):
						continue
					_, jb, ka, _ = zmat_conn[row]
					if jb is None or ka is None:
						continue
					c_lines.append(f"   {{ A {row} {jb} {ka} C }}")
	
				# Dihedrals
				for (row, _target) in constraints.dihedrals:
					if not (0 <= row < len(zmat_conn)):
						continue
					_, jb, ka, ld = zmat_conn[row]
					if jb is None or ka is None or ld is None:
						continue
					c_lines.append(f"   {{ D {row} {jb} {ka} {ld} C }}")
	
				return c_lines
	
			cons_lines = add_constraint_block()
			if cons_lines:
				lines.append("   CONSTRAINTS")
				lines.extend(cons_lines)
				lines.append("   END")  # end of CONSTRAINTS
	
			lines.append("END")  # end of %GEOM
		lines.append("")

		# ---------- Title (optional) ----------
		if title_lines:
			for t in title_lines:
				lines.append(f"# {t}")
			lines.append("")

		# ---------- gzmt block ----------
		lines.append("# Molecule definition")
		lines.append(f"* gzmt {charge} {multiplicity}")

		# Emit each Z-matrix row. ORCA gzmt expects references **1-indexed**.
		for i, (row, conn) in enumerate(zip(zmat, zmat_conn)):
			sym = str(row[0])  # symbol from zmat (or conn[0])
			# prefer symbol from zmat if present; fall back to zmat_conn[0]
			if not sym or sym == "None":
				sym = str(conn[0])

			# Unpack values
			b_val = row[1] if len(row) > 1 else None
			a_val = row[2] if len(row) > 2 else None
			d_val = row[3] if len(row) > 3 else None

			# References (0-indexed) → write 1-indexed for ORCA
			_, jb, ka, ld = conn

			if i == 0:
				lines.append(f"{sym:<2}")
			elif i == 1:
				if jb is None or b_val is None:
					raise ValueError(f"Row {i}: missing bond reference/value.")
				lines.append(f"{sym:<2} {jb+1:>8d} {fnum(b_val):>14}")
			elif i == 2:
				if jb is None or b_val is None or ka is None or a_val is None:
					raise ValueError(f"Row {i}: missing angle info.")
				lines.append(
					f"{sym:<2} {jb+1:>8d} {fnum(b_val):>14} {ka+1:>8d} {fnum(a_val):>14}"
				)
			else:
				if jb is None or b_val is None or ka is None or a_val is None or ld is None or d_val is None:
					raise ValueError(f"Row {i}: missing dihedral info.")
				lines.append(
					f"{sym:<2} {jb+1:>8d} {fnum(b_val):>14} {ka+1:>8d} {fnum(a_val):>14} {ld+1:>8d} {fnum(d_val):>14}"
				)

		lines.append("*")  # end of gzmt block
		lines.append("")

		return "\n".join(lines)
	
	@staticmethod
	def print_orca_extopt_input(
		zmat: List[List[object]],
		zmat_conn: List[Tuple[object, Optional[int], Optional[int], Optional[int]]],
		wrapper_path,
		constraints = None,
		*,
		maxcore_mb: int = 1000,
		nproc: int = 1,
		scf_maxiter: int = 256,
		scf_conv: str = "STRONG",
		Ext_Params: str = "",
		geom_maxiter: int = 128,
		geom_conv: str = "TIGHT",
		tol_maxg: float = 4.5e-4,
		tol_rmsg: float = 3.0e-4,
		tol_maxd: float = 1.8e-3,
		tol_rmsd: float = 1.2e-3,
		use_symmetry: bool = False,
		charge: int = 0,
		multiplicity: int = 1,
		title_lines: Optional[List[str]] = None,
	) -> str:
		"""
		Build an ORCA input file (as a string) from zmat, zmat_conn, and constraints.

		Parameters
		----------
		zmat
			[[symbol, bond, angle, dihedral], ...]; numeric entries can be None.
		zmat_conn
			[(symbol, b_ref, a_ref, d_ref), ...] with 0-based integer refs or None.
		constraints
			{"bonds":[(row_idx, None)], "angles":[(row_idx, None)], "dihedrals":[(row_idx, None)]}
			Indices 0-based (your convention). We print ORCA constraints 0-indexed.
		Other keyword args configure header blocks; defaults match your earlier example.

		Returns
		-------
		str
			Complete ORCA input file content.
		"""

		def fnum(x: Optional[float]) -> str:
			# Format numbers compactly but stably for ORCA
			return f"{x:.6f}"
		


		# If no constraints are provided, set to an object with empty lists.
		if constraints is None:
			class DummyConstraints:
				def __init__(self):
					self.bonds = []
					self.angles = []
					self.dihedrals = []
			constraints = DummyConstraints()


		route_line: str = "!ExtOpt OPT"
		# ---------- Build header ----------
		lines: List[str] = []
		lines.append("# Calculation type")
		lines.append(route_line)
		lines.append("")
		lines.append("# Calculation specifications")
		lines.append(f"%MaxCore {maxcore_mb}")
		lines.append("%PAL")
		lines.append(f"   NPROC {nproc}")
		lines.append("END")
		lines.append("%SYMMETRY")
		lines.append(f"   USESYMMETRY {'TRUE' if use_symmetry else 'FALSE'}")
		lines.append("END")
		lines.append("%SCF")
		lines.append(f"   MAXITER {scf_maxiter}")
		lines.append(f"   CONVERGENCE {scf_conv}")
		lines.append("END")
		lines.append("%METHOD")
		lines.append(f'   ProgExt "{wrapper_path}"')
		lines.append(f'   ExtParams "{Ext_Params}"')
		lines.append("END")
		lines.append("%GEOM")
		lines.append(f"   MAXITER     {geom_maxiter}")
		lines.append(f"   CONVERGENCE     {geom_conv}")
		lines.append(f"   TolMaxG     {tol_maxg:.6f}")
		lines.append(f"   TolRMSG     {tol_rmsg:.6f}")
		lines.append(f"   TolMaxD     {tol_maxd:.6f}")
		lines.append(f"   TolRMSD     {tol_rmsd:.6f}")

		# ---------- Constraints (0-indexed as per your convention) ----------
		def add_constraint_block():
			# Gather constraint lines, deriving j/k/l from zmat_conn
			c_lines: List[str] = []

			# Bonds
			for (row, _target) in constraints.bonds:
				if not (0 <= row < len(zmat_conn)):
					continue
				_, jb, _, _ = zmat_conn[row]
				if jb is None:
					continue
				# ORCA constraints we output 0-indexed (your rule)
				c_lines.append(f"   {{ B {row} {jb} C }}")

			# Angles
			for (row, _target) in constraints.angles:
				if not (0 <= row < len(zmat_conn)):
					continue
				_, jb, ka, _ = zmat_conn[row]
				if jb is None or ka is None:
					continue
				c_lines.append(f"   {{ A {row} {jb} {ka} C }}")

			# Dihedrals
			for (row, _target) in constraints.dihedrals:
				if not (0 <= row < len(zmat_conn)):
					continue
				_, jb, ka, ld = zmat_conn[row]
				if jb is None or ka is None or ld is None:
					continue
				c_lines.append(f"   {{ D {row} {jb} {ka} {ld} C }}")

			return c_lines

		cons_lines = add_constraint_block()
		if cons_lines:
			lines.append("   CONSTRAINTS")
			lines.extend(cons_lines)
			lines.append("   END")  # end of CONSTRAINTS

		lines.append("END")  # end of %GEOM
		lines.append("")

		# ---------- Title (optional) ----------
		if title_lines:
			for t in title_lines:
				lines.append(f"# {t}")
			lines.append("")

		# ---------- gzmt block ----------
		lines.append("# Molecule definition")
		lines.append(f"* gzmt {charge} {multiplicity}")

		# Emit each Z-matrix row. ORCA gzmt expects references **1-indexed**.
		for i, (row, conn) in enumerate(zip(zmat, zmat_conn)):
			sym = str(row[0])  # symbol from zmat (or conn[0])
			# prefer symbol from zmat if present; fall back to zmat_conn[0]
			if not sym or sym == "None":
				sym = str(conn[0])

			# Unpack values
			b_val = row[1] if len(row) > 1 else None
			a_val = row[2] if len(row) > 2 else None
			d_val = row[3] if len(row) > 3 else None

			# References (0-indexed) → write 1-indexed for ORCA
			_, jb, ka, ld = conn

			if i == 0:
				lines.append(f"{sym:<2}")
			elif i == 1:
				if jb is None or b_val is None:
					raise ValueError(f"Row {i}: missing bond reference/value.")
				lines.append(f"{sym:<2} {jb+1:>8d} {fnum(b_val):>14}")
			elif i == 2:
				if jb is None or b_val is None or ka is None or a_val is None:
					raise ValueError(f"Row {i}: missing angle info.")
				lines.append(
					f"{sym:<2} {jb+1:>8d} {fnum(b_val):>14} {ka+1:>8d} {fnum(a_val):>14}"
				)
			else:
				if jb is None or b_val is None or ka is None or a_val is None or ld is None or d_val is None:
					raise ValueError(f"Row {i}: missing dihedral info.")
				lines.append(
					f"{sym:<2} {jb+1:>8d} {fnum(b_val):>14} {ka+1:>8d} {fnum(a_val):>14} {ld+1:>8d} {fnum(d_val):>14}"
				)

		lines.append("*")  # end of gzmt block
		lines.append("")

		return "\n".join(lines)

	@staticmethod
	def print_mace_training_xyz(atoms, energy, forces, comp="Mol(1)", pbc=(False, False, False)):
		"""
		Print an xyz block for a single molecule for MACE training.
		
		Output example:
		76
		Properties=species:S:1:pos:R:3:molID:I:1:forces_xtb:R:3 Nmols=1 Comp=Mol(1) energy_xtb=-1151.6688033579703 pbc="F F F"
		C   2.69088602   2.54591513  -2.12696218   0  -1.44302882   4.71896954   1.76744346
		...
		"""

		symbols = atoms.get_chemical_symbols()
		positions = np.array(atoms.get_positions())
		forces = np.array(forces)
		forces = forces.reshape((-1, 3))
		n_atoms = len(symbols)

		# pbc flags
		pbc_flags = " ".join("T" if x else "F" for x in pbc)

		# header line
		lines = [(
			f'\n{n_atoms}\n'
			f'Properties=species:S:1:pos:R:3:molID:I:1:forces_ref:R:3 '
			f'Nmols=1 Comp={comp} energy_ref={energy:.16f} pbc="{pbc_flags}"'
		)]

		# atom lines (molID always 0)
		for sym, (x, y, z), (fx, fy, fz) in zip(symbols, positions, forces):
			lines.append(f"{sym:<2s} {x:15.8f} {y:15.8f} {z:15.8f} {0:5d} {fx:15.8f} {fy:15.8f} {fz:15.8f}")

		

		return "\n".join(lines)
