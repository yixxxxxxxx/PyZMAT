# src/zmatopt/parse_utils.py

import ast
import re
import numpy as np
from ase import Atoms
from ase.units import Bohr, Ha
from typing import List, Tuple, Optional

from .constraints import Constraints
from .zmat_utils import ZmatUtils

# --------------------------- regex helpers ---------------------------
_FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_INT   = r'[+-]?\d+'

# INTERNAL COORDINATES lines like:
# "C	  2   1   0	 1.380335007249   105.11267024	 0.00000000"
COORD_LINE_RE = re.compile(
	rf'^\s*([A-Za-z]+)\s+({_INT})\s+({_INT})\s+({_INT})\s+'
	rf'({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$'
)

# Energy header (take the last occurrence)
ENERGY_RE = re.compile(
	r'FINAL\s+SINGLE\s+POINT\s+ENERGY.*?(' + _FLOAT + r')',
	re.IGNORECASE
)

# CARTESIAN GRADIENT header trio
DASH_RE = re.compile(r'^\s*-{2,}\s*$')
TITLE_GRAD_RE = re.compile(r'^\s*CARTESIAN\s+GRADIENT\s*$', re.IGNORECASE)

# Gradient lines like: "   1   C   :	0.000307233   -0.000390900	0.002737333"
GRAD_LINE_RE = re.compile(
	r'^\s*(\d+)\s+[A-Za-z]+\s*:\s*'
	r'(' + _FLOAT + r')\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')\s*$'
)

# INPUT FILE block delimiters
EQ_RE  = re.compile(r'^\s*=+\s*$')
TITLE_INPUTFILE_RE = re.compile(r'input file', re.IGNORECASE)
NUM_RE = re.compile(rf'^{_FLOAT}$', re.IGNORECASE)

# INTERNAL COORDS block delimiters
DASH5_RE = re.compile(r'^\s*-{5,}\s*$')
TITLE_INTERNAL_RE = re.compile(r'^\s*INTERNAL COORDINATES\s*\(ANGSTROEM\)\s*$', re.IGNORECASE)



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
		  - optimised Cartesian coords to ASE Atoms
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
		Parse a Gaussian .log optimised run, extracting:
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




	
	
	# --------------------------- small helpers ---------------------------
	@staticmethod
	def _idx_or_none(tok: str) -> Optional[int]:
		"""ORCA uses 0 to mean 'no reference'. Convert to None, else zero-index."""
		i = int(tok)
		return None if i == 0 else (i - 1)

	@staticmethod
	def _split_brace_payload(line: str) -> Tuple[Optional[str], List[str]]:
		"""Extract {...} payload and split. Returns (kind, tokens_after_kind)."""
		m = re.search(r'\{([^}]*)\}', line)
		if not m:
			return None, []
		toks = m.group(1).strip().split()
		if not toks:
			return None, []
		kind = toks[0].upper()
		return kind, toks[1:]

	@staticmethod
	def _parse_constraint_line(kind: str, toks: List[str]) -> Optional[Tuple[str, int, Optional[float]]]:
		"""Return (kind, zero_based_first_index, value_or_None) for B/A/D lines."""
		if kind == 'B':
			need = 2
		elif kind == 'A':
			need = 3
		elif kind == 'D':
			need = 4
		else:
			return None
	
		numeric = [t for t in toks if NUM_RE.match(t)]
		if len(numeric) < need:
			return None
	
		idx_tokens  = numeric[:need]						# the integer indices
		value_token = numeric[need] if len(numeric) > need else None
	
		try:
			idx_ints = [int(float(t)) for t in idx_tokens]
		except ValueError:
			return None
	
		# Use the FIRST index; convert to zero-based.
		cons_index = idx_ints[0] - 1
	
		cons_value: Optional[float] = None
		if value_token is not None:
			try:
				cons_value = float(value_token)
			except ValueError:
				pass
	
		return (kind, cons_index, cons_value)
	
	
	# --------------------------- block finders ---------------------------
	@staticmethod
	def _find_last_internal_block(lines: List[str]) -> Tuple[int, int]:
		"""Return [start,end) line indices of the last INTERNAL COORDINATES block."""
		headers = []
		for i in range(len(lines) - 2):
			if DASH5_RE.match(lines[i]) and TITLE_INTERNAL_RE.match(lines[i + 1]) and DASH5_RE.match(lines[i + 2]):
				headers.append(i)
		if not headers:
			raise ValueError("No 'INTERNAL COORDINATES (ANGSTROEM)' block found.")
		hdr = headers[-1]
		start = hdr + 3
		end = start
		while end < len(lines) and COORD_LINE_RE.match(lines[end]):
			end += 1
		return start, end
	
	def _find_last_inputfile_block(lines: List[str]) -> Optional[Tuple[int, int]]:
		"""Return [start,end) line indices of the last INPUT FILE block delimited by '=' lines."""
		bounds = []
		for i in range(len(lines) - 2):
			if EQ_RE.match(lines[i]) and TITLE_INPUTFILE_RE.search(lines[i + 1]) and EQ_RE.match(lines[i + 2]):
				# find the next '=' that closes this block
				end = None
				for j in range(i + 3, len(lines)):
					if EQ_RE.match(lines[j]):
						end = j
						break
				bounds.append((i, len(lines) if end is None else end))
		return bounds[-1] if bounds else None
	
	
	# --------------------------- block parsers ---------------------------

	@staticmethod
	def _parse_internal_block(lines: List[str], start: int, end: int):
		"""Build zmat and zmat_conn from INTERNAL COORDS lines[start:end]."""
		zmat: List[List[object]] = []
		zmat_conn: List[Tuple[str, Optional[int], Optional[int], Optional[int]]] = []
		for L in lines[start:end]:
			m = COORD_LINE_RE.match(L)
			if not m:
				break
			sym, ib, ia, idh, cb, ca, cd = m.groups()
			rb  = ParseUtils._idx_or_none(ib)
			ra  = ParseUtils._idx_or_none(ia)
			rd  = ParseUtils._idx_or_none(idh)
	
			bval = float(cb)
			aval = float(ca)
			dval = float(cd)
	
			zmat.append([sym,
						 None if rb is None else bval,
						 None if ra is None else aval,
						 None if rd is None else dval])
			zmat_conn.append((sym, rb, ra, rd))
		if not zmat:
			raise ValueError("Found INTERNAL COORDINATES header, but no parsable lines.")
		return zmat, zmat_conn

	@staticmethod
	def _parse_constraints_from_inputfile(lines: List[str], start: int, end: int) -> Constraints:
		"""Parse the CONSTRAINTS section inside the INPUT FILE block, if present."""
		# locate 'constraints' header
		cons_header = None
		for k in range(start, end):
			if re.search(r'\bconstraints\b', lines[k], re.IGNORECASE):
				cons_header = k
				break
		if cons_header is None:
			return Constraints(bonds=[], angles=[], dihedrals=[])
	
		bonds: List[Tuple[int, Optional[float]]] = []
		angles: List[Tuple[int, Optional[float]]] = []
		diheds: List[Tuple[int, Optional[float]]] = []
	
		for L in lines[cons_header + 1:end]:
			if '{' not in L or '}' not in L:
				if L.strip() == '' or re.match(r'^\s*\|\s*\d+>', L):
					continue
				else:
					break  # leave constraints section heuristically
	
			kind, toks = ParseUtils._split_brace_payload(L)
			if not kind or not toks:
				continue
			parsed = ParseUtils._parse_constraint_line(kind, toks)
			if not parsed:
				continue
			knd, idx, val = parsed
			if knd == 'B':
				bonds.append((idx, val))
			elif knd == 'A':
				angles.append((idx, val))
			elif knd == 'D':
				diheds.append((idx, val))
	
		return Constraints(bonds=bonds, angles=angles, dihedrals=diheds)

	@staticmethod	
	def _parse_energy(lines: List[str]) -> float:
		"""Return the last reported final single-point energy in eV."""
		energy_eV = None
		for L in lines:
			m = ENERGY_RE.search(L)
			if m:
				energy_eV = float(m.group(1)) * Ha
		if energy_eV is None:
			raise ValueError("Final single point energy not found.")
		return energy_eV

	
	@staticmethod
	def _parse_forces(lines: List[str]) -> List[float]:
		"""Return flat 3N forces (eV/Å) from the last CARTESIAN GRADIENT block."""
		headers = []
		for i in range(len(lines) - 2):
			if DASH_RE.match(lines[i]) and TITLE_GRAD_RE.match(lines[i + 1]) and DASH_RE.match(lines[i + 2]):
				headers.append(i)
		if not headers:
			raise ValueError("No 'CARTESIAN GRADIENT' block found.")
		
		start = headers[-1] + 3  # line after the header trio
		
		rows = []
		seen_any = False
		for L in lines[start:]:
			# Stop if we hit the next dashed separator *after* we've started reading rows
			if seen_any and DASH_RE.match(L):
				break
		
			m = GRAD_LINE_RE.match(L)
			if m:
				seen_any = True
				idx = int(m.group(1))      # 1-based atom index
				fx = float(m.group(2))     # Ha/Bohr
				fy = float(m.group(3))
				fz = float(m.group(4))
				rows.append((idx, fx, fy, fz))
				continue
		
			# If we haven't started, tolerate blanks/junk and keep scanning
			if not seen_any:
				if L.strip() == "":
					continue
				# Also tolerate label lines or spacing artifacts
				if re.match(r'^\s*\d+\s+[A-Za-z]+\s*:\s*$', L):
					# rare case: wrapped lines — keep scanning
					continue
				# otherwise just keep scanning until first match
				continue
		
			# Once rows have started, a non-matching non-blank typically ends the table
			if L.strip() == "":
				# allow trailing blank; if more content follows it will trip the break above
				continue
			break
		
		if not rows:
			raise ValueError("Gradient block header found, but no gradient rows parsed.")
		
		# Order by atom index and convert units
		rows.sort(key=lambda t: t[0])
		factor = Ha / Bohr  # Ha/Bohr -> eV/Å
		forces = []
		for _, fx, fy, fz in rows:
			forces.extend([fx * factor, fy * factor, fz * factor])
		return forces
	
	
	# --------------------------- public API ---------------------------
	@staticmethod
	def parse_orca_output(orcarpt_path: str):
		"""
		Parse an ORCA output file and return:
			zmat, zmat_conn, constraints, energy_eV, forces_eV_per_A
	
		Behaviour:
		  - Uses the **last** 'INTERNAL COORDINATES (ANGSTROEM)' table.
		  - Uses the **last** 'INPUT FILE' → 'CONSTRAINTS' section if present.
		  - Energy: last 'FINAL SINGLE POINT ENERGY ...' (eV).
		  - Forces: last 'CARTESIAN GRADIENT' block, flattened 3N array (eV/Å).
		  - All references are **zero-indexed**; printed 0 → None (for zmat_conn).
		  - Constraints use the **first** index on each B/A/D line, zero-indexed;
			an extra numeric token is interpreted as the fixed value.
		"""
		with open(orcarpt_path, 'r', encoding='utf-8', errors='ignore') as f:
			lines = f.readlines()
	
		# Z-matrix + connectivity
		istart, iend = ParseUtils._find_last_internal_block(lines)
		zmat, zmat_conn = ParseUtils._parse_internal_block(lines, istart, iend)
	
		# Constraints (optional)
		inp_bounds = ParseUtils._find_last_inputfile_block(lines)
		if inp_bounds is not None:
			cstart, cend = inp_bounds
			constraints = ParseUtils._parse_constraints_from_inputfile(lines, cstart, cend)
		else:
			constraints = Constraints(bonds=[], angles=[], dihedrals=[])
	
		# Energy & forces
		energy_eV = ParseUtils._parse_energy(lines)
		forces_eV_per_A = ParseUtils._parse_forces(lines)
	
		return zmat, zmat_conn, constraints, energy_eV, forces_eV_per_A
