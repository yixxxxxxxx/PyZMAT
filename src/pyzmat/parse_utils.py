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




	
	### Functions for parsing orca output
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
	def parse_orca_output_depr(orcarpt_path: str):
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

	def parse_orca_input(filepath: str):
		"""
		Parse an ORCA input using a gzmt Z-matrix and a %GEOM CONSTRAINTS block.

		Returns
		-------
		zmat : List[List[ symbol:str, bond:Optional[float], angle:Optional[float], dihedral:Optional[float] ]]
		zmat_conn : List[Tuple[ symbol:str, b_ref:Optional[int], a_ref:Optional[int], d_ref:Optional[int] ]]
			All reference indices are 0-based (None where absent).
		constraints : dict
			{"bonds":[(row_idx, None)], "angles":[(row_idx, None)], "dihedrals":[(row_idx, None)]}
			Indices are 0-based and refer to the Z-matrix row being fixed; value None => use current zmat value.
		"""
		# ---------- helpers ----------
		def clean(s: str) -> str:
			return s.strip()

		def is_blank_or_comment(s: str) -> bool:
			t = s.strip()
			return (not t) or t.startswith("#")  # tolerate your '#...' headers

		def to_float(tok: str) -> float:
			# Tolerate Fortran 'D' exponents if they appear
			return float(tok.replace('d', 'E').replace('D', 'E'))

		def wrap_dihedral(x: float) -> float:
			# Normalise into (-180, 180]
			while x <= -180.0:
				x += 360.0
			while x > 180.0:
				x -= 360.0
			return x

		# ---------- read file ----------
		with open(filepath, "r", encoding="utf-8") as f:
			lines = f.readlines()

		# ---------- locate gzmt block ----------
		gzmt_start = None
		for i, ln in enumerate(lines):
			t = ln.strip().lower()
			if t.startswith("*") and "gzmt" in t.split():
				gzmt_start = i
				break
		if gzmt_start is None:
			raise ValueError("Could not find '* gzmt' line.")

		gzmt_end = None
		for i in range(gzmt_start + 1, len(lines)):
			if lines[i].strip() == "*":
				gzmt_end = i
				break
		if gzmt_end is None:
			raise ValueError("Could not find terminating '*' after gzmt block.")

		z_lines = [ln for ln in lines[gzmt_start + 1:gzmt_end] if not is_blank_or_comment(ln)]

		# ---------- parse gzmt block ----------
		zmat: List[List[object]] = []
		zmat_conn: List[Tuple[object, Optional[int], Optional[int], Optional[int]]] = []

		for i, raw in enumerate(z_lines):
			toks = raw.split()
			sym = toks[0]

			if i == 0:
				# first atom: just the symbol
				zmat.append([sym, None, None, None])
				zmat_conn.append((sym, None, None, None))
			elif i == 1:
				# sym, b_ref(1-based), b_val
				b_ref = int(toks[1]) - 1
				b_val = to_float(toks[2])
				zmat.append([sym, b_val, None, None])
				zmat_conn.append((sym, b_ref, None, None))
			elif i == 2:
				# sym, b_ref, b_val, a_ref, a_val
				b_ref = int(toks[1]) - 1
				b_val = to_float(toks[2])
				a_ref = int(toks[3]) - 1
				a_val = to_float(toks[4])
				zmat.append([sym, b_val, a_val, None])
				zmat_conn.append((sym, b_ref, a_ref, None))
			else:
				# sym, b_ref, b_val, a_ref, a_val, d_ref, d_val
				b_ref = int(toks[1]) - 1
				b_val = to_float(toks[2])
				a_ref = int(toks[3]) - 1
				a_val = to_float(toks[4])
				d_ref = int(toks[5]) - 1
				d_val = wrap_dihedral(to_float(toks[6]))
				zmat.append([sym, b_val, a_val, d_val])
				zmat_conn.append((sym, b_ref, a_ref, d_ref))

		# ---------- locate %GEOM CONSTRAINTS block ----------
		# We assume: %GEOM ... CONSTRAINTS ... END  (then later another END for %GEOM)
		geom_start = None
		for i, ln in enumerate(lines):
			if ln.strip().lower().startswith("%geom"):
				geom_start = i
				break

		constraints_block_lines: List[str] = []
		if geom_start is not None:
			# find CONSTRAINTS within %GEOM, then gather until the next 'END'
			i = geom_start + 1
			while i < len(lines):
				t = lines[i].strip()
				if t.lower() == "constraints":
					# collect following lines until a line 'END'
					j = i + 1
					while j < len(lines):
						tj = lines[j].strip()
						if tj.lower() == "end":
							break
						if not is_blank_or_comment(tj):
							constraints_block_lines.append(lines[j])
						j += 1
					break
				# stop if we leave %GEOM by hitting its END
				if t.lower() == "end":
					break
				i += 1

		# ---------- parse constraints (already 0-indexed as per your note) ----------
		bonds_c: List[Tuple[int, Optional[float]]] = []
		ang_c:   List[Tuple[int, Optional[float]]] = []
		dih_c:   List[Tuple[int, Optional[float]]] = []

		def parse_braced(line: str) -> Optional[List[str]]:
			# Expect something like: { D 6 5 1 0 C }
			s = line.strip()
			if not s.startswith("{") or not s.endswith("}"):
				return None
			inner = s[1:-1].strip()
			# split by whitespace
			return [tok for tok in inner.split() if tok]

		# quick accessor for row's refs
		def row_refs(row: int):
			_, jb, ka, ld = zmat_conn[row]
			return jb, ka, ld

		for ln in constraints_block_lines:
			toks = parse_braced(ln)
			if not toks:
				continue
			typ = toks[0].upper()
			# minimal validation
			if typ not in ("B", "A", "D"):
				continue
			if len(toks) < 6:
				continue
			# ORCA constraints in your setup are 0-indexed already
			try:
				i_idx = int(toks[1])  # row / anchor atom in z-matrix order
				j_idx = int(toks[2])
				k_idx = int(toks[3])
				l_idx = int(toks[4])
			except ValueError:
				continue  # skip malformed

			# Ensure in range
			if not (0 <= i_idx < len(zmat_conn)):
				continue

			jb, ka, ld = row_refs(i_idx)

			if typ == "B":
				# row i has a bond DOF if jb is not None; match the referenced partner
				if jb is not None and jb == j_idx:
					bonds_c.append((i_idx, None))
			elif typ == "A":
				if jb is not None and ka is not None and jb == j_idx and ka == k_idx:
					ang_c.append((i_idx, None))
			elif typ == "D":
				if jb is not None and ka is not None and ld is not None and jb == j_idx and ka == k_idx and ld == l_idx:
					dih_c.append((i_idx, None))

		constraints = {"bonds": bonds_c, "angles": ang_c, "dihedrals": dih_c}
		return zmat, zmat_conn, constraints
	
	# --------------------------- small helpers (unchanged) ---------------------------
	@staticmethod
	def _idx_or_none(tok: str) -> Optional[int]:
		i = int(tok)
		return None if i == 0 else (i - 1)

	@staticmethod
	def _strip_inputfile_prefix(line: str) -> str:
		"""Lines inside INPUT FILE are often like: '| 41>   text'. Strip the left prefix."""
		# Remove an initial pipe + counter prefix if present.
		m = re.match(r'^\s*\|\s*\d+>\s*(.*)$', line)
		return m.group(1) if m else line.rstrip("\n")

	@staticmethod
	def _is_elem(token: str) -> bool:
		# Very lenient: 1–2 letter element symbol (H, He, C, Si, ...); ORCA accepts others too,
		# but this is good enough for gzmt parsing. You can widen if needed.
		return bool(re.match(r'^[A-Za-z]{1,2}$', token))

	# --------------------------- block finders ---------------------------
	@staticmethod
	def _find_first_inputfile_block(lines: List[str]) -> Optional[Tuple[int, int]]:
		"""Return [start,end) of the FIRST INPUT FILE block delimited by '=' lines."""
		for i in range(len(lines) - 2):
			if EQ_RE.match(lines[i]) and TITLE_INPUTFILE_RE.search(lines[i + 1]) and EQ_RE.match(lines[i + 2]):
				# find the next '=' that closes this block
				end = None
				for j in range(i + 3, len(lines)):
					if EQ_RE.match(lines[j]):
						end = j
						break
				return (i, len(lines) if end is None else end)
		return None

	@staticmethod
	def _find_final_xyz_after_stationary_point(lines: List[str]) -> Tuple[int, int]:
		"""
		Find the CARTESIAN COORDINATES (ANGSTROEM) table that appears after the last
		'FINAL ENERGY EVALUATION AT THE STATIONARY POINT' header. Return [start,end).
		As a fallback, use the last 'CARTESIAN COORDINATES (ANGSTROEM)' in the file.
		"""
		# 1) Find last stationary-point banner
		last_banner = None
		banner_re = re.compile(r'FINAL ENERGY EVALUATION AT THE STATIONARY POINT', re.I)
		for i, L in enumerate(lines):
			if banner_re.search(L):
				last_banner = i

		def _coords_block_from(idx0: int) -> Optional[Tuple[int, int]]:
			title_re = re.compile(r'^\s*-+\s*$')
			coords_title = re.compile(r'^\s*CARTESIAN COORDINATES\s*\(ANGSTROEM\)\s*$', re.I)
			# search forward for the title trio (----- / title / -----)
			for i in range(idx0, len(lines) - 2):
				if title_re.match(lines[i]) and coords_title.match(lines[i + 1]) and title_re.match(lines[i + 2]):
					start = i + 3
					end = start
					while end < len(lines):
						L = lines[end].rstrip("\n")
						if not L.strip():
							break
						# typical coord line starts with an element symbol
						m = re.match(r'^\s*([A-Za-z]{1,3})\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$', L)
						if not m:
							break
						end += 1
					return (start, end)
			return None

		if last_banner is not None:
			block = _coords_block_from(last_banner)
			if block:
				return block

		# Fallback: last coordinates table anywhere
		last = None
		title_re = re.compile(r'^\s*-+\s*$')
		coords_title = re.compile(r'^\s*CARTESIAN COORDINATES\s*\(ANGSTROEM\)\s*$', re.I)
		for i in range(len(lines) - 2):
			if title_re.match(lines[i]) and coords_title.match(lines[i + 1]) and title_re.match(lines[i + 2]):
				start = i + 3
				end = start
				while end < len(lines):
					L = lines[end].rstrip("\n")
					if not L.strip():
						break
					m = re.match(r'^\s*([A-Za-z]{1,3})\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$', L)
					if not m:
						break
					end += 1
				last = (start, end)
		if last is None:
			raise ValueError("Could not locate a 'CARTESIAN COORDINATES (ANGSTROEM)' table.")
		return last

	# --------------------------- parsers for new requirement ---------------------------
	@staticmethod
	def _parse_gzmt_zmat_conn_from_inputfile(lines: List[str], start: int, end: int) -> List[Tuple[str, Optional[int], Optional[int], Optional[int]]]:
		"""
		Inside the INPUT FILE block, locate '* gzmt charge mult' and parse the following
		Z-matrix lines to produce zmat_conn [(sym, rb, ra, rd), ...] using zero-based refs.
		Stops at the terminating '*' (end of gzmt block) or when lines stop matching.
		"""
		zconn: List[Tuple[str, Optional[int], Optional[int], Optional[int]]] = []

		# find the '* gzmt' line
		gzmt_line = None
		gzmt_re = re.compile(r'^\*\s*gzmt\b', re.I)
		for i in range(start, end):
			payload = ParseUtils._strip_inputfile_prefix(lines[i]).strip()
			if gzmt_re.match(payload):
				gzmt_line = i
				break
		if gzmt_line is None:
			raise ValueError("No '* gzmt' block found inside the INPUT FILE section.")

		# walk forward, parse each Z-matrix line until the terminating '*'
		for j in range(gzmt_line + 1, end):
			raw = ParseUtils._strip_inputfile_prefix(lines[j]).strip()
			if not raw:
				continue
			if raw.startswith('*'):
				break  # end of gzmt block

			toks = raw.split()
			if not toks or not ParseUtils._is_elem(toks[0]):
				# gzmt block ended (or unexpected line)
				break

			sym = toks[0]
			rb = ra = rd = None

			# tokens layout (variable length):
			# 1: sym
			# 2-3: idxB, valB
			# 4-5: idxA, valA
			# 6-7: idxD, valD
			# 0 means "no reference"
			if len(toks) >= 3:
				try:
					rb = ParseUtils._idx_or_none(toks[1])
				except ValueError:
					rb = None
			if len(toks) >= 5:
				try:
					ra = ParseUtils._idx_or_none(toks[3])
				except ValueError:
					ra = None
			if len(toks) >= 7:
				try:
					rd = ParseUtils._idx_or_none(toks[5])
				except ValueError:
					rd = None

			zconn.append((sym, rb, ra, rd))

		if not zconn:
			raise ValueError("Found '* gzmt' but no parsable Z-matrix lines.")
		return zconn

	@staticmethod
	def _parse_final_atoms(lines: List[str]) -> Atoms:
		"""
		Build an ASE Atoms from the final 'CARTESIAN COORDINATES (ANGSTROEM)' table
		that follows the stationary-point banner.
		"""
		start, end = ParseUtils._find_final_xyz_after_stationary_point(lines)
		symbols: List[str] = []
		positions: List[Tuple[float, float, float]] = []
		row_re = re.compile(r'^\s*([A-Za-z]{1,3})\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$')
		for k in range(start, end):
			m = row_re.match(lines[k])
			if not m:
				break
			symbols.append(m.group(1))
			x = float(m.group(2)); y = float(m.group(3)); z = float(m.group(4))
			positions.append((x, y, z))
		if not symbols:
			raise ValueError("Final coordinates table located, but no rows parsed.")
		return Atoms(symbols=symbols, positions=positions)

	# --------------------------- existing energy/forces parsers (unchanged) ---------------------------
	@staticmethod
	def _parse_energy(lines: List[str]) -> float:
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
		headers = []
		for i in range(len(lines) - 2):
			if DASH_RE.match(lines[i]) and TITLE_GRAD_RE.match(lines[i + 1]) and DASH_RE.match(lines[i + 2]):
				headers.append(i)
		if not headers:
			raise ValueError("No 'CARTESIAN GRADIENT' block found.")
		start = headers[-1] + 3
		rows = []
		seen_any = False
		for L in lines[start:]:
			if seen_any and DASH_RE.match(L):
				break
			m = GRAD_LINE_RE.match(L)
			if m:
				seen_any = True
				idx = int(m.group(1))
				fx = float(m.group(2)); fy = float(m.group(3)); fz = float(m.group(4))
				rows.append((idx, fx, fy, fz))
				continue
			if not seen_any:
				if L.strip() == "" or re.match(r'^\s*\d+\s+[A-Za-z]+\s*:\s*$', L):
					continue
				continue
			if L.strip() == "":
				continue
			break
		if not rows:
			raise ValueError("Gradient block header found, but no gradient rows parsed.")
		rows.sort(key=lambda t: t[0])
		factor = Ha / Bohr
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

		Behaviour (updated):
		  - zmat_conn: parsed from the **top** INPUT FILE → '* gzmt …' block.
		  - atoms: parsed from the **final** 'CARTESIAN COORDINATES (ANGSTROEM)'
		    that appears after 'FINAL ENERGY EVALUATION AT THE STATIONARY POINT'.
		  - zmat: built via ZmatUtils.atoms_2_zmat(atoms, zmat_conn).
		  - constraints: parsed from the same (first) INPUT FILE block if present.
		  - Energy & forces: unchanged (last reported).
		"""
		with open(orcarpt_path, 'r', encoding='utf-8', errors='ignore') as f:
			lines = f.readlines()

		# INPUT FILE (first) → constraints + gzmt connectivity
		inp_bounds = ParseUtils._find_first_inputfile_block(lines)
		if inp_bounds is None:
			raise ValueError("INPUT FILE block not found at the top of the output.")
		ifirst, ilast = inp_bounds

		# constraints from the same first INPUT FILE block
		constraints = ParseUtils._parse_constraints_from_inputfile(lines, ifirst, ilast)

		# zmat_conn from '* gzmt'
		zmat_conn = ParseUtils._parse_gzmt_zmat_conn_from_inputfile(lines, ifirst, ilast)

		# final xyz → ASE Atoms
		atoms = ParseUtils._parse_final_atoms(lines)

		# zmat from Atoms + connectivity
		zmat = ZmatUtils.atoms_2_zmat(atoms, zmat_conn)

		# energy & forces from the usual places
		energy_eV = ParseUtils._parse_energy(lines)
		forces_eV_per_A = ParseUtils._parse_forces(lines)

		return zmat, zmat_conn, constraints, energy_eV, forces_eV_per_A