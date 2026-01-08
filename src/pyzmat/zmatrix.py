from ase import Atoms
from ase.io import read, write
from ase.geometry import *
from ase.constraints import *
from ase.units import *


#from ase.optimize.bfgslinesearch import BFGSLineSearch
#from ase.optimize import BFGS
#from psi4 import geometry
#import optking


import time

import numpy as np

from .zmat_utils import ZmatUtils
from .print_utils import PrintUtils
from .constraints import Constraints
from .parse_utils import ParseUtils

import json

from typing import Tuple, Optional

import io
from contextlib import redirect_stdout

from typing import List, Tuple, Optional, Dict

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

class ZMatrix:

    def __init__(self, zmat, zmat_conn, constraints = None, name = None, energy = None, forces = None, hessian = None, forces_cart = None, hessian_cart = None):

        self.name = name if name else "unnamed molecule"
        self.zmat = zmat
        self.zmat_conn = zmat_conn
        self._constraints = constraints if constraints else Constraints()  # Default to empty constraints
        changes = self.apply_constraints_to_zmat()

        
        self.b_matrix = None
        self.energy = energy if energy is not None else None
        self.forces = forces if forces is not None else None
        self.hessian = hessian if hessian is not None else None
        self.forces_cart = forces_cart if forces_cart is not None else None
        self.hessian_cart = hessian_cart if hessian_cart is not None else None
        self.ase_constraints = self._get_ase_constraints()
        
        self.iteration = 0

    def __repr__(self):
        return f"ZMatrix({len(self.zmat_conn)} atoms, {self.constraints})"
    

    ## Class methods for loading from various file types #########################################################################################################

    @classmethod
    def load_from_gaussian(cls, filename: str) -> "ZMatrix":
        zmat, zmat_conn, constraints = ParseUtils.parse_gaussian_input(filename)
        obj = cls(zmat = zmat, zmat_conn = zmat_conn, constraints = constraints, name = filename)
        return obj

    @classmethod
    def load_from_orca_output(cls, filename: str) -> "ZMatrix":
        zmat, zmat_conn, constraints, energy, forces_cart = ParseUtils.parse_orca_output(filename)
        B = ZmatUtils.get_B_matrix(zmat, zmat_conn)
        forces = (B @ forces_cart.flatten()).flatten()
        obj = cls(zmat = zmat, zmat_conn = zmat_conn, constraints = constraints, energy = energy, forces = forces, forces_cart = forces_cart, name = filename)
        return obj

    @classmethod
    def load_from_orca_input(cls, filename: str) -> "ZMatrix":
        zmat, zmat_conn, constraints = ParseUtils.parse_orca_input(filename)
        obj = cls(zmat = zmat, zmat_conn = zmat_conn, constraints = constraints, name = filename)
        return obj
        
    @classmethod
    def load_json(cls, filename: str, load_hessian = False) -> "ZMatrix":
        """
        Load a ZMatrix object (and its last forces/energy/hessian, if present)
        from a JSON file produced by dump_json().
        The .json files dumped by pyzmat.ZMatrix has name_ZMatrix.json by default. 
        """
        with open(filename, "r") as f:
            state = json.load(f)

        zmat	  = [list(row) for row in state["zmat"]]
        zmat_conn = [tuple(item) for item in state["zmat_conn"]]

        cons = state["constraints"]
        constraints = Constraints(
            bonds	 = [tuple(item) for item in cons["bonds"]],
            angles	= [tuple(item) for item in cons["angles"]],
            dihedrals = [tuple(item) for item in cons["dihedrals"]],
        )

        obj = cls(zmat = zmat,
                  zmat_conn = zmat_conn,
                  constraints = constraints,
                  name = state.get("name"))

        if state.get("forces") is not None:
            obj.forces = np.array(state["forces"], dtype = float)
        if state.get("energy") is not None:
            obj.energy = float(state["energy"])
        if load_hessian is True:
            if state.get("hessian") is not None:
                obj.hessian = np.array(state["hessian"], dtype = float)

        return obj
    
    @classmethod
    def load_pickle(cls, filename: str) -> "ZMatrix":
        """Load an instance back (only for trusted files!)."""
        import pickle
        
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("Pickle file did not contain a ZMatrix instance")
        return obj

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        self._constraints = value
        changes = self.apply_constraints_to_zmat()
        self.ase_constraints = self._get_ase_constraints()

    def clear_constraints(self):
        self._constraints = Constraints()
        self.ase_constraints = self._get_ase_constraints()

    def apply_constraints_to_zmat(self, *,
    normalise_angles: bool = True,
    strict_connectivity: bool = True,
    ) -> list[tuple]:
        """
        Apply *supplied* constraint values (not None) directly into self.zmat.
    
        Parameters
        ----------
        normalise_angles : bool
            If True, wrap dihedrals to [0, 360) and bend angles to [0, 180].
        strict_connectivity : bool
            If True, raise if a constrained coord doesn't exist in this z-matrix
            line (e.g. angle/dihedral is None because it's the first/second atom).
    
        Returns
        -------
        changes : list of tuples
            [(kind, idx, old_value, new_value)] for each change made, where
            kind ∈ {"bond", "angle", "dihedral"} and idx is the atom index.
        """
        def _wrap360(x: float) -> float:
            # 0 ≤ angle < 360
            y = float(x) % 360.0
            # Keep 360 exactly as 0 for neatness
            return 0.0 if abs(y - 360.0) < 1e-12 else y
    
        def _bend_0_180(x: float) -> float:
            # Map to [0, 180] using 0–360 periodicity
            y = float(x) % 360.0
            return 360.0 - y if y > 180.0 else y
    
        changes: list[tuple] = []
    
        # Bonds: (index, value_in_Å)
        for idx, val in getattr(self._constraints, "bonds", []):
            if val is None:
                continue  # user didn't supply a fixed numeric value
            if self.zmat[idx][1] is None and strict_connectivity:
                raise ValueError(f"Bond for atom {idx} is undefined in z-matrix connectivity.")
            old = self.zmat[idx][1]
            new = float(val)
            self.zmat[idx][1] = new
            changes.append(("bond", idx, old, new))
    
        # Angles: (index, value_in_degrees)
        for idx, val in getattr(self._constraints, "angles", []):
            if val is None:
                continue
            if self.zmat[idx][2] is None and strict_connectivity:
                raise ValueError(f"Angle for atom {idx} is undefined in z-matrix connectivity.")
            old = self.zmat[idx][2]
            new = float(val)
            if normalise_angles:
                new = _bend_0_180(new)
            self.zmat[idx][2] = new
            changes.append(("angle", idx, old, new))
    
        # Dihedrals: (index, value_in_degrees)
        for idx, val in getattr(self._constraints, "dihedrals", []):
            if val is None:
                continue
            if self.zmat[idx][3] is None and strict_connectivity:
                raise ValueError(f"Dihedral for atom {idx} is undefined in z-matrix connectivity.")
            old = self.zmat[idx][3]
            new = float(val)
            if normalise_angles:
                new = _wrap360(new)
            self.zmat[idx][3] = new
            changes.append(("dihedral", idx, old, new))
    
        # Rebuild ASE constraints so that any None-values that default to zmat now
        # inherit the updated numbers consistently.
        self.ase_constraints = self._get_ase_constraints()
    
        return changes

        


    def attach_calculator(self, model, model_size = 'large', gpu = False):
        """Attach an MLIP as an ASE calculator to the ZMatrix object. Only supports MACE-off23 ('mace-off') and AIMNet2 ('aimnet2') at the moment."""
        if model not in ['mace-off', 'aimnet2', 'mace-omol']:
            raise ValueError("Only MACE-OFF ('mace-off'), MACE-omol ('mace-omol'), and AIMNet2 ('aimnet2') are currently supported")
        if model == 'mace-off':
            from mace.calculators import mace_off
            if model_size not in ['small', 'medium', 'large']:
                raise ValueError("Model size must be one of 'small', 'medium', or 'large'")
            if gpu == True:
                self.calculator = mace_off(model = model_size, device='cuda')
            else:
                self.calculator = mace_off(model = model_size)
            self.model_size = model_size
            
        elif model == 'aimnet2':
            if model_size:
                print("Warning: the parameter model_size is unused when model = 'aimnet2'")
            from aimnet.calculators import AIMNet2ASE
            self.calculator = AIMNet2ASE('aimnet2')
            self.model_size = 'NaN'

        elif model == 'mace-omol':
            if model_size:
                print("Warning: the parameter model_size is unused when model = 'mace-omol'")
            from mace.calculators import mace_omol 
            if gpu == True:
                self.calculator = mace_omol(model = 'extra_large', device = 'cuda')
            else:
                self.calculator = mace_omol(model = 'extra_large', device = 'cpu')

            self.model_size = model_size

        self.model = model

    ## Calculate tensors of coordinate derivatives from pyzmat.ZmatUtils #######################################################

    def _get_B_matrix(self):
        B = ZmatUtils.get_B_matrix(self.zmat, self.zmat_conn)
        return B

    def _get_K_tensor(self):
        K = ZmatUtils.get_curvature_tensor_numba(self.zmat, self.zmat_conn)
        return K

    ## Extract ase constraints from pyzmat.Constraint object ##################################################################

    def _get_ase_constraints(self):
        # Build ASE constraints using the provided (or default) values.
        bonds = []
        for index, val in self._constraints.bonds:
            bond_length = val if val is not None else self.zmat[index][1]
            # For bonds, use connectivity index from zmat_conn: (symbol, bond, angle, dihedral)
            j = self.zmat_conn[index][1]
            bond = [bond_length, [index, j]]
            bonds.append(bond)
        angles = []
        for index, val in self._constraints.angles:
            bond_angle = val if val is not None else self.zmat[index][2]
            j = self.zmat_conn[index][1]
            k = self.zmat_conn[index][2]
            angle = [bond_angle, [index, j, k]]
            angles.append(angle)
        dihedrals = []
        for index, val in self._constraints.dihedrals:
            dihedral_angle = val if val is not None else self.zmat[index][3]
            j = self.zmat_conn[index][1]
            k = self.zmat_conn[index][2]
            l = self.zmat_conn[index][3]
            dihedral = [dihedral_angle, [index, j, k, l]]
            dihedrals.append(dihedral)
        kwargs = {}
        if bonds:
            kwargs["bond"] = bonds
        if angles:
            kwargs["angle"] = angles
        if dihedrals:
            kwargs["dihedrals_deg"] = dihedrals
        c = FixInternals(**kwargs)
        return c

    ## Functions for transforming ZMatrix format into a few useful formats (AIMNet2 and ASE) ###################################
    @staticmethod
    def get_aimnet_data(atoms):
        """Form the input data format for the standalone aimnet2 calculator (non-ASE). Primarily used for Hessian calculation, rest is done through ASE"""
        data = {
            # N×3 float32 coords
            "coord": [atoms.get_positions().tolist()],
            # N ints
            "numbers": [atoms.get_atomic_numbers().astype(np.int64)],
            # total charge as length-1 float32 array
            "charge": 0,
        }
        # if you need PBC/cell:
        if atoms.get_pbc().any():
            data["cell"] = atoms.get_cell().array.astype(np.float32)
        return data

    def get_atoms(self):
        return ZmatUtils.zmat_2_atoms(self.zmat, self.zmat_conn)

    ## Function for energy ##############################################################################################################

    def get_energy(self):
        if not hasattr(self, "model"):
           raise AttributeError(f"{self.__class__.__name__!r} has no attribute 'model'. Did you forget to run ZMatrix.attach_calculator()?")
        atoms = self.get_atoms()
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        return energy
        
    ## Functions related to calculating and converting forces ######################################################################################

    def get_forces(self):
        if not hasattr(self, "model"):
            raise AttributeError(f"{self.__class__.__name__!r} has no attribute 'model'. Did you forget to run ZMatrix.attach_calculator()?")
        atoms = ZmatUtils.zmat_2_atoms(self.zmat, self.zmat_conn)
        atoms.calc = self.calculator
        forces_cart_2d = atoms.get_forces(apply_constraint=False)
        forces_cart = np.array(forces_cart_2d).reshape(-1, 1)
        B = ZmatUtils.get_B_matrix(self.zmat, self.zmat_conn)
        forces = (B @ forces_cart).flatten()
        return forces

    def get_fd_forces(self):
        if not hasattr(self, "calculator"):
            raise AttributeError(f"{self.__class__.__name__!r} has no attribute 'calculator'. Did you forget to run ZMatrix.attach_calculator()?")
        atoms = ZmatUtils.zmat_2_atoms(self.zmat, self.zmat_conn)
        atoms.calc = self.calculator
        forces_cart_2d = atoms.get_forces(apply_constraint=False)
        forces_cart = np.array(forces_cart_2d).reshape(-1, 1)
        B = ZmatUtils.get_fd_B_matrix(self.zmat, self.zmat_conn, 1e-5, 1e-3, 1e-3)
        forces = (B @ forces_cart).flatten()
        return forces

    def get_forces_cart(self):
        if not hasattr(self, "calculator"):
            raise AttributeError(f"{self.__class__.__name__!r} has no attribute 'calculator'. Did you forget to run ZMatrix.attach_calculator()?")
        atoms = ZmatUtils.zmat_2_atoms(self.zmat, self.zmat_conn)
        atoms.calc = self.calculator
        forces_cart_2d = atoms.get_forces(apply_constraint=False)
        forces_cart = np.array(forces_cart_2d).reshape(-1)
        return forces_cart

    def get_jacobian(self, vars):
        """Tied to the unused ZMatrix.optimise() routine"""
        zmat = self._reconstruct_full_z_matrix(vars)
        atoms = ZmatUtils.zmat_2_atoms(zmat, self.zmat_conn)
        atoms.calc = self.calculator
        forces_cart_2d = atoms.get_forces(apply_constraint=False)
        forces_cart = np.array(forces_cart_2d).reshape(-1, 1)
        B_full = ZmatUtils.get_B_matrix(zmat, self.zmat_conn)
        if self.con_ids:
            B_reduced = np.delete(B_full, self.con_ids, axis=0)
        else:
            B_reduced = B_full
        forces = (B_reduced @ forces_cart).flatten()
        jacobian = -1 * forces
        for i, element in enumerate(jacobian):
            if i == 2:
                element *= np.pi/180
        return jacobian

    ## Functions related to calculating and converting Hessian matrices ####################################################################

    def get_hess_cart(self):
        if not hasattr(self, "model"):
            raise AttributeError(f"{self.__class__.__name__!r} has no attribute 'model'. Did you forget to run ZMatrix.attach_calculator()?")
        if self.model == 'mace-off' or self.model == 'mace-omol':
            H_cart = self.calculator.get_hessian(atoms = self.get_atoms())
            
        elif self.model == 'aimnet2':
            from aimnet.calculators import AIMNet2Calculator
            
            data = ZMatrix.get_aimnet_data(self.get_atoms())
            calc = AIMNet2Calculator(model="aimnet2")
            out = calc.eval(data, forces = True, hessian = True)
            H_cart = out['hessian']

        N = H_cart.shape[0]
        return H_cart.reshape((N, N))


    @staticmethod
    def _get_temp_forces(zmat, zmat_conn, calculator):

        atoms = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
        atoms.calc = calculator
        forces_cart_2d = atoms.get_forces(apply_constraint=False)
        forces_cart = np.array(forces_cart_2d).reshape(-1, 1)
        B = ZmatUtils.get_B_matrix(zmat, zmat_conn)
        forces = (B @ forces_cart).flatten()
        return forces
    
    def _hessian_column(self, idx, delta, valid_indices):
        """
        Central difference of *forces* with an explicit sign-flip so that
        H =  ∂²E / ∂qₖ ∂qⱼ
        """
        if not hasattr(self, "calculator"):
            raise AttributeError(f"{self.__class__.__name__!r} has no attribute 'calculator'. Did you forget to run ZMatrix.attach_calculator()?")
        i, j = valid_indices[idx]
    
        # +δ -------------------------------------------------------------
        z_plus	   = copy.deepcopy(self.zmat)
        z_plus[i][j] += delta
        F_plus	   = self._get_temp_forces(z_plus, self.zmat_conn, self.calculator)
    
        # −δ -------------------------------------------------------------
        z_minus	   = copy.deepcopy(self.zmat)
        z_minus[i][j] -= delta
        F_minus	   = self._get_temp_forces(z_minus, self.zmat_conn, self.calculator)
    
        #   F = −∂E/∂q	⇒   dF/dq   =  −∂²E/∂q²
        # so we have to invert the sign once:
        col = -(F_plus - F_minus) / (2.0 * delta)		  #  <-- sign fixed   (NEW)
    
        # if *this* coordinate qⱼ is an angle or torsion, rescale the whole column
        if j in (2, 3):									# 1 = bond, 2 = angle, 3 = torsion
            col *= 180 / np.pi							#  <-- unit fix	 (NEW)
    
        return col
    
    
    def get_full_fd_hessian(self, db, da, dt):
        """
        Symmetric (3N-6) × (3N-6) Hessian obtained from analytic forces,
        computed serially. For testing only.
        """
        deltas = [db, da, dt]
    
        # list of internal-coordinate positions (i, j) that are not None
        valid_indices = [(i, j)
                         for i, row in enumerate(self.zmat)
                         for j in range(1, len(row))
                         if row[j] is not None]
    
        M  = len(valid_indices)
        H  = np.empty((M, M))
    
        # --- build each column in a simple Python loop -------------------
        for k in range(M):
            delta = deltas[valid_indices[k][1] - 1]
            H[:, k] = self._hessian_column(k, delta, valid_indices)
    
        # enforce exact symmetry
        #H = 0.5 * (H + H.T)
    
        # copy already-scaled angle/torsion columns into their mirror rows
        for k, (_, j) in enumerate(valid_indices):
            if j in (2, 3):				# angle or torsion
                H[k, :] = H[:, k]
    
        return H
        

    def get_geom_fd_hessian(self, db, da, dt):
        """Quasi-analytical Hessian from a finite-difference curvature tensor, for testing only"""
        H_cart = np.array(self.get_hess_cart())
        B = self._get_B_matrix()
        K = ZmatUtils.get_fd_curvature_tensor(self.zmat, self.zmat_conn, db, da, dt)

        F_cart = self.get_forces_cart()

        H_min = B @ H_cart @ np.transpose(B)
        H_res = -1 * np.einsum('isp,i->sp', K, F_cart)
        return H_min + H_res
    
    def get_hessian(self):
        """Calculate fully analytical Hessian. Recommended for production use."""
        H_cart = np.array(self.get_hess_cart())
        B = self._get_B_matrix()
        K = self._get_K_tensor()
        F_cart = self.get_forces_cart()

        H_min = B @ H_cart @ np.transpose(B)
        H_res = -1 * np.einsum('isp,i->sp', K, F_cart)
        hessian = H_min + H_res

        self.hessian = hessian
        return hessian

    ## Visualisation ###############################################################################

    def view_ase(self):
        from ase.visualize import view
        return view(self.get_atoms(), viewer = 'x3d')

    ## Optimisation with ASE ########################################################################
    # Not recommended (especially with large molecules), use ORCA extopt instead
    
    def optimise_ase(self, trajectory = None, mode = 'linesearch', fmax = 2.31e-2, calc_hess = False, fix_rototrans = False):
        from ase.optimize.bfgslinesearch import BFGSLineSearch
        from ase.optimize import BFGS
        
        print('Initialising minimisation routine')
        start_tot = time.perf_counter()
        print('Model used:', self.calculator, self.model_size)
        print('Input Z-matrix:')
        print('======================================================================================')
        PrintUtils.print_zmat(self.zmat, self.zmat_conn, self.constraints)
        print('======================================================================================')
        print('Building cartesian molecule from input geometry...')
        atoms = self.get_atoms()
        print('======================================================================================')
        PrintUtils.print_xyz(atoms, comment='Input coordinates of ' + self.name, fmt='%22.15f')
        print('======================================================================================')
        atoms.calc = self.calculator
        del atoms.constraints

        if fix_rototrans:
            from ase.constraints import FixAtoms, FixedLine, FixedPlane
            fix_atom_0 = FixAtoms(indices = [0])
            fix_atom_1 = FixedLine(indices = [1], direction = [1, 0, 0])
            fix_atom_2 = FixedPlane(indices = [2], direction = [0, 0, 1])
            atoms.set_constraint([self.ase_constraints, fix_atom_0, fix_atom_1, fix_atom_2])

        else:
            atoms.set_constraint([self.ase_constraints])

        if mode == 'linesearch':

            dyn = BFGSLineSearch(atoms, trajectory = trajectory, restart = f'{self.name}_linesearch_opt.json')
            print(f'Now beginning ASE BFGS Line Search minimisation routine, convergence threshold: fmax = {fmax}')
            print('--------------------------------------------------------------------------------------')
            start_min = time.perf_counter()
            dyn.run(fmax = fmax)
            end_min = time.perf_counter()
            energy = atoms.get_potential_energy()
            print('--------------------------------------------------------------------------------------')
            print('! ASE minimisation complete. U_tot =', energy * 0.0367493, 'Ha, or', energy, 'eV')
            print('--------------------------------------------------------------------------------------')

        elif mode == 'bfgs':
            dyn = BFGS(atoms, trajectory = trajectory, restart = f'{self.name}_bfgs_opt.json')
            print(f'Now beginning ASE BFGS minimisation routine, convergence threshold: fmax = {fmax}')
            print('--------------------------------------------------------------------------------------')
            start_min = time.perf_counter()
            dyn.run(fmax = fmax)
            end_min = time.perf_counter()
            energy = atoms.get_potential_energy()
            print('--------------------------------------------------------------------------------------')
            print('! ASE minimisation complete. U_tot =', energy * 0.0367493, 'Ha, or', energy, 'eV')
            print('--------------------------------------------------------------------------------------')

        zmat_minimised = ZmatUtils.atoms_2_zmat(atoms, self.zmat_conn)
        atoms_minimised = ZmatUtils.zmat_2_atoms(zmat_minimised, self.zmat_conn)
        atoms_minimised.calc = self.calculator
        
        forces_cart_2d = atoms_minimised.get_forces(apply_constraint=False)
        forces_cart = np.array(forces_cart_2d).reshape(-1, 1)
        self.zmat = zmat_minimised
        print('Optimised Z-matrix:')
        print('======================================================================================')
        PrintUtils.print_zmat(self.zmat, self.zmat_conn, self.constraints)
        print('======================================================================================')
        forces = self.get_forces()

        print('Optimised cartesian coordinates:')		
        print('======================================================================================')
        PrintUtils.print_xyz(atoms_minimised, comment='ASE minimised ' + self.name, fmt='%22.15f')
        print('======================================================================================')
        print('Forces in terms of dU/db [Ha/bohr], dU/da [Ha/rad], and dU/dt [Ha/rad]:')
        print('======================================================================================')
        PrintUtils.print_forces(forces, self.zmat)
        print('======================================================================================')
        if calc_hess:
            print('Calculating Hessian matrix as calc_hess is set to True...')
            hessian = self.get_hessian()
            print('Calculated Hessian:')
            PrintUtils.print_hessian(hessian, self.zmat, constraints = self.constraints, block_size = 5)
            print('======================================================================================')
        print('Routine finished successfully.')

        self.forces = forces
        self.energy = energy

        
        end_tot = time.perf_counter()
        wall_tot = end_tot - start_tot
        wall_min = end_min - start_min
        
        print(f'Total wall time = {wall_tot:.6f} seconds')
        print(f'Minimisation wall time = {wall_min:.6f} seconds')
        
        return zmat_minimised, energy, forces

    ## Setting up optking minimisation ########################################################################
    # Not recommended, use ORCA external opt instead

    def form_psi4_geom(self):
        '''
        Form a psi4.geometry molecule from a Z-matrix and its connectivities. 
        '''
        
        geom_parts = []
        for i, zmat_line in enumerate(self.zmat):
            geom_line = " " + str(zmat_line[0])
            if zmat_line[1] is not None:
                bond_ref = self.zmat_conn[i][1] + 1
                bond_val = zmat_line[1]
                geom_line += ' ' + str(bond_ref) + ' ' + str(bond_val)
            if zmat_line[2] is not None:
                ang_ref = self.zmat_conn[i][2] + 1
                ang_val = zmat_line[2]
                geom_line += ' ' + str(ang_ref) + ' ' + str(ang_val)
            if zmat_line[3] is not None:
                dih_ref = self.zmat_conn[i][3] + 1
                dih_val = zmat_line[3]
                geom_line += ' ' + str(dih_ref) + ' ' + str(dih_val)
            geom_parts.append(geom_line)
        geom_string = "\n".join(geom_parts)
        return geometry(geom_string)


    def get_optking_constraints(self):
        # Build ASE constraints using the provided (or default) values.
        bonds = []
        for index, val in self.constraints.bonds:
            j = self.zmat_conn[index][1]
            bonds.append(j + 1)
            bonds.append(index + 1)
        angles = []
        for index, val in self.constraints.angles:
            j = self.zmat_conn[index][1]
            k = self.zmat_conn[index][2]
            angles.append(k + 1)
            angles.append(j + 1)
            angles.append(index + 1)
        dihedrals = []
        for index, val in self.constraints.dihedrals:
            dihedral_angle = val if val is not None else self.zmat[index][3]
            j = self.zmat_conn[index][1]
            k = self.zmat_conn[index][2]
            l = self.zmat_conn[index][3]
            dihedrals.append(l + 1)
            dihedrals.append(k + 1)
            dihedrals.append(j + 1)
            dihedrals.append(index + 1)
    
        psi4_constraints = {}
        if bonds:
            psi4_constraints["frozen_distance"] = " ".join([str(bond) for bond in bonds])
        if angles:
            psi4_constraints["frozen_bend"] = " ".join([str(ang) for ang in angles])
        if dihedrals:
            psi4_constraints["frozen_dihedral"] = " ".join([str(dih) for dih in dihedrals])
        return psi4_constraints

    def ase_engine(self, atoms, hess_step = False):
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy() / Ha
        forces = atoms.get_forces()
        forces = np.array(forces).flatten() / Ha * Bohr * -1
        hessian = None
        if hess_step:
            hessian = calc.get_hessian(atoms = ZmatUtils.zmat_2_atoms(mol.zmat, mol.zmat_conn))
            hessian = np.array(hessian).reshape(len(forces), len(forces)) / Ha * Bohr
        return energy, forces, hessian

        
    def run_optking(self, geometry, symbols, opt_options, max_iter):

        if self.constraints is not None:
            psi4_constraints = self.get_optking_constraints()
            opt_options.update(psi4_constraints)

        opt = optking.CustomHelper(geometry, opt_options)
        
        print('step', '	  ', 'E [eV]', '	  ', 'fmax [eV/A]')
        
        for step in range(max_iter):

            # Determine whether the analytical Hessian matrix needs to be calculated at a given step

            if opt_options["full_hess_every"] != -1:
                if opt_options["full_hess_every"] == 0 and step == 0:
                    hess_step = True
                    print('Calculating Hessian for step 0 as full_hess_every is set to 0')
                elif opt_options["full_hess_every"] != 0 and step % opt_options["full_hess_every"] == 0:
                    hess_step = True
                    print(f'Calculating Hessian for step {step} as full_hess_every is set to {opt_options["full_hess_every"]}')
                else:
                    hess_step = False
            else:
                hess_step = False

            # Send optimiser parameters into ASE engine to get energy, Cartesian forces, and optionally Cartesian Hessian

            atoms_step = Atoms(symbols = symbols, positions = opt.geom * Bohr)
                
            E, gX, HX = self.ase_engine(atoms_step, hess_step)
            
            print(step, '	  ', E * Ha, '	  ', max(gX) * Ha / Bohr)

            # Send ASE outputs into optimiser to process and take a step
        
            opt.E = E
            opt.gX = gX

            if hess_step:
                opt.HX = HX
            
            opt.compute()
            opt.take_step()
        
            conv = opt.test_convergence()
        
            if conv is True:
                print('--------------------------------------------------------------------------------------')
                print('! OptKing minimisation complete. U_tot =', E, 'Ha, or', E * Ha, 'eV')
                print('--------------------------------------------------------------------------------------')
                break
        else:
            print(f"Minimisation FAILED to converge within {max_iter} steps")
            raise RuntimeError(f"Minimisation FAILED to converge within {max_iter} steps")

        res = opt.summarize_result()
        json_output = opt.close()

        atoms_opt = Atoms(symbols = symbols, positions = res[1] * Bohr)
        
        return atoms_opt

    def optimise_optking(self, attempt_recovery = True, g_convergence = "gau", full_hess_every = -1, opt_coordinates = "internal", intrafrag_hess = "SCHLEGEL", calc_hess = False, max_iter = 500):
        restart_flag = False
        
        opt_options = {"g_convergence": g_convergence, "full_hess_every": full_hess_every, "intrafrag_hess": intrafrag_hess, "opt_coordinates": opt_coordinates}

        print('Initialising minimisation routine')
        start_tot = time.perf_counter()
        print('Model used:', self.calculator, self.model_size)
        print('Input Z-matrix:')
        print('======================================================================================')
        PrintUtils.print_zmat(self.zmat, self.zmat_conn, self.constraints)
        print('======================================================================================')
        print('Building cartesian molecule from input geometry...')
        atoms = self.get_atoms()
        print('======================================================================================')
        PrintUtils.print_xyz(atoms, comment='Input coordinates of ' + self.name, fmt='%22.15f')
        print('======================================================================================')

        geometry = self.form_psi4_geom()
        symbols = [line[0] for line in self.zmat]

        print(f'Now beginning OptKing minimisation routine in {opt_coordinates} coordinates, convergence threshold: {g_convergence}')
        print('--------------------------------------------------------------------------------------')

        try:
            start_min1 = time.perf_counter()
            atoms_opt = self.run_optking(geometry, symbols, opt_options, max_iter)
            end_min1 = time.perf_counter()
        except Exception:
            if (opt_coordinates == "internal" or opt_coordinates == "redundant") and attempt_recovery:
                try:
                    restart_flag = True
                    end_min1 = time.perf_counter()
                    print('Initial minimisation in redundant coordinates has failed catastrophically. Re-attempting optimisation in Cartesian coordinates from starting geometry...')
                    print(f'Now beginning OptKing minimisation routine in cartesian coordinates, convergence threshold: {g_convergence}')
                    print('--------------------------------------------------------------------------------------')
                    opt_options_restart = {"g_convergence": g_convergence, "full_hess_every": full_hess_every, "intrafrag_hess": "SIMPLE", "opt_coordinates": "cartesian"}
                    start_min2 = time.perf_counter()
                    atoms_opt = self.run_optking(geometry, symbols, opt_options_restart, max_iter)
                    end_min2 = time.perf_counter()
                except Exception:
                    raise RuntimeError("OptKing minimisation routine has failed during re-attempt. Please try to minimise with ZMatrix.optimise_ase() instead.")
            
        zmat_minimised = ZmatUtils.atoms_2_zmat(atoms_opt, self.zmat_conn)
        atoms_minimised = ZmatUtils.zmat_2_atoms(zmat_minimised, self.zmat_conn)
        atoms_minimised.calc = self.calculator

        energy = atoms_minimised.get_potential_energy()
        forces_cart_2d = atoms_minimised.get_forces(apply_constraint = False)
        forces_cart = np.array(forces_cart_2d).reshape(-1, 1)
        self.zmat = zmat_minimised
        print('Optimised Z-matrix:')
        print('======================================================================================')
        PrintUtils.print_zmat(self.zmat, self.zmat_conn, self.constraints)
        print('======================================================================================')
        forces = self.get_forces()

        print('Optimised cartesian coordinates:')		
        print('======================================================================================')
        PrintUtils.print_xyz(atoms_minimised, comment='OptKing minimised ' + self.name, fmt='%22.15f')
        print('======================================================================================')
        print('Forces in terms of dU/db [Ha/bohr], dU/da [Ha/rad], and dU/dt [Ha/rad]:')
        print('======================================================================================')
        PrintUtils.print_forces(forces, self.zmat)
        print('======================================================================================')
        if calc_hess:
            print('Calculating Hessian matrix as calc_hess is set to True...')
            hessian = self.get_hessian()
            print('Calculated Hessian:')
            PrintUtils.print_hessian(hessian, self.zmat, constraints = self.constraints, block_size = 5)
            print('======================================================================================')
        print('Routine finished successfully.')

        self.forces = forces
        self.energy = energy

        
        end_tot = time.perf_counter()
        wall_tot = end_tot - start_tot
        wall_min = end_min1 - start_min1
        
        print(f'Total wall time = {wall_tot:.6f} seconds')
        print(f'Minimisation wall time = {wall_min:.6f} seconds')
        if restart_flag == True:
            wall_min2= end_min2 - start_min2
            print('Re-attempt detected:')
            print(f'2nd minimisation wall time = {wall_min2:.6f} seconds')
        
        return zmat_minimised, energy, forces	

        
    ## Functions for saving output ################################################################################

    def dump_json(self, filename = None):
        """
        Save all core data to a JSON file.
        """
        if filename is None:
            filename = f'{self.name}_ZMatrix.json'
        state = {
            "zmat":	  self.zmat,
            "zmat_conn": self.zmat_conn,
            "constraints": {
                "bonds":	 [[idx, val] for idx, val in self.constraints.bonds],
                "angles":	[[idx, val] for idx, val in self.constraints.angles],
                "dihedrals": [[idx, val] for idx, val in self.constraints.dihedrals],
            },
            "energy":  self.energy if self.energy is not None else None,
            "forces":  self.forces.tolist() if self.forces is not None else None,
            "hessian": self.hessian.tolist() if self.hessian is not None else None,
        }
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)


    def save_pickle(self, filename: str):
        """Binary dump of the entire instance (fast & automatic)."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)


    def save_gaussian_com(self, filename, preamble, postamble = None):
        '''
        Saves the current geometry as a gaussian .com file. 
        Preamble should be a docstring containing gaussian settings (%mem, %nprocshared etc) 
        '''
        # Capture the printed Z-matrix
        buf = io.StringIO()
        with redirect_stdout(buf):
            PrintUtils.print_zmat(self.zmat, self.zmat_conn, self.constraints)
        zmat_text = buf.getvalue()

        # Write preamble + zmat + postamble
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(preamble)
                if not preamble.endswith('\n'):
                    f.write('\n')
                f.write(zmat_text)
                if not zmat_text.endswith('\n'):
                    f.write('\n')
                if postamble:
                    f.write(postamble)
                if not zmat_text.endswith('\n'):
                    f.write('\n') 
        except IOError as e:
            print(f"Error writing to {filename}: {e}")

    def save_mace_train_xyz(self, filename):
        '''
        Saves the current geometry as an xyz file formatted for MACE training data.
        '''
        file_content = PrintUtils.print_mace_training_xyz(self.get_atoms(), self.energy, self.forces_cart)
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(file_content)
        except IOError as e:
            print(f"Error writing to {filename}: {e}")

    def save_orca_input(self, filename, *,
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
        charge: int = 0,
        multiplicity: int = 1,
        title_lines: Optional[List[str]] = None,
    ):
        '''
        Saves the current Z-matrix as an ORCA input file.
        '''
        orca_input = PrintUtils.print_orca_input(self.zmat, self.zmat_conn, self.constraints, level_of_theory = level_of_theory, basis_set = basis_set, maxcore_mb = maxcore_mb, nproc = nproc, scf_maxiter = scf_maxiter, scf_conv = scf_conv, geom_maxiter = geom_maxiter, geom_conv = geom_conv, tol_maxg = tol_maxg, tol_rmsg = tol_rmsg, tol_maxd = tol_maxd, tol_rmsd = tol_rmsd, use_symmetry = use_symmetry, charge = charge, multiplicity = multiplicity, title_lines = title_lines)
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(orca_input)
        except IOError as e:
            print(f"Error writing to {filename}: {e}")

    def save_orca_extopt_input(self, filename, wrapper_path, *,
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
    ):
        '''
        Saves the current Z-matrix as an ORCA input file for external optimisation.
        '''
        orca_input = PrintUtils.print_orca_extopt_input(self.zmat, self.zmat_conn, wrapper_path, self.constraints, maxcore_mb = maxcore_mb, nproc = nproc, scf_maxiter = scf_maxiter, scf_conv = scf_conv, geom_maxiter = geom_maxiter, geom_conv = geom_conv, tol_maxg = tol_maxg, tol_rmsg = tol_rmsg, tol_maxd = tol_maxd, tol_rmsd = tol_rmsd, use_symmetry = use_symmetry, charge = charge, multiplicity = multiplicity, title_lines = title_lines)
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(orca_input)
        except IOError as e:
            print(f"Error writing to {filename}: {e}")

    ## Functions for generating conformations for finetuning #######################################################
                
    @staticmethod
    def sobol_nd_in_ranges(n_dim, n_samples, ranges, scramble=True, seed=None):
        from scipy.stats import qmc
    
        """
        Generate a Sobol sequence within user-specified ranges on each dimension.
    
        Parameters
        ----------
        n_dim : int
            Number of dimensions.
        n_samples : int
            Number of Sobol samples to generate.
        ranges : list[tuple[float, float]]
            List of (lower, upper) bounds for each dimension.
            Example: [(0,1), (-5,5), (100,200)]
        scramble : bool, optional
            Whether to use a scrambled Sobol sequence (better uniformity for finite n).
        seed : int, optional
            Random seed for reproducibility (only used if scramble=True).
    
        Returns
        -------
        samples : np.ndarray, shape (n_samples, n_dim)
            Sobol samples scaled to the given ranges.
        """
        if len(ranges) != n_dim:
            raise ValueError("Length of 'ranges' must equal n_dim")
    
        # Initialise Sobol sampler
        sampler = qmc.Sobol(d=n_dim, scramble=scramble, seed=seed)
    
        # Generate points in [0,1]^n
        sobol_unit = sampler.random(n_samples)
    
        # Scale to user ranges
        lower = np.array([lo for lo, hi in ranges])
        upper = np.array([hi for lo, hi in ranges])
        sobol_scaled = qmc.scale(sobol_unit, lower, upper)
    
        return sobol_scaled

    @staticmethod
    def ase_get_connectivity(atoms, bonding_cutoffs = None, print_flag = False):
        from ase.neighborlist import neighbor_list, natural_cutoffs
        '''
        Determines the connectivity matrix of a molecule 
        Based on code by BIT in CrystalSetupPy
        '''
        # Basic info
        N = len(atoms.get_chemical_symbols())
    
        # Generate cutoffs if needed
        if (bonding_cutoffs is None):
            bonding_cutoffs = natural_cutoffs(atoms, mult=1.1)
    
        # Get the neighbours
        at1, at2, = neighbor_list('ij', atoms,				   
                                    bonding_cutoffs, 
                                    self_interaction = False)
        
        # Conenctivity matrix
        connectivity = np.zeros((N, N))
        for i in range(len(at1)):
            connectivity[at1[i]][at2[i]]=1
    
        # Print
        if (print_flag):
            #Header
            print("	",end='')
            for i in range(natms):
                print("%3i " % int(i+1),end='')
    
            for i in range(natms):  # rows
                print("\n%-3i " % int(i+1),end='')
    
                for j in range(natms): # cols
                    print("%3i " % connectivity[i][j],end='')
            print("")
    
        return connectivity


    def generate_sobol_conformations(self,
                                     n_conformations: int,
                                     seed: int = 345,
                                     frac_opt = 0.10, 
                                     path: str = "",
                                     name_pattern: str = "conf",
                                     mode: str = "orca",
                                     level_of_theory: str = "WB97M-D3BJ",
                                     basis_set: str = "def2-TZVPPD",
                                     maxcore_mb: int = 4000,
                                     nproc: int = 16) -> None:
        """
        Generate n conformations by sampling constrained dihedral(s) with a Sobol sequence.
        Intended for a gas-phase minimum with flexible torsions declared in self.constraints.dihedrals
        as (dih_atom_index, None). This function writes input files for ORCA or Gaussian
        only if the topological connectivity is preserved.
        """
        import os
        from copy import deepcopy
        import random
    
        # --- Safety checks ---------------------------------------------------
        if len(self.constraints.dihedrals) == 0:
            raise ValueError("No dihedral constraints set. Populate self.constraints.dihedrals first.")
    
        # Reference connectivity from the starting geometry
        connectivity_ref = ZMatrix.ase_get_connectivity(self.get_atoms())
    
        # Prepare ranges for each free dihedral (degrees)
        n_dim = len(self.constraints.dihedrals)
        ranges = [(0.0, 360.0)] * n_dim
    
        # Backup original state (we'll restore each iteration)
        zmat_backup = deepcopy(self.zmat)
        constraints_backup = deepcopy(self.constraints)
    
        # Sobol samples
        samples = ZMatrix.sobol_nd_in_ranges(n_dim, n_conformations, ranges, scramble=False, seed=seed)

        random.seed(seed)

        num_128s = int(n_conformations * frac_opt)
        num_zeros = n_conformations - num_128s  # ensures total length is exactly N

        geom_maxiters = [0]*num_zeros + [128]*num_128s
        random.shuffle(geom_maxiters)
        conf_idx = 0
        for i, sample in enumerate(samples):
            
            # --- Reset to the original geometry and constraints --------------
            self.zmat = deepcopy(zmat_backup)
            self.constraints = deepcopy(constraints_backup)
    
            # Build a fresh Constraints object with *values* for dihedrals
            new_constraints = Constraints()
            # Bonds and angles carry over unchanged
            for b in self.constraints.bonds:
                new_constraints.bonds.append(tuple(b))
            for a in self.constraints.angles:
                new_constraints.angles.append(tuple(a))
            # Dihedrals get assigned the sampled values
            for (dih_idx, _), val in zip(self.constraints.dihedrals, sample):
                new_constraints.dihedrals.append((dih_idx, float(val)))
    
            # Install and apply to zmat (mutates self.zmat)
            self.constraints = new_constraints
            _ = self.apply_constraints_to_zmat(normalise_angles=True, strict_connectivity=True)
            print(f'Generating conformation {i} with {new_constraints}')
    
            # Check topology preserved
            connectivity_mod = ZMatrix.ase_get_connectivity(self.get_atoms())
            if not np.array_equal(connectivity_mod, connectivity_ref):
                # Skip conformations that break bonds/close new ones
                print(f'Skipping conformation {i} due to changes in connectivity')
                continue
    
            # Prepare output dir and filenames
            subdir = os.path.join(path, f"{name_pattern}_{conf_idx}")
            os.makedirs(subdir, exist_ok=True)
    
            if mode.lower() == "orca":
                # If you extend your save_orca_input signature to accept LoT/basis, call it here.
                # Example: self.save_orca_input(filename, level_of_theory, basis_set, maxcore_mb, nproc)
                filename = os.path.join(subdir, f"{name_pattern}_{conf_idx}.inp")
                try:
                    # Option A: if you KEEP your simple wrapper:
                    #   orca_input = PrintUtils.print_orca_input(self.zmat, self.zmat_conn, self.constraints,
                    #											level_of_theory, basis_set, maxcore_mb, nproc)
                    #   with open(filename, "w", encoding="utf-8") as f: f.write(orca_input)
                    # Option B: if you EXTEND save_orca_input to accept the parameters:
                    self.save_orca_input(filename, level_of_theory = level_of_theory, basis_set = basis_set, maxcore_mb = maxcore_mb, nproc = nproc, geom_maxiter = geom_maxiters[i])
                    print(f'Conformation {i} saved to {filename}')
                    conf_idx += 1
                except TypeError:
                    # Back-compat path: older save_orca_input(filename) only
                    orca_input = PrintUtils.print_orca_input(self.zmat, self.zmat_conn, self.constraints,
                                                             level_of_theory = level_of_theory, basis_set = basis_set, maxcore_mb = maxcore_mb, nproc = nproc, geom_maxiter = geom_maxiters[i])
                    conf_idx += 1
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(orca_input)
    
            elif mode.lower() == "gaussian":
                filename = os.path.join(subdir, f"{name_pattern}_{i}.com")
                preamble = f"""%mem={maxcore_mb}MB
    %nprocshared={nproc}
    #P {level_of_theory} {basis_set} Opt=CalcFC
    
    {self.name or 'Sobol conformation'}
    
    0 1
    """
                self.save_gaussian_com(filename, preamble)
                conf_idx += 1
            else:
                raise ValueError("mode must be 'orca' or 'gaussian'")
    
        # Restore original state at the end (not strictly necessary due to per-iter reset)
        self.zmat = zmat_backup
        self.constraints = constraints_backup
        self.apply_constraints_to_zmat
        
