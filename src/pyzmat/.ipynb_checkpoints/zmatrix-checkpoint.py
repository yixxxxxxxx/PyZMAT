from ase import Atoms
from ase.io import read, write
from ase.geometry import *
from ase.constraints import *
from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.units import *

import time

from scipy.optimize import minimize
import numpy as np

from .zmat_utils import ZmatUtils
from .print_utils import PrintUtils
from .constraints import Constraints
from .parse_utils import ParseUtils

import json

from typing import Tuple, Optional

import io
from contextlib import redirect_stdout

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

class ZMatrix:

    def __init__(self, zmat, zmat_conn, constraints = None, name = None, energy = None, forces = None, hessian = None):
        self.name = name if name else "unnamed molecule"
        self.zmat = zmat
        self.zmat_conn = zmat_conn  # Each connectivity is now (symbol, bond, angle, dihedral)
        self._constraints = constraints if constraints else Constraints()  # Default to empty constraints
    

        # Pretty unused, ignore these
#        self.con_dict = self._find_constraint_values()
#        self.con_ids = list(self.con_dict.keys())
#        self._apply_constraints()
#        self.var_ids = self._find_var_ids()  # Find variable indices
#        self.var_list = self._extract_variables()  # Extract initial variables


        
        self.b_matrix = None
        self.energy = energy if energy is not None else None
        self.forces = forces if forces is not None else None
        self.hessian = hessian if hessian is not None else None
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
    def load_json(cls, filename: str) -> "ZMatrix":
        """
        Load a ZMatrix object (and its last forces/energy/hessian, if present)
        from a JSON file produced by dump_json().
        The .json files dumped by pyzmat.ZMatrix has name_ZMatrix.json by default. 
        """
        with open(filename, "r") as f:
            state = json.load(f)

        zmat      = [list(row) for row in state["zmat"]]
        zmat_conn = [tuple(item) for item in state["zmat_conn"]]

        cons = state["constraints"]
        constraints = Constraints(
            bonds     = [tuple(item) for item in cons["bonds"]],
            angles    = [tuple(item) for item in cons["angles"]],
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
        self.ase_constraints = self._get_ase_constraints()

    def clear_constraints(self):
        self._constraints = Constraints()
        self.ase_constraints = self._get_ase_constraints()

    def attach_calculator(self, model, model_size = 'large', gpu = False):
        """Attach an MLIP as an ASE calculator to the ZMatrix object. Only supports MACE-off23 ('mace') and AIMNet2 ('aimnet2') at the moment."""
        if model not in ['mace', 'aimnet2']:
            raise ValueError("Only MACE-off23 ('mace') and AIMNet2 ('aimnet2') are currently supported")
        if model == 'mace':
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

        self.model = model

    
    


#    ## Some helper functions for the unused custom optimisation routine ############################################################################################
#
#    def _find_constraint_values(self):
#        """Tied to the unused ZMatrix.optimise() routine"""
#        con_dict = {}
#        # Bonds: global index = index * 3
#        for index, val in self.constraints.bonds:
#            global_index = index * 3
#            if val is None:
#                val = self._get_value_by_index(global_index)
#            con_dict[global_index] = val
#        # Angles: global index = index * 3 + 1
#        for index, val in self.constraints.angles:
#            global_index = index * 3 + 1
#            if val is None:
#                val = self._get_value_by_index(global_index)
#            con_dict[global_index] = val
#        # Dihedrals: global index = index * 3 + 2
#        for index, val in self.constraints.dihedrals:
#            global_index = index * 3 + 2
#            if val is None:
#                val = self._get_value_by_index(global_index)
#            con_dict[global_index] = val
#        return con_dict
#
#    def _apply_constraints(self):
#        """Tied to the unused ZMatrix.optimise() routine"""
#        for global_index, value in self.con_dict.items():
#            atom_index = global_index // 3
#            coord_index = (global_index % 3) + 1
#            self.zmat[atom_index][coord_index] = value
#
#    def _find_var_ids(self):
#        """Tied to the unused ZMatrix.optimise() routine"""
#        total_vars = 3 * len(self.zmat)
#        var_ids = [i for i in range(total_vars) if i not in self.con_ids and self._get_value_by_index(i) is not None]
#        return var_ids
#
#    def _get_value_by_index(self, index):
#        """Tied to the unused ZMatrix.optimise() routine"""
#        atom_idx = index // 3
#        coord_idx = index % 3
#        return self.zmat[atom_idx][coord_idx + 1]
#
#    def _extract_variables(self):
#        """Tied to the unused ZMatrix.optimise() routine"""
#        all_values = [coord for row in self.zmat for coord in row[1:]]
#        all_values = np.array(all_values)
#        return all_values[self.var_ids]
#
#    def _reconstruct_full_z_matrix(self, vars):
#        """Tied to the unused ZMatrix.optimise() routine"""
#        full_values = np.array([coord for row in self.zmat for coord in row[1:]])
#        var_id = 0
#        for i in range(len(full_values)):
#            if i in self.var_ids:
#                full_values[i] = vars[var_id]
#                var_id += 1
#            elif i in self.con_ids:
#                full_values[i] = self.con_dict[i]
#        reconstructed_zmat = []
#        for i, (atom, bond, angle, dihedral) in enumerate(self.zmat):
#            bond_val, angle_val, dihedral_val = full_values[i * 3:(i + 1) * 3]
#            new_values = [
#                bond_val if bond is not None else None,
#                angle_val if angle is not None else None,
#                dihedral_val if dihedral is not None else None
#            ]
#            reconstructed_zmat.append((atom, *new_values))
#        return reconstructed_zmat

    ## Calculate tensors of coordinate derivatives from pyzmat.ZmatUtils #######################################################

    def _get_B_matrix(self):
        B = ZmatUtils.get_B_matrix(self.zmat, self.zmat_conn)
        return B

    def _get_K_tensor(self):
        K = ZmatUtils.get_curvature_tensor(self.zmat, self.zmat_conn)
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

    #def get_energy(self, vars):
    #    if not hasattr(self, "model"):
    #       raise AttributeError(f"{self.__class__.__name__!r} has no attribute 'model'. Did you forget to run ZMatrix.attach_calculator()?")
    #    zmat = self._reconstruct_full_z_matrix(vars)
    #   atoms = ZmatUtils.zmat_2_atoms(zmat, self.zmat_conn)
    #    atoms.calc = self.calculator
    #    energy = atoms.get_potential_energy()
    #    self.iteration += 1
    #    print('Iteration', self.iteration)
    #    PrintUtils.print_xyz(atoms, comment='', fmt='%22.15f')
    #    print(energy)
    #    return energy

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
        if self.model == 'mace':
            H_cart = self.calculator.get_hessian(atoms = self.get_atoms())
            
        elif self.model == 'aimnet2':
            from aimnet.calculators import AIMNet2Calculator
            
            data = ZMatrix.get_aimnet_data(self.get_atoms())
            calc = AIMNet2Calculator(model="aimnet2")
            out = calc.eval(data, forces = True, hessian = True)
            H_cart = out['hessian']

        N = H_cart.shape[1]
        return H_cart.reshape((3 * N, 3 * N))


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
        z_plus       = copy.deepcopy(self.zmat)
        z_plus[i][j] += delta
        F_plus       = self._get_temp_forces(z_plus, self.zmat_conn, self.calculator)
    
        # −δ -------------------------------------------------------------
        z_minus       = copy.deepcopy(self.zmat)
        z_minus[i][j] -= delta
        F_minus       = self._get_temp_forces(z_minus, self.zmat_conn, self.calculator)
    
        #   F = −∂E/∂q    ⇒   dF/dq   =  −∂²E/∂q²
        # so we have to invert the sign once:
        col = -(F_plus - F_minus) / (2.0 * delta)          #  <-- sign fixed   (NEW)
    
        # if *this* coordinate qⱼ is an angle or torsion, rescale the whole column
        if j in (2, 3):                                    # 1 = bond, 2 = angle, 3 = torsion
            col *= 180 / np.pi                            #  <-- unit fix     (NEW)
    
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
            if j in (2, 3):                # angle or torsion
                H[k, :] = H[:, k]
    
        return H
        

    def get_geom_fd_hessian(self, db, da, dt):
        """Quasi-analytical Hessian from a finite-difference curvature tensor, for testing only"""
        H_cart = self.get_hess_cart()
        B = self._get_B_matrix()
        K = ZmatUtils.get_fd_curvature_tensor(self.zmat, self.zmat_conn, db, da, dt)

        F_cart = self.get_forces_cart()

        H_min = B @ H_cart @ np.transpose(B)
        H_res = -1 * np.einsum('isp,i->sp', K, F_cart)
        return H_min + H_res
    
    def get_hessian(self):
        """Calculate fully analytical Hessian. Recommended for production use."""
        H_cart = self.get_hess_cart()
        B = self._get_B_matrix()
        K = self._get_K_tensor()
        F_cart = self.get_forces_cart()

        H_min = B @ H_cart @ np.transpose(B)
        H_res = -1 * np.einsum('isp,i->sp', K, F_cart)
        hessian = H_min + H_res

        self.hessian = hessian
        return hessian


    ## Unused custom optimisation routine ###########################################################################################

    def callback(self, vars):
        """Tied to the unused ZMatrix.optimise() routine"""
        self.iteration += 1
        zmat = self._reconstruct_full_z_matrix(vars)
        atoms = self.get_atoms()
        print("Iteration:", self.iteration, "Current Z-matrix:", zmat, flush=True)
        PrintUtils.print_xyz(atoms, comment=self.name + str(self.iteration), fmt='%22.15f')

    def optimise(self):
        """Imcomplete/deprecated optimisation routine. Will likely produce incorrect results. Use ZMatrix.optimise_ase() instead"""
        print('WARNING: imcomplete/deprecated function. Will likely produce incorrect results. Use ZMatrix.optimise_ase() instead')
        result = minimize(
            self.get_energy,
            self.var_list,
            method='BFGS',
            jac=self.get_jacobian,
            callback=self.callback,
            options={'gtol': 1e-8}
        )
        self.zmat = self._reconstruct_full_z_matrix(result.x)
        self.b_matrix = self._get_B_matrix()  # Update B-matrix
        self.con_dict = self._find_constraint_values()  # Update constraint values
        self.con_ids = list(self.con_dict.keys())
        self.var_ids = self._find_var_ids()  # Update variable indices
        self.var_list = self._extract_variables()  # Extract new variables
        return self.zmat, result.fun


    ## Visualisation and optimisation ########################################################################

    def view_ase(self):
        from ase.visualize import view
        return view(self.get_atoms(), viewer = 'x3d')
    
    def optimise_ase(self, trajectory = None, mode = 'linesearch', fmax = 1e-5, calc_hess = False):
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
        
        atoms.set_constraint([self.ase_constraints])

        if mode == 'linesearch':
            dyn = BFGSLineSearch(atoms, trajectory = trajectory, restart = f'{self.name}_linesearch_opt.json')
            print('Now beginning ASE BFGS Line Search minimisation routine')
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
            print('Now beginning ASE BFGS minimisation routine')
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
        if calc_hess == True:
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

    ## Functions for saving output ################################################################################

    def dump_json(self, filename = None):
        """
        Save all core data to a JSON file.
        """
        if filename is None:
            filename = f'{self.name}_ZMatrix.json'
        state = {
            "zmat":      self.zmat,
            "zmat_conn": self.zmat_conn,
            "constraints": {
                "bonds":     [[idx, val] for idx, val in self.constraints.bonds],
                "angles":    [[idx, val] for idx, val in self.constraints.angles],
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