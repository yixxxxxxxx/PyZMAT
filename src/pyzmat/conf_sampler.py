from ase import Atoms
from ase.io import read, write
from ase.geometry import *
from ase.constraints import *
from ase.units import *

import numpy as np
from scipy.stats import qmc

from .zmat_utils import ZmatUtils
from .print_utils import PrintUtils
from .constraints import Constraints
from .parse_utils import ParseUtils
from .zmatrix import ZMatrix

import os

from copy import deepcopy

class TorsionSampler:
    def __init__(self, mol_ref, sample_dir):
        if not isinstance(mol_ref, ZMatrix):
            raise TypeError("template_mol must be a ZMatrix instance")
        self.mol_ref = deepcopy(mol_ref)
        self.campaign_dir = sample_dir
        os.makedirs(self.campaign_dir, exist_ok=True)

        self.conn_ref = ZMatrix.ase_get_connectivity(
            self.mol_ref.get_atoms()
        )

        self.state = {
            "candidate_pool_file": None,
            "valid_mask_file": None,
            "selected_indices": [],
            "written_indices": [],
            "completed_indices": [],
            "failed_indices": [],
            "next_batch_number": 1,
        }

    def generate_sobol_pool(
        self,
        n_samples: int,
        ranges: list[tuple[float, float]],
        seed: int | None = None,
        scramble: bool = False,
    ) -> np.ndarray:
        n_dim = len(ranges)
        # Initialise Sobol sampler
        sampler = qmc.Sobol(d = n_dim, scramble=scramble, seed=seed)
    
        # Generate points in [0,1]^n
        sobol_unit = sampler.random(n_samples)
    
        # Scale to user ranges
        lower = np.array([lo for lo, hi in ranges])
        upper = np.array([hi for lo, hi in ranges])
        sobol_scaled = qmc.scale(sobol_unit, lower, upper)
        pool_path = os.path.join(self.campaign_dir, "candidate_pool.npy")
        np.save(pool_path, sobol_scaled)
    
        return sobol_scaled
    
    def generate_geometry(
        self, 
        torsions: np.ndarray,
    ) -> ZMatrix:
        """
        Create a fresh constrained geometry from a torsion vector.

        Parameters
        ----------
        torsions : array-like
            One sampled value per constrained dihedral, in degrees.

        Returns
        -------
        mol : ZMatrix
            Deep-copied molecule with the sampled torsions applied.
        """
        mol = deepcopy(self.mol_ref)

        torsions = np.asarray(torsions, dtype = float)

        if len(torsions) != len(mol.constraints.dihedrals):
            raise ValueError(
                "Length of torsions does not match number of constrained dihedrals"
            )

        old_constraints = mol.constraints
        new_constraints = Constraints()

        for bond in old_constraints.bonds:
            new_constraints.bonds.append(tuple(bond))

        for angle in old_constraints.angles:
            new_constraints.angles.append(tuple(angle))

        for (dih_idx, _), val in zip(old_constraints.dihedrals, torsions):
            new_constraints.dihedrals.append((dih_idx, float(val)))

        mol.constraints = new_constraints
        mol.apply_constraints_to_zmat(normalise_angles=True, strict_connectivity=True)

        return mol
        

    def check_connectivity(
        self,
        mol: ZMatrix, 
    ) -> bool:
        conn = ZMatrix.ase_get_connectivity(mol.get_atoms())

        if np.array_equal(conn, self.conn_ref):
            return True
        else:
            return False
        
    def check_sterics(
        self,
        mol: ZMatrix, 
    ) -> :
        ... #will code this later

    def filter_valid_candidates(self, torsion_points):
        for torsions in torsion_points:
            mol = self.generate_geometry(torsions)
            if self.check_connectivity(mol, self.mol_ref) and self.check_sterics(mol, self.mol_ref):




    def choose_initial_batch()

    def choose_next_batch(self, ...):
        ...

    def write_batch(self, ...):
        ...

    def save_state(self):
        ...
    

