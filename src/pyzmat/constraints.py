"""
Module defining the Constraints class for internal coordinate optimization.
"""


class Constraints:
    """
    A class to store constraints for bonds, angles, and dihedrals.
    Each entry can be an integer (the index) or a tuple (index, value).
    If value is not provided (or is None), the current Z-matrix value will be used.
    """
    def __init__(self, bonds=None, angles=None, dihedrals=None):
        self.bonds = self._validate_dof_list(bonds if bonds is not None else [], "bonds")
        self.angles = self._validate_dof_list(angles if angles is not None else [], "angles")
        self.dihedrals = self._validate_dof_list(dihedrals if dihedrals is not None else [], "dihedrals")

    @staticmethod
    def _validate_dof_list(value, name):
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list of integers or (integer, value) tuples")
        validated = []
        for item in value:
            if isinstance(item, int):
                validated.append((item, None))
            elif isinstance(item, tuple):
                if len(item) != 2:
                    raise ValueError(f"Each tuple in {name} must have exactly 2 elements: (index, value)")
                index, val = item
                if not isinstance(index, int):
                    raise TypeError(f"In {name}, the index must be an integer")
                if val is not None and not isinstance(val, (int, float)):
                    raise TypeError(f"In {name}, the value must be a number or None")
                validated.append((index, val))
            else:
                raise TypeError(f"Each element in {name} must be an integer or a (integer, value) tuple")
        if name == "angles":
            for index, _ in validated:
                if index < 2:
                    raise ValueError("All indices in the angle list must be at least 2")
        if name == "dihedrals":
            for index, _ in validated:
                if index < 3:
                    raise ValueError("All indices in the dihedral list must be at least 3")
        return validated

    def __repr__(self):
        return f"Constraints(bonds={self.bonds}, angles={self.angles}, dihedrals={self.dihedrals})"
