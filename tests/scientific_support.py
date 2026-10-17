"""Independent polynomial oracle in eV and angstrom."""
import copy
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from pyzmat import ZMatrix


class Polynomial(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions
        self.results = dict(energy=float((x*x).sum()+x.sum()**2+(x**4).sum()),
                            forces=-(2*x+2*x.sum()+4*x**3))

    def get_hessian(self, atoms=None):
        x = atoms.positions.ravel()
        return np.diag(2+12*x*x)+2*np.ones((len(x),len(x)))


def molecule(data):
    m = ZMatrix(*copy.deepcopy(data))
    m.model = 'mace-off'
    m.model_size = 'polynomial'
    m.calculator = Polynomial()
    return m


def energy_fd(m, step=1e-4):
    """Differentiate scalar E directly; no production force, B or K routines."""
    indices = [(i,j) for i,row in enumerate(m.zmat) for j in (1,2,3) if row[j] is not None]
    base = copy.deepcopy(m.zmat)
    def energy(displacement):
        m.zmat = copy.deepcopy(base)
        for dq,(i,j) in zip(displacement, indices):
            m.zmat[i][j] += dq if j==1 else np.rad2deg(dq)
        return m.get_energy()
    size=len(indices); eye=np.eye(size)*step
    try:
        zero=energy(np.zeros(size))
        force=np.empty(size); h=np.empty((size,size))
        for i in range(size):
            plus,minus=energy(eye[i]),energy(-eye[i])
            force[i]=-(plus-minus)/(2*step)
            h[i,i]=(plus-2*zero+minus)/step**2
            for j in range(i):
                h[i,j]=h[j,i]=(energy(eye[i]+eye[j])-energy(eye[i]-eye[j])-energy(-eye[i]+eye[j])+energy(-eye[i]-eye[j]))/(4*step**2)
        return force,h
    finally:
        m.zmat=base
