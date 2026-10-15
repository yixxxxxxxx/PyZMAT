"""Batched analytical coordinate derivatives; imported only by the Torch API."""

from numbers import Integral

import torch


class _Jet:
    """Value, gradient and Hessian, with Cartesian components on the last axis.

    Derivative axes precede the component axis; all earlier axes are batches.
    These explicit chain rules also work inside torch.inference_mode().
    """

    def __init__(self, value, grad, hess):
        self.v, self.d, self.h = value, grad, hess

    def __add__(self, other):
        return _Jet(self.v + other.v, self.d + other.d, self.h + other.h)

    def __neg__(self):
        return _Jet(-self.v, -self.d, -self.h)

    def __sub__(self, other):
        return self + (-other)

    def _product(self, other, op):
        a, b = self, other
        return _Jet(
            op(a.v, b.v),
            op(a.d, b.v.unsqueeze(-2)) + op(a.v.unsqueeze(-2), b.d),
            op(a.h, b.v[..., None, None, :])
            + op(a.d.unsqueeze(-2), b.d.unsqueeze(-3))
            + op(a.d.unsqueeze(-3), b.d.unsqueeze(-2))
            + op(a.v[..., None, None, :], b.h),
        )

    def __mul__(self, other):
        return self._product(other, torch.mul)

    def cross(self, other):
        return self._product(other, lambda a, b: torch.linalg.cross(a, b, dim=-1))

    def unary(self, value, first, second):
        return _Jet(
            value,
            first.unsqueeze(-2) * self.d,
            first[..., None, None, :] * self.h
            + second[..., None, None, :]
            * self.d.unsqueeze(-2) * self.d.unsqueeze(-3),
        )

    def sin(self):
        return self.unary(self.v.sin(), self.v.cos(), -self.v.sin())

    def cos(self):
        return self.unary(self.v.cos(), -self.v.sin(), -self.v.cos())

    def unit(self, valid):
        square = self * self
        norm2 = _Jet(*(x.sum(dim=-1, keepdim=True)
                       for x in (square.v, square.d, square.h)))
        valid.append((norm2.v > 1e-24) & torch.isfinite(norm2.v))
        inverse = norm2.unary(
            norm2.v.rsqrt(), -0.5 * norm2.v.pow(-1.5),
            0.75 * norm2.v.pow(-2.5),
        )
        return self * inverse


def get_coordinate_derivatives(q, zmat_conn, *, validate=True):
    """Implementation of ZmatUtils.get_coordinate_derivatives_torch."""
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor.")
    if q.dtype not in (torch.float32, torch.float64):
        raise TypeError("q must have dtype torch.float32 or torch.float64.")
    n = len(zmat_conn)
    m = 3 * n - 6
    if n < 3:
        raise ValueError("At least three atoms are required.")
    if q.ndim < 1 or q.shape[-1] != m:
        raise ValueError(f"q must have shape (..., {m}) for {n} atoms.")
    refs = []
    for i, row in enumerate(zmat_conn):
        if len(row) != 4:
            raise ValueError("Connectivity rows must be (symbol, j, k, l).")
        active = row[1:1 + min(i, 3)]
        if any(not isinstance(j, Integral) or isinstance(j, bool)
               or not 0 <= j < i for j in active) or len(set(active)) != len(active):
            raise ValueError(f"Atom {i} must reference distinct earlier atoms.")
        refs.append(tuple(active))

    batch = q.shape[:-1]
    eye = torch.eye(m, dtype=q.dtype, device=q.device)
    zero_d = q.new_zeros((*batch, m, 1))
    zero_h = q.new_zeros((*batch, m, m, 1))

    def constant(value):
        return _Jet(q.new_tensor(value).expand(*batch, len(value)), zero_d, zero_h)

    def coordinate(index):
        return _Jet(q[..., index:index + 1],
                    eye[:, index:index + 1].expand(*batch, m, 1), zero_h)

    origin = constant([0., 0., 0.])
    x_axis, y_axis = constant([1., 0., 0.]), constant([0., 1., 0.])
    positions = [origin, coordinate(0) * x_axis]
    r, a = coordinate(1), coordinate(2)
    third = r * a.cos() * x_axis + r * a.sin() * y_axis
    positions.append(third if refs[2][0] == 0 else positions[1] - third)
    valid = []
    for i in range(3, n):
        j, k, l = refs[i]
        t1 = (positions[j] - positions[k]).unit(valid)
        t2 = (positions[l] - positions[k]).unit(valid)
        normal = t1.cross(t2).unit(valid)
        in_plane = normal.cross(t1)
        r, a, d = (coordinate(3 * i - 6 + offset) for offset in range(3))
        positions.append(positions[j] + r * a.sin() * d.cos() * in_plane
                         + r * a.sin() * d.sin() * normal - r * a.cos() * t1)

    if validate:
        lengths = q[..., [0, 1] + list(range(3, m, 3))]
        ok = torch.isfinite(q).all() & (lengths > 0).all()
        if valid:
            ok = ok & torch.stack(valid).all()
        if not bool(ok):
            raise ValueError("Coordinates must be finite with positive bond lengths "
                             "and non-degenerate reference frames (norm > 1e-12).")

    # Constant jets have singleton component dimensions: expand before assembling.
    xyz = torch.stack([p.v for p in positions], dim=-2)
    b = torch.cat([p.d.expand(*batch, m, 3) for p in positions], dim=-1)
    k = torch.cat([p.h.expand(*batch, m, m, 3) for p in positions], dim=-1)
    return xyz, b, k.movedim(-1, -3)
