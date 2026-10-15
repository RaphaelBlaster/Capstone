"""Weighted nonnegative projection onto the documented biomass sum identities."""
import numpy as np

from .data import WEIGHTS


def reconcile(predictions):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError('Reconciliation needs finite five-target predictions.')
    # Primitive order Green, Dead, Clover; GDM=Green+Clover; Total=GDM+Dead.
    design = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    best = np.zeros_like(values)
    distance = (values**2) @ WEIGHTS
    # Enumerate the eight active sets of this three-variable convex NNLS problem.
    for mask in range(1, 8):
        active = [i for i in range(3) if mask & (1 << i)]
        basis = design[:, active]
        coefficients = np.linalg.solve(basis.T @ (WEIGHTS[:, None]*basis),
                                       basis.T @ (WEIGHTS[:, None]*values.T)).T
        feasible = (coefficients >= 0).all(axis=1)
        candidate = coefficients @ basis.T
        error = ((values-candidate)**2) @ WEIGHTS
        better = feasible & (error < distance)
        best[better], distance[better] = candidate[better], error[better]
    best[:, 3] = best[:, 0] + best[:, 2]
    best[:, 4] = best[:, 3] + best[:, 1]
    return best
