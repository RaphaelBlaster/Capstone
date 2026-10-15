import numpy as np
from .data import TARGETS, WEIGHTS


def scores(truth, prediction):
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 5:
        raise ValueError("Expected matching N x 5 arrays in canonical target order.")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError("Finite non-negative targets and predictions required.")
    per_target = {}
    for j, name in enumerate(TARGETS):
        denom = np.square(y[:, j] - y[:, j].mean()).sum()
        per_target[name] = float(1 - np.square(y[:, j] - p[:, j]).sum()/denom) if denom else None
    mean = np.sum(y * WEIGHTS) / (len(y) * WEIGHTS.sum())
    denominator = np.sum(WEIGHTS * np.square(y-mean))
    pooled = float(1 - np.sum(WEIGHTS * np.square(y-p))/denominator) if denominator else None
    logs_y, logs_p = np.log1p(y), np.log1p(p)
    log_r2 = []
    for j in range(5):
        denom = np.square(logs_y[:, j] - logs_y[:, j].mean()).sum()
        log_r2.append(float(1 - np.square(logs_y[:, j]-logs_p[:, j]).sum()/denom) if denom else None)
    return {"global_weighted_r2_raw": pooled,
            "weighted_mean_target_r2_raw": float(np.dot(WEIGHTS, list(per_target.values()))) if all(v is not None for v in per_target.values()) else None,
            "weighted_mean_target_r2_log1p": float(np.dot(WEIGHTS, log_r2)) if all(v is not None for v in log_r2) else None,
            "per_target_r2_raw": per_target,
            "per_target_rmse_g": dict(zip(TARGETS, np.sqrt(np.mean(np.square(y-p), axis=0)).tolist()))}


def group_bootstrap(truth, prediction, groups, draws=1000, seed=42):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"interval": None, "reason": "Fewer than two acquisition groups."}
    rng = np.random.default_rng(seed)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    values = []
    for _ in range(draws):
        rows = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        score = scores(truth[rows], prediction[rows])["global_weighted_r2_raw"]
        if score is not None:
            values.append(score)
    return {"interval": np.quantile(values, [.025, .975]).tolist() if values else None,
            "groups": len(unique), "draws": draws,
            "scope": "Group resampling of fixed OOF predictions; excludes training-seed variability."}
