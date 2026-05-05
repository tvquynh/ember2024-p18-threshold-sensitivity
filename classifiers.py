#!/usr/bin/env python3
"""classifiers.py — Uniform API for the four classifier families.

Each trainer exposes:
    train_<name>(X, y, seed, sample_weight=None, X_val=None, y_val=None)
    predict_proba_<name>(model, X)   -> np.ndarray (n,) positive-class score

All models run on CPU. `n_jobs=-1` / `n_workers=-1` fans out across the
73 cores per the server spec.

XGBoost is configured with subsample=0.8 + colsample_bytree=0.8 so seeds
actually produce different models (v1 of the manuscript bumped into
deterministic outputs under the hist method).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from config import LGBM_PARAMS, XGB_PARAMS, XGB_PARAMS_DETERMINISTIC, RF_PARAMS, MLP_PARAMS
from ember_v3_schema import CATEGORICAL_FEATURES


# ── LightGBM ─────────────────────────────────────────────────────────────────
def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
    params_override: Optional[dict] = None,
):
    import lightgbm as lgb
    p = dict(LGBM_PARAMS if params_override is None else params_override)
    p["seed"] = seed
    p["feature_pre_filter"] = False
    n_estimators = p.pop("n_estimators", 500)

    train_set = lgb.Dataset(
        X_train, label=y_train, weight=sample_weight,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=-1)]
    valid_sets = None
    if early_stopping_rounds and X_val is not None and y_val is not None:
        val_set = lgb.Dataset(
            X_val, label=y_val, reference=train_set,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )
        valid_sets = [val_set]
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    return lgb.train(
        p, train_set, num_boost_round=n_estimators,
        valid_sets=valid_sets, callbacks=callbacks,
    )


def predict_proba_lgbm(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=np.float64)


# ── XGBoost ──────────────────────────────────────────────────────────────────
def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    deterministic: bool = False,
):
    import xgboost as xgb
    base = XGB_PARAMS_DETERMINISTIC if deterministic else XGB_PARAMS
    p = dict(base)
    p["random_state"] = seed
    p["seed"] = seed
    n_estimators = p.pop("n_estimators", 500)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    return xgb.train(p, dtrain, num_boost_round=n_estimators)


def predict_proba_xgb(model, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb
    return np.asarray(model.predict(xgb.DMatrix(X)), dtype=np.float64)


# ── Random Forest ────────────────────────────────────────────────────────────
def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


def predict_proba_rf(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


# ── MLP ──────────────────────────────────────────────────────────────────────
def _seed_torch(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _make_mlp(input_dim: int, seed: int):
    import torch
    import torch.nn as nn
    _seed_torch(seed)
    hid = MLP_PARAMS["hidden_sizes"]
    dropout = MLP_PARAMS["dropout"]
    layers = []
    prev = input_dim
    for h in hid:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class _MLPBundle:
    """MLP network + the StandardScaler fit on its training data.

    Feature scaling is essential for MLP on EMBER features — without it,
    high-magnitude features (raw size, string counts) dominate gradients
    and the network fails to separate classes (all-zero prediction).
    """
    __slots__ = ("net", "scaler")

    def __init__(self, net, scaler):
        self.net = net
        self.scaler = scaler


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
):
    """Stratified 10% internal val split is made by caller. This trainer
    consumes X_val/y_val for early stopping."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
    device = torch.device("cpu")

    # Scale once on the passed-in training data (val + test use the same scaler).
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train.astype(np.float32))
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0
                              ).astype(np.float32)

    mlp = _make_mlp(X_train_s.shape[1], seed).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=MLP_PARAMS["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    w = sample_weight if sample_weight is not None else np.ones(len(y_train), dtype=np.float32)
    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(w.astype(np.float32)),
    )
    loader = DataLoader(
        train_ds, batch_size=MLP_PARAMS["batch_size"], shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    val_ok = X_val is not None and y_val is not None
    if val_ok:
        X_val_s = np.nan_to_num(
            scaler.transform(X_val.astype(np.float32)),
            nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32)
        Xv = torch.from_numpy(X_val_s)
        yv = torch.from_numpy(y_val.astype(np.float32))

    best_loss = float("inf")
    patience = MLP_PARAMS["early_stopping_patience"]
    no_improve = 0
    best_state = None

    for epoch in range(MLP_PARAMS["epochs"]):
        mlp.train()
        for xb, yb, wb in loader:
            opt.zero_grad()
            logits = mlp(xb).squeeze(1)
            per = loss_fn(logits, yb)
            loss = (per * wb).mean()
            loss.backward()
            opt.step()
        if val_ok:
            mlp.eval()
            with torch.no_grad():
                vloss = nn.functional.binary_cross_entropy_with_logits(
                    mlp(Xv).squeeze(1), yv
                ).item()
            if vloss < best_loss - 1e-6:
                best_loss = vloss
                no_improve = 0
                best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()
    return _MLPBundle(mlp, scaler)


def predict_proba_mlp(model, X: np.ndarray) -> np.ndarray:
    import torch
    net = model.net if isinstance(model, _MLPBundle) else model
    scaler = model.scaler if isinstance(model, _MLPBundle) else None
    X_in = X.astype(np.float32)
    if scaler is not None:
        X_in = np.nan_to_num(scaler.transform(X_in),
                             nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(X_in)).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


# ── Dispatcher ───────────────────────────────────────────────────────────────
TRAINERS = {
    "LightGBM":     train_lgbm,
    "XGBoost":      train_xgb,
    "RandomForest": train_rf,
    "MLP":          train_mlp,
}

PREDICTORS = {
    "LightGBM":     predict_proba_lgbm,
    "XGBoost":      predict_proba_xgb,
    "RandomForest": predict_proba_rf,
    "MLP":          predict_proba_mlp,
}


def train(name: str, X: np.ndarray, y: np.ndarray, seed: int,
          sample_weight: Optional[np.ndarray] = None,
          X_val: Optional[np.ndarray] = None,
          y_val: Optional[np.ndarray] = None,
          **kwargs):
    trainer = TRAINERS[name]
    if name == "MLP":
        return trainer(X, y, seed, sample_weight=sample_weight,
                       X_val=X_val, y_val=y_val)
    if name == "LightGBM":
        return trainer(X, y, seed, sample_weight=sample_weight,
                       X_val=X_val, y_val=y_val, **kwargs)
    return trainer(X, y, seed, sample_weight=sample_weight, **kwargs)


def predict_proba(name: str, model, X: np.ndarray) -> np.ndarray:
    return PREDICTORS[name](model, X)


def stratified_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float, seed: int):
    """10% stratified val split for MLP early stopping."""
    rs = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0 = max(1, int(round(val_fraction * len(idx0))))
    n1 = max(1, int(round(val_fraction * len(idx1))))
    val_idx = np.concatenate([
        rs.choice(idx0, n0, replace=False),
        rs.choice(idx1, n1, replace=False),
    ])
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return X[mask], y[mask], X[val_idx], y[val_idx]
