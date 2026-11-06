"""LSSVM regressor (RBF kernel) and loaders for the trained bored-pile settlement models.

The trained models were saved from the Jupyter notebook as scikit-learn objects
(TransformedTargetRegressor -> Pipeline(MinMaxScaler, LSSVM)) in ``models/*.joblib``.
Because the LSSVM class was defined inside the notebook, it must be importable
under the same name before the ``.joblib`` files are unpickled. If unpickling
fails (for example with a different scikit-learn version), the same model is
rebuilt from ``models/<name>_weights.npz`` (alpha, b, scaled support vectors)
and ``models/model_params.json`` (scalers, gamma, sigma).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_NAMES = ["LSSVM", "LSSVM-PSO", "LSSVM-GWO"]


class LSSVM(BaseEstimator, RegressorMixin):
    """Least Squares Support Vector Machine regressor with RBF kernel."""

    def __init__(self, gamma=1.0, sigma=1.0):
        self.gamma = gamma
        self.sigma = sigma

    def _kernel(self, A, B):
        sq = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-np.maximum(sq, 0) / (2 * self.sigma**2))

    def fit(self, X, y):
        from scipy.linalg import solve
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n = len(y)
        A = np.zeros((n + 1, n + 1))
        A[0, 1:] = 1.0
        A[1:, 0] = 1.0
        A[1:, 1:] = self._kernel(X, X) + np.eye(n) / self.gamma
        sol = solve(A, np.concatenate([[0.0], y]), assume_a="sym")
        self.b_, self.alpha_, self.X_fit_ = sol[0], sol[1:], X
        return self

    def predict(self, X):
        return self._kernel(np.asarray(X, float), self.X_fit_) @ self.alpha_ + self.b_


class SettlementModel:
    """Unscaled interface: predict(X) with X in engineering units -> settlement (mm)."""

    def __init__(self, name, xmin, xscale, ymin, yscale, sigma, gamma, alpha, b, X_fit, q90, source):
        self.name, self.source = name, source
        self.xmin, self.xscale = np.asarray(xmin, float), np.asarray(xscale, float)
        self.ymin, self.yscale = float(ymin), float(yscale)
        self.sigma, self.gamma, self.q90 = float(sigma), float(gamma), float(q90)
        self.core = LSSVM(gamma=gamma, sigma=sigma)
        self.core.alpha_, self.core.b_, self.core.X_fit_ = np.asarray(alpha, float), float(b), np.asarray(X_fit, float)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        ys = self.core.predict(X * self.xscale + self.xmin)
        return (ys - self.ymin) / self.yscale

    def training_inputs(self):
        """Training inputs in engineering units (inverse of the MinMax scaling)."""
        return (self.core.X_fit_ - self.xmin) / self.xscale


def _from_joblib(name, params):
    import joblib
    import __main__
    # the notebook pickled the class as __main__.LSSVM
    __main__.LSSVM = LSSVM
    sys.modules["__main__"].LSSVM = LSSVM
    m = joblib.load(MODEL_DIR / f"{name}.joblib")
    pipe = m.regressor_
    sc, ls, ty = pipe.named_steps["scaler"], pipe.named_steps["lssvm"], m.transformer_
    return SettlementModel(name, sc.min_, sc.scale_, ty.min_[0], ty.scale_[0], ls.sigma, ls.gamma,
                           ls.alpha_, ls.b_, ls.X_fit_, params["q90"], "joblib")


def _from_npz(name, params):
    w = np.load(MODEL_DIR / f"{name}_weights.npz")
    return SettlementModel(name, params["xmin"], params["xscale"], params["ymin"], params["yscale"],
                           params["sigma"], params["gamma"], w["alpha"], w["b"], w["X_support_scaled"],
                           params["q90"], "npz")


def load_models():
    """Load the three trained models; joblib first, npz + json as fallback."""
    P = json.loads((MODEL_DIR / "model_params.json").read_text())
    out = {}
    for name in MODEL_NAMES:
        p = P["models"][name]
        try:
            out[name] = _from_joblib(name, p)
        except Exception:  # different sklearn version, etc.
            out[name] = _from_npz(name, p)
    return P["features"], out
