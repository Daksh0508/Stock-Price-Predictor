"""
models.py — Model Definitions & Training
Contains:
  - build_lstm()       : Primary deep learning model
  - build_gru()        : Optional GRU variant
  - train_lstm()       : Training loop with callbacks
  - BaselineModels     : Linear Regression, Decision Tree, Random Forest
  - save/load helpers
"""

import os
import numpy as np
import logging
from typing import Tuple

# Keras / TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Scikit-learn baselines
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

from config import (
    LSTM_UNITS, DROPOUT_RATE, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, EARLY_STOP_PATIENCE, MODEL_DIR, SEQUENCE_LENGTH
)

logger = logging.getLogger(__name__)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Why LSTM for stock prices?
# ──────────────────────────────────────────────────────────────────────────────
# Stock prices are a time-series: tomorrow's price depends on recent history.
# LSTMs have a "memory" — the cell state (c_t) accumulates information across
# many time steps, while the forget gate decides what to discard.  Traditional
# models (Linear Regression, Random Forest) treat each day independently and
# miss these temporal dependencies.
#
# A GRU is a simplified LSTM: fewer parameters, often similar performance,
# faster to train.  Good to compare both.
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# 1. LSTM
# ──────────────────────────────────────────────────────────────────────────────

def build_lstm(seq_len: int, n_features: int) -> keras.Model:
    """
    Two-layer stacked LSTM with dropout regularisation.

    Architecture
    ------------
    Input  : (batch, seq_len, n_features)
    LSTM₁  : 128 units, return_sequences=True  (feeds into next LSTM)
    Dropout: 0.2
    LSTM₂  : 64  units, return_sequences=False
    Dropout: 0.2
    Dense  : 25  units, ReLU
    Dense  : 1   unit  (predicted Close price, scaled)

    Parameters
    ----------
    seq_len    : sliding window length (e.g. 60 days)
    n_features : number of input features per time step
    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        LSTM(LSTM_UNITS[0], return_sequences=True),
        Dropout(DROPOUT_RATE),

        LSTM(LSTM_UNITS[1], return_sequences=False),
        Dropout(DROPOUT_RATE),

        Dense(25, activation="relu"),
        Dense(1),                      # linear output → raw price prediction
    ], name="LSTM_StockPredictor")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    logger.info("LSTM built: %s", model.summary())
    return model


def build_gru(seq_len: int, n_features: int) -> keras.Model:
    """
    GRU variant — fewer parameters than LSTM, often trains faster.
    Same interface as build_lstm() for easy comparison.
    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        GRU(LSTM_UNITS[0], return_sequences=True),
        Dropout(DROPOUT_RATE),

        GRU(LSTM_UNITS[1], return_sequences=False),
        Dropout(DROPOUT_RATE),

        Dense(25, activation="relu"),
        Dense(1),
    ], name="GRU_StockPredictor")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def train_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    ticker:  str,
    model_type: str = "lstm",
) -> keras.callbacks.History:
    """
    Train *model* with:
    - EarlyStopping  : halt when val_loss stops improving
    - ModelCheckpoint: save the best weights automatically
    - ReduceLROnPlateau: shrink learning rate when stuck

    Returns Keras History object (loss curves accessible as history.history).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, f"{ticker}_{model_type}.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training done. Best val_loss=%.6f", min(history.history["val_loss"]))
    return history


def save_deep_model(model: keras.Model, ticker: str, model_type: str = "lstm") -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{ticker}_{model_type}.keras")
    model.save(path)
    logger.info("Saved %s → %s", model_type.upper(), path)
    return path


def load_deep_model(ticker: str, model_type: str = "lstm") -> keras.Model:
    path = os.path.join(MODEL_DIR, f"{ticker}_{model_type}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Train first.")
    return load_model(path)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Baseline Models
# ──────────────────────────────────────────────────────────────────────────────

class BaselineModels:
    """
    Wraps three scikit-learn regressors with a unified fit/predict interface.
    Baseline models receive FLATTENED sequences (no 3D tensor needed).
    """

    MODELS = {
        "linear_regression": LinearRegression(),
        "decision_tree"    : DecisionTreeRegressor(max_depth=10, random_state=42),
        "random_forest"    : RandomForestRegressor(
                                n_estimators=100, max_depth=15,
                                random_state=42, n_jobs=-1),
    }

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.trained = {}

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """Reshape (samples, seq_len, features) → (samples, seq_len*features)."""
        return X.reshape(X.shape[0], -1)

    def fit_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_flat = self._flatten(X_train)
        for name, clf in self.MODELS.items():
            logger.info("Training %s…", name)
            clf.fit(X_flat, y_train)
            self.trained[name] = clf
        self._save()

    def predict_all(self, X: np.ndarray) -> dict:
        """Return dict mapping model name → prediction array."""
        X_flat = self._flatten(X)
        return {name: clf.predict(X_flat) for name, clf in self.trained.items()}

    def _save(self) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, clf in self.trained.items():
            path = os.path.join(MODEL_DIR, f"{self.ticker}_{name}.pkl")
            joblib.dump(clf, path)

    def load(self) -> None:
        for name in self.MODELS:
            path = os.path.join(MODEL_DIR, f"{self.ticker}_{name}.pkl")
            if os.path.exists(path):
                self.trained[name] = joblib.load(path)
