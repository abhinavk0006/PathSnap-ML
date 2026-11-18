"""Model definitions used in the comparison (step 3) and re-used by the tuner (step 4)."""
from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# name -> (family, key configuration string, factory, max training rows (None = all))
ZOO = {
    "LogisticRegression": ("Linear / probabilistic", "StandardScaler + L2, C=1.0, lbfgs",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300)), None),
    "GaussianNB": ("Linear / probabilistic", "var_smoothing=1e-9, StandardScaler",
                   lambda: make_pipeline(StandardScaler(), GaussianNB()), None),
    "DecisionTree": ("Tree-based", "max_depth=12, min_samples_leaf=50",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED), None),
    "RandomForest": ("Bagging ensemble", "40 trees, max_depth=14, min_samples_leaf=20, max_samples=0.3",
                     lambda: RandomForestClassifier(n_estimators=40, max_depth=14, min_samples_leaf=20, max_samples=0.3,
                                                    n_jobs=2, random_state=SEED), None),
    "HistGradientBoosting": ("Boosting", "max_iter=150, lr=0.1, max_leaf_nodes=31",
                             lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                                                    random_state=SEED), None),
    "LightGBM": ("Boosting", "n_estimators=200, lr=0.1, num_leaves=31, subsample=0.8, colsample=0.8",
                 lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, subsample=0.8,
                                            subsample_freq=1, colsample_bytree=0.8, n_jobs=2, random_state=SEED,
                                            verbose=-1), None),
    "XGBoost": ("Boosting", "n_estimators=200, lr=0.1, max_depth=6, hist, subsample=0.8",
                lambda: xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,
                                          colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=SEED,
                                          eval_metric="logloss"), None),
    "MLP": ("Neural network", "hidden=(64,32), relu, adam, alpha=1e-4, batch=1024, 20 epochs, trained on <=300k rows",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                  batch_size=1024, max_iter=20, random_state=SEED)),
            300_000),
}
