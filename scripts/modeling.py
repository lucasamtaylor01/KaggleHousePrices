import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_log_error


def _fit_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fits the Ridge and Gradient Boosting models with their tuned
    hyperparameters. It should be used JUST on modeling().

    Args:
        X_train (pd.DataFrame): train features
        y_train (pd.Series): train target (SALEPRICE_LOG)

    Returns:
        dict: fitted models keyed by name
    """

    ridge_model = Ridge(alpha=2.5, random_state=0, solver="svd")
    ridge_model.fit(X_train, y_train)

    gb_model = GradientBoostingRegressor(
        learning_rate=0.045,
        max_depth=6,
        max_features=0.7,
        min_samples_leaf=40,
        min_samples_split=90,
        n_estimators=575,
        random_state=0,
        subsample=0.85,
    )
    gb_model.fit(X_train, y_train)

    return {"RIDGE": ridge_model, "GB": gb_model}


def _ensemble_weights(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Weights each model by the inverse of its train RMSLE, so the more
    accurate model contributes more to the ensemble prediction. It should be
    used JUST on modeling().

    Args:
        models (dict): fitted models keyed by name
        X_train (pd.DataFrame): train features
        y_train (pd.Series): train target (SALEPRICE_LOG)

    Returns:
        dict: normalized weight per model name
    """

    inv_rmsle = {}
    for name, model in models.items():
        y_pred = model.predict(X_train)
        rmsle = root_mean_squared_log_error(y_true=y_train, y_pred=y_pred)
        inv_rmsle[name] = 1 / rmsle

    total_inv_rmsle = sum(inv_rmsle.values())
    weights = {name: value / total_inv_rmsle for name, value in inv_rmsle.items()}

    return weights


def _predict_ensemble(models: dict, weights: dict, X: pd.DataFrame) -> np.ndarray:
    """Predicts SALEPRICE_LOG as the weighted average of every fitted model.
    It should be used JUST on modeling().

    Args:
        models (dict): fitted models keyed by name
        weights (dict): normalized weight per model name
        X (pd.DataFrame): features to predict on

    Returns:
        np.ndarray: weighted SALEPRICE_LOG predictions
    """

    y_pred = np.zeros(X.shape[0])
    for name, model in models.items():
        y_pred += weights[name] * model.predict(X)

    return y_pred


def modeling(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    try:
        print("[!] STARTING MODELING PROCESS ...\n")

        X_train = df_train.drop(columns=["SALEPRICE_LOG"])
        y_train = df_train["SALEPRICE_LOG"]
        X_test = df_test[X_train.columns]

        models = _fit_models(X_train, y_train)
        print("[1/3] MODELS FITTED")

        weights = _ensemble_weights(models, X_train, y_train)
        print("[2/3] ENSEMBLE WEIGHTS COMPUTED")

        y_pred_test = _predict_ensemble(models, weights, X_test)
        print("[3/3] TEST PREDICTIONS GENERATED\n")

        submission = pd.DataFrame(
            {"Id": df_test["ID"], "SalePrice": np.expm1(y_pred_test)}
        )

        print("[!] MODELING PROCESS WAS SUCCESSFUL...\n")
        print(75 * "-", "\n")
        return submission

    except:
        print("[!] AN ERROR WAS OCURRED ON MODELING PROCESS. \n")
