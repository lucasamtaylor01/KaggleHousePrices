import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from category_encoders import TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


def _zero_pct(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Returns the share of zero values per column, sorted descending.

    Args:
        df (pd.DataFrame): DataFrame to inspect
        cols: columns to inspect

    Returns:
        pd.DataFrame: ZERO_PCT per column, only columns with zeros
    """

    df_zero = (df[cols] == 0).mean().to_frame("ZERO_PCT").sort_values(
        "ZERO_PCT", ascending=False
    )
    df_zero = df_zero[df_zero["ZERO_PCT"] > 0]

    return df_zero


def _combine_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Derives combined/relative features (baths, ages, areas) and drops the
    original columns they replace. Rule-based only, so it applies the same
    way to train and test. It should be used JUST on feature_engineering().

    Args:
        df_clean (pd.DataFrame): cleaned DataFrame

    Returns:
        pd.DataFrame: DataFrame with combined features
    """

    df_comb = df_clean.copy()
    df_comb.columns = df_comb.columns.str.upper()

    df_comb["TOTALBATHS"] = (
        df_comb["FULLBATH"]
        + (0.5 * df_comb["HALFBATH"])
        + df_comb["BSMTFULLBATH"]
        + (0.5 * df_comb["BSMTHALFBATH"])
    )
    df_comb = df_comb.drop(
        columns=["FULLBATH", "HALFBATH", "BSMTFULLBATH", "BSMTHALFBATH"]
    )

    df_comb = df_comb.drop(columns=["GARAGEAREA", "GARAGECARS"])

    df_comb["YEARBUILT"] = df_comb["YRSOLD"] - df_comb["YEARBUILT"]
    df_comb["YEARREMODADD"] = df_comb["YRSOLD"] - df_comb["YEARREMODADD"]
    df_comb = df_comb.drop(columns=["YRSOLD", "MOSOLD"])

    df_comb["BSMTFINSF"] = df_comb["BSMTFINSF1"] + df_comb["BSMTFINSF2"]
    df_comb["TOTALFLRSF"] = df_comb["1STFLRSF"] + df_comb["2NDFLRSF"]
    df_comb = df_comb.drop(columns=["BSMTFINSF1", "BSMTFINSF2", "1STFLRSF", "2NDFLRSF"])

    return df_comb


def _classify_variables(df_comb: pd.DataFrame, df_type: int, artifacts: dict):
    """Classifies columns into qualitative nominal and quantitative
    discrete/continuous groups. The grouping is decided JUST from the train
    DataFrame (df_type == 0) and stored in artifacts, then reused as-is for
    the test DataFrame (df_type == 1) so both go through the exact same
    columns. It should be used JUST on feature_engineering().

    Args:
        df_comb (pd.DataFrame): DataFrame with combined features
        df_type (int): 0 for train, 1 for test
        artifacts (dict): fitted objects/decisions shared between train and test

    Returns:
        tuple: (nom_cols, disc_cols, cont_cols)
    """

    if df_type == 0:
        ord_cols = [
            "LOTSHAPE", "UTILITIES", "LANDSLOPE", "OVERALLQUAL",
            "OVERALLCOND", "EXTERQUAL", "EXTERCOND", "BSMTQUAL",
            "BSMTCOND", "BSMTEXPOSURE", "BSMTFINTYPE1", "BSMTFINTYPE2",
            "HEATINGQC", "ELECTRICAL", "KITCHENQUAL", "FUNCTIONAL",
            "FIREPLACEQU", "GARAGEFINISH", "GARAGEQUAL", "GARAGECOND",
            "PAVEDDRIVE", "FENCE",
        ]

        nom_cols = [
            "MSSUBCLASS", "MSZONING", "STREET", "ALLEY", "LANDCONTOUR",
            "LOTCONFIG", "NEIGHBORHOOD", "CONDITION1", "CONDITION2",
            "BLDGTYPE", "HOUSESTYLE", "ROOFSTYLE", "ROOFMATL",
            "EXTERIOR1ST", "EXTERIOR2ND", "MASVNRTYPE", "FOUNDATION",
            "HEATING", "CENTRALAIR", "GARAGETYPE", "MISCFEATURE",
            "SALETYPE", "SALECONDITION",
        ]

        disc_cols = list(
            df_comb.select_dtypes(include="int64")
            .drop(columns=["ID"] + ord_cols + nom_cols, errors="ignore")
            .columns
        )

        cont_cols = list(
            df_comb.select_dtypes(include="float64")
            .drop(columns=["SALEPRICE"], errors="ignore")
            .columns
        )

        artifacts["nom_cols"] = nom_cols
        artifacts["disc_cols"] = disc_cols
        artifacts["cont_cols"] = cont_cols
    else:
        nom_cols = artifacts["nom_cols"]
        disc_cols = artifacts["disc_cols"]
        cont_cols = artifacts["cont_cols"]

    return nom_cols, disc_cols, cont_cols


def _ordinal_treatment(df_comb: pd.DataFrame) -> pd.DataFrame:
    """Maps qualitative ordinal columns to numeric grades and converts the
    binary ones (fence, paved drive, lot shape) to boolean flags. Rule-based
    only, so it applies the same way to train and test. It should be used
    JUST on feature_engineering().

    Args:
        df_comb (pd.DataFrame): DataFrame with combined features

    Returns:
        pd.DataFrame: DataFrame with ordinal variables treated
    """

    df_ord = df_comb.copy()

    # GRADES 1-10
    grade_map = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3}
    df_ord["OVERALLCOND"] = df_ord["OVERALLCOND"].map(grade_map)
    df_ord["OVERALLQUAL"] = df_ord["OVERALLQUAL"].map(grade_map)

    # FIVE QUALITIES
    qual5_map = {"NA": 0, "PO": 0, "FA": 2, "TA": 2, "GD": 3, "EX": 3}
    df_ord["EXTERQUAL"] = df_ord["EXTERQUAL"].map(qual5_map)
    df_ord["EXTERCOND"] = df_ord["EXTERCOND"].map(qual5_map)
    df_ord["HEATINGQC"] = df_ord["HEATINGQC"].map(qual5_map)
    df_ord["KITCHENQUAL"] = df_ord["KITCHENQUAL"].map(qual5_map)

    # SIX QUALITIES
    qual6_map = {"NA": 0, "PO": 0, "FA": 2, "TA": 2, "GD": 3, "EX": 3}
    df_ord["BSMTQUAL"] = df_ord["BSMTQUAL"].map(qual6_map)
    df_ord["BSMTCOND"] = df_ord["BSMTCOND"].map(qual6_map)
    df_ord["FIREPLACEQU"] = df_ord["FIREPLACEQU"].map(qual6_map)
    df_ord["GARAGEQUAL"] = df_ord["GARAGEQUAL"].map(qual6_map)
    df_ord["GARAGECOND"] = df_ord["GARAGECOND"].map(qual6_map)

    # BSMT TYPE
    bsmt_fin_map = {"NA": 0, "UNF": 1, "LWQ": 2, "REC": 2, "BLQ": 2, "ALQ": 3, "GLQ": 3}
    df_ord["BSMTFINTYPE1"] = df_ord["BSMTFINTYPE1"].map(bsmt_fin_map)
    df_ord["BSMTFINTYPE2"] = df_ord["BSMTFINTYPE2"].map(bsmt_fin_map)

    # SPECIAL CASES
    df_ord["LANDSLOPE"] = df_ord["LANDSLOPE"].map({"NA": 0, "SEV": 3, "MOD": 2, "GTL": 1})
    df_ord["BSMTEXPOSURE"] = df_ord["BSMTEXPOSURE"].map(
        {"NA": 0, "NO": 1, "MN": 1, "AV": 1, "GD": 2}
    )
    df_ord["GARAGEFINISH"] = df_ord["GARAGEFINISH"].map(
        {"NA": 0, "UNF": 1, "RFN": 2, "FIN": 3}
    )
    df_ord["ELECTRICAL"] = df_ord["ELECTRICAL"].map(
        {"MIX": 1, "FUSEP": 1, "FUSEF": 2, "FUSEA": 2, "SBRKR": 3}
    )
    df_ord["FENCE"] = df_ord["FENCE"].map(
        {"NA": 0, "MNWW": 1, "GDWO": 1, "MNPRV": 1, "GDPRV": 1}
    )
    df_ord["FUNCTIONAL"] = df_ord["FUNCTIONAL"].map(
        {
            "SEV": 0, "MAJ2": 0, "MAJ1": 0, "MOD": 1,
            "MIN2": 1, "MIN1": 1, "TYP": 2, "NA": 0,
        }
    )

    # BOOLEAN CONVERSION
    df_ord = df_ord.rename(columns={"FENCE": "HASFENCE"})
    df_ord["HASFENCE"] = df_ord["HASFENCE"].astype("bool")

    df_ord["PAVEDDRIVE"] = df_ord["PAVEDDRIVE"].map(
        {"NA": 0, "MNWW": 1, "GDWO": 1, "MNPRV": 1, "GDPRV": 1}
    )
    df_ord = df_ord.rename(columns={"PAVEDDRIVE": "HASPAVEDDRIVE"})
    df_ord["HASPAVEDDRIVE"] = df_ord["HASPAVEDDRIVE"].astype("bool")

    df_ord["LOTSHAPE"] = df_ord["LOTSHAPE"].map({"REG": 1, "IR1": 0, "IR2": 0, "IR3": 0})
    df_ord = df_ord.rename(columns={"LOTSHAPE": "HASREGULARLOTSHAPE"})
    df_ord["HASREGULARLOTSHAPE"] = df_ord["HASREGULARLOTSHAPE"].astype("bool")

    return df_ord


def _remove_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Removes outliers from continuous columns using an IQR fence, keeping
    zeros untouched for zero-heavy columns. Drops rows, so it must only run
    on the train DataFrame. It should be used JUST on feature_engineering().

    Args:
        df (pd.DataFrame): DataFrame with continuous variables treated
        cols (list): continuous columns to check for outliers

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """

    ZERO_PCT_THRESH = 0.5

    df_zero = _zero_pct(df, cols)
    df_zero = df_zero[df_zero["ZERO_PCT"] > ZERO_PCT_THRESH]
    zero_heavy_final_cols = df_zero.index

    df_out = df.copy()

    for col in cols:
        if col in zero_heavy_final_cols:
            non_zero_values = df.loc[df[col] > 0, col]
            Q1 = non_zero_values.quantile(0.25)
            Q3 = non_zero_values.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            mask_outliers = (df_out[col] == 0) | (
                (df_out[col] > lower_bound) & (df_out[col] < upper_bound)
            )
            df_out = df_out[mask_outliers]
        else:
            Q1 = df_out[col].quantile(0.15)
            Q3 = df_out[col].quantile(0.85)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            mask_outliers = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
            df_out = df_out[~mask_outliers]

    pct_removed = round((df.shape[0] - df_out.shape[0]) / df.shape[0] * 100, 2)

    return df_out


def _continuous_treatment(
    df_ord: pd.DataFrame, cont_cols: list, df_type: int, artifacts: dict
) -> pd.DataFrame:
    """Treats quantitative continuous columns by skewness: power-transforms
    high-skew variables, flags zero-heavy ones as booleans, log-transforms the
    rest, and removes outliers from the resulting set.

    Every decision (which columns are high/moderate/symmetric skew, which are
    zero-heavy, the fitted PowerTransformers) is made JUST from the train
    DataFrame (df_type == 0) and stored in artifacts. The test DataFrame
    (df_type == 1) reuses those same decisions/objects via `.transform()`
    instead of `.fit_transform()`, and skips outlier removal since it must
    keep every row. It should be used JUST on feature_engineering().

    Args:
        df_ord (pd.DataFrame): DataFrame with ordinal variables treated
        cont_cols (list): continuous columns
        df_type (int): 0 for train, 1 for test
        artifacts (dict): fitted objects/decisions shared between train and test

    Returns:
        pd.DataFrame: DataFrame with continuous variables treated
    """

    df_cont = df_ord.copy()

    if df_type == 0:
        df_skew = df_cont[cont_cols].skew().to_frame("SKEWNESS")
        high_skew_cols = list(df_skew[df_skew["SKEWNESS"].abs() > 1.0].index)

        # HIGH SKEW TREATMENT
        ZERO_PCT_THRESH = 0.50
        df_zero = _zero_pct(df_cont, cont_cols)
        df_zero = df_zero[df_zero["ZERO_PCT"] > ZERO_PCT_THRESH]
        zero_heavy_cols = list(df_zero.index)

        high_skew_cols = list(set(high_skew_cols) - set(zero_heavy_cols))
        df_zero = _zero_pct(df_cont, zero_heavy_cols)

        pt_map = {}
        for col in high_skew_cols:
            pt = PowerTransformer()
            df_cont[col + "_TRANSFORMED"] = pt.fit_transform(df_cont[[col]])[:, 0]
            pt_map[col] = pt

        pt_cols = [c for c in df_cont.columns if "_TRANSFORMED" in c]

        ZERO_PCT_THRESH = 70
        zero_flag_cols = list(
            df_zero[df_zero["ZERO_PCT"] > ZERO_PCT_THRESH].index
        )
        for old_col in zero_flag_cols:
            new_col = "HAS" + old_col
            df_cont = df_cont.rename(columns={old_col: new_col})
            df_cont[new_col] = np.where(df_cont[new_col] == 0, 0, 1).astype("bool")

        zero_heavy_cols = list(set(zero_heavy_cols) - set(zero_flag_cols))

        # REMAINING SKEWED VARIABLES: BOOLEAN FLAG + LOG TRANSFORM
        log_flag_cols = zero_heavy_cols
        for old_col in log_flag_cols:
            df_cont["HAS" + old_col] = np.where(df_cont[old_col] == 0, 0, 1).astype("bool")
            df_cont[old_col + "_LOG"] = np.log1p(df_cont[old_col])

        log_cols = [c for c in df_cont.columns if "_LOG" in c]
        high_skew_final = list(set(pt_cols).union(set(log_cols)))

        # MODERATE SKEW TREATMENT
        bsmt_unf_pt = PowerTransformer()
        df_cont["BSMTUNFSF_TRANSFORMED"] = bsmt_unf_pt.fit_transform(
            df_cont[["BSMTUNFSF"]]
        )[:, 0]

        mod_skew_final = list(set(["BSMTUNFSF_TRANSFORMED"]).union(set(log_cols)))

        # APPROXIMATELY SYMMETRIC TREATMENT
        df_cont["LOTFRONTAGE_LOG"] = np.log1p(df_cont["LOTFRONTAGE"])
        df_cont["HASLOTFRONTAGE"] = np.where(df_cont["LOTFRONTAGE"] == 0, 0, 1).astype("bool")
        sym_final = ["LOTFRONTAGE_LOG"]

        final_cont_cols = list(set(sym_final).union(set(mod_skew_final)))
        final_cont_cols = list(set(final_cont_cols).union(set(high_skew_final)))

        df_cont = _remove_outliers(df_cont, final_cont_cols)

        artifacts["cont_pt_map"] = pt_map
        artifacts["cont_bsmt_unf_pt"] = bsmt_unf_pt
        artifacts["cont_zero_flag_cols"] = zero_flag_cols
        artifacts["cont_log_flag_cols"] = log_flag_cols

    else:
        pt_map = artifacts["cont_pt_map"]
        bsmt_unf_pt = artifacts["cont_bsmt_unf_pt"]
        zero_flag_cols = artifacts["cont_zero_flag_cols"]
        log_flag_cols = artifacts["cont_log_flag_cols"]

        for col, pt in pt_map.items():
            df_cont[col + "_TRANSFORMED"] = pt.transform(df_cont[[col]])[:, 0]

        for old_col in zero_flag_cols:
            new_col = "HAS" + old_col
            df_cont = df_cont.rename(columns={old_col: new_col})
            df_cont[new_col] = np.where(df_cont[new_col] == 0, 0, 1).astype("bool")

        for old_col in log_flag_cols:
            df_cont["HAS" + old_col] = np.where(df_cont[old_col] == 0, 0, 1).astype("bool")
            df_cont[old_col + "_LOG"] = np.log1p(df_cont[old_col])

        df_cont["BSMTUNFSF_TRANSFORMED"] = bsmt_unf_pt.transform(
            df_cont[["BSMTUNFSF"]]
        )[:, 0]

        df_cont["LOTFRONTAGE_LOG"] = np.log1p(df_cont["LOTFRONTAGE"])
        df_cont["HASLOTFRONTAGE"] = np.where(df_cont["LOTFRONTAGE"] == 0, 0, 1).astype("bool")

        # NO OUTLIER REMOVAL: every test row must be kept for prediction

    return df_cont


def _discrete_treatment(
    df_cont: pd.DataFrame, disc_cols: list, df_type: int, artifacts: dict
) -> pd.DataFrame:
    """Flags zero-heavy quantitative discrete columns as booleans. Which
    columns get flagged is decided JUST from the train DataFrame (df_type ==
    0) and stored in artifacts, then reused as-is for the test DataFrame
    (df_type == 1). It should be used JUST on feature_engineering().

    Args:
        df_cont (pd.DataFrame): DataFrame with continuous variables treated
        disc_cols (list): discrete columns
        df_type (int): 0 for train, 1 for test
        artifacts (dict): fitted objects/decisions shared between train and test

    Returns:
        pd.DataFrame: DataFrame with discrete variables treated
    """

    df_disc = df_cont.copy()

    if df_type == 0:
        df_disc_zero = _zero_pct(df_disc, disc_cols)
        ZERO_PCT_THRESH = 0.50
        zero_flag_cols = list(
            df_disc_zero[df_disc_zero["ZERO_PCT"] > ZERO_PCT_THRESH].index
        )
        artifacts["disc_zero_flag_cols"] = zero_flag_cols
    else:
        zero_flag_cols = artifacts["disc_zero_flag_cols"]

    for old_col in zero_flag_cols:
        new_col = "HAS" + old_col
        df_disc = df_disc.rename(columns={old_col: new_col})
        df_disc[new_col] = np.where(df_disc[new_col] == 0, 0, 1).astype("bool")

    return df_disc


def _target_treatment(df_disc: pd.DataFrame) -> pd.DataFrame:
    """Log-transforms the target variable. SALEPRICE only exists on the train
    DataFrame, so it should be used JUST on feature_engineering() for
    df_type == 0.

    Args:
        df_disc (pd.DataFrame): DataFrame with discrete variables treated

    Returns:
        pd.DataFrame: DataFrame with SALEPRICE_LOG added
    """

    df_target = df_disc.copy()
    df_target["SALEPRICE_LOG"] = np.log1p(df_target[["SALEPRICE"]])

    return df_target


def _nominal_treatment(
    df: pd.DataFrame, nom_cols: list, df_type: int, artifacts: dict
) -> pd.DataFrame:
    """Target-encodes qualitative nominal columns. The encoder is FIT only
    on the train DataFrame (df_type == 0), where SALEPRICE_LOG is available
    to encode against, and stored in artifacts. On the test DataFrame
    (df_type == 1), which has no target column, it is only ever
    `.transform()`-ed with that already-fitted encoder — never refit — to
    avoid leaking test statistics into the encoding. It should be used JUST
    on feature_engineering().

    Args:
        df (pd.DataFrame): DataFrame with discrete variables treated (train)
            or target treated (train) / discrete treated (test)
        nom_cols (list): nominal columns
        df_type (int): 0 for train, 1 for test
        artifacts (dict): fitted objects/decisions shared between train and test

    Returns:
        pd.DataFrame: DataFrame with nominal variables encoded
    """

    df_nom = df.copy()

    if df_type == 0:
        encoder = TargetEncoder(cols=nom_cols, smoothing=5.0)
        df_nom[nom_cols] = encoder.fit_transform(df_nom[nom_cols], df_nom["SALEPRICE_LOG"])
        artifacts["nom_encoder"] = encoder
    else:
        encoder = artifacts["nom_encoder"]
        df_nom[nom_cols] = encoder.transform(df_nom[nom_cols])

    return df_nom


def _remove_high_vif(df: pd.DataFrame, threshold: float, exclude_cols: list) -> list:
    """Iteratively drops the column with the highest Variance Inflation Factor
    until every remaining column is below the threshold.

    Args:
        df (pd.DataFrame): candidate model DataFrame
        threshold (float): maximum VIF allowed
        exclude_cols (list): columns to keep out of the VIF check

    Returns:
        list: columns that should be dropped for high multicollinearity
    """

    X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    X = X.drop(columns=X.select_dtypes(include=["bool"]).columns)
    X = add_constant(X).astype(float)

    dropped_cols = []

    while True:
        vif_data = pd.DataFrame()
        vif_data["VARIAVEL"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]
        vif_data = vif_data[vif_data["VARIAVEL"] != "const"]

        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            break

        var_to_drop = vif_data.sort_values("VIF", ascending=False).iloc[0]["VARIAVEL"]
        X = X.drop(columns=[var_to_drop])
        dropped_cols.append(var_to_drop)

    return dropped_cols


def _select_model_features(df_nom: pd.DataFrame, df_type: int, artifacts: dict) -> pd.DataFrame:
    """Drops the raw columns replaced by engineered features, then removes
    high-multicollinearity columns (VIF) and columns weakly correlated with
    the target. Both selections need the target and are only computable on
    the train DataFrame (df_type == 0); the resulting column lists are stored
    in artifacts and simply dropped again from the test DataFrame (df_type ==
    1), which has no target to recompute them from. It should be used JUST on
    feature_engineering().

    Args:
        df_nom (pd.DataFrame): DataFrame with nominal variables encoded
        df_type (int): 0 for train, 1 for test
        artifacts (dict): fitted objects/decisions shared between train and test

    Returns:
        pd.DataFrame: model-ready DataFrame
    """

    df_model = df_nom.copy()

    original_to_drop = [
        "LOTAREA", "TOTALBSMTSF", "GRLIVAREA", "LOWQUALFINSF",
        "OPENPORCHSF", "MASVNRAREA", "WOODDECKSF", "ENCLOSEDPORCH",
        "3SSNPORCH", "SCREENPORCH", "BSMTUNFSF", "BSMTFINSF",
        "TOTALFLRSF", "LOTFRONTAGE", "SALEPRICE", "UTILITIES",
    ]
    df_model = df_model.drop(columns=original_to_drop, errors="ignore")

    if df_type == 0:
        vif_dropped_cols = _remove_high_vif(
            df_model, threshold=5, exclude_cols=["ID", "SALEPRICE_LOG"]
        )
        artifacts["vif_dropped_cols"] = vif_dropped_cols
    else:
        vif_dropped_cols = artifacts["vif_dropped_cols"]
    df_model = df_model.drop(columns=vif_dropped_cols, errors="ignore")

    if df_type == 0:
        numeric_columns = df_model.select_dtypes(include=["int64", "float64"]).columns
        df_model_corr = df_model[numeric_columns].corr()
        df_model_corr = df_model_corr.drop(index=["ID", "SALEPRICE_LOG"], columns="ID")

        df_model_corr = pd.DataFrame(df_model_corr["SALEPRICE_LOG"]).rename(
            columns={"SALEPRICE_LOG": "CORRELATION"}
        )
        df_model_corr["CORRELATION"] = df_model_corr["CORRELATION"].abs()
        df_model_corr = df_model_corr.sort_values(by=["CORRELATION"], ascending=False)

        low_corr_dropped_cols = list(
            df_model_corr[df_model_corr["CORRELATION"] < 0.3].index
        )
        artifacts["low_corr_dropped_cols"] = low_corr_dropped_cols
    else:
        low_corr_dropped_cols = artifacts["low_corr_dropped_cols"]
    df_model = df_model.drop(columns=low_corr_dropped_cols, errors="ignore")

    return df_model


def feature_engineering(df_clean: pd.DataFrame, df_type: int, artifacts: dict = None):
    try:
        if df_type == 0:
            artifacts = {}
            print("[!] STARTING FEATURE ENGINEERING PROCESS OF TRAIN DATAFRAME ...\n")
        else:
            print("[!] STARTING FEATURE ENGINEERING PROCESS OF TEST DATAFRAME ...\n")

        df_comb = _combine_features(df_clean)
        print("[1/6] FEATURES COMBINED")

        nom_cols, disc_cols, cont_cols = _classify_variables(df_comb, df_type, artifacts)

        df_ord = _ordinal_treatment(df_comb)
        print("[2/6] QUALITATIVE ORDINAL VARIABLES TREATED")

        df_cont = _continuous_treatment(df_ord, cont_cols, df_type, artifacts)
        print("[3/6] QUANTITATIVE CONTINUOUS VARIABLES TREATED")

        df_disc = _discrete_treatment(df_cont, disc_cols, df_type, artifacts)
        print("[4/6] QUANTITATIVE DISCRETE VARIABLES TREATED")

        if df_type == 0:
            df_target = _target_treatment(df_disc)
        else:
            df_target = df_disc

        df_nom = _nominal_treatment(df_target, nom_cols, df_type, artifacts)
        print("[5/6] TARGET TRANSFORMED AND NOMINAL VARIABLES ENCODED")

        df_model = _select_model_features(df_nom, df_type, artifacts)
        print("[6/6] MODEL DATAFRAME PREPARED\n")

        if df_type == 0:
            print("[!] FEATURE ENGINEERING PROCESS OF TRAIN DATAFRAME WAS SUCCESSFUL...\n")
        else:
            print("[!] FEATURE ENGINEERING PROCESS OF TEST DATAFRAME WAS SUCCESSFUL...\n")

        print(75 * "-", "\n")
        return df_model, artifacts

    except:
        print("[!] AN ERROR WAS OCURRED ON FEATURE ENGINEERING PROCESS. \n")
