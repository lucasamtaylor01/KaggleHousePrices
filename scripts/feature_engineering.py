import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from category_encoders import TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


def _zero_pct(df: pd.DataFrame, cols: list)-> pd.DataFrame:
    return (df[cols] == 0).mean().to_frame("ZERO_PCT").sort_values("ZERO_PCT", ascending=False)

def _classify_variables(df_clean: pd.DataFrame, df_type: int)-> pd.DataFrame:
    ord_cols = [
        "LOTSHAPE", "UTILITIES", "LANDSLOPE", "OVERALLQUAL",
        "OVERALLCOND", "EXTERQUAL", "EXTERCOND", "BSMTQUAL",
        "BSMTCOND", "BSMTEXPOSURE", "BSMTFINTYPE1", "BSMTFINTYPE2",
        "HEATINGQC", "ELECTRICAL", "KITCHENQUAL", "FUNCTIONAL",
        "FIREPLACEQU", "GARAGEFINISH", "GARAGEQUAL", "GARAGECOND",
        "PAVEDDRIVE", "FENCE"
    ]

    nom_cols = [
        "MSSUBCLASS", "MSZONING", "STREET", "ALLEY", "LANDCONTOUR",
        "LOTCONFIG", "NEIGHBORHOOD", "CONDITION1", "CONDITION2",
        "BLDGTYPE", "HOUSESTYLE", "ROOFSTYLE", "ROOFMATL",
        "EXTERIOR1ST", "EXTERIOR2ND", "MASVNRTYPE", "FOUNDATION",
        "HEATING", "CENTRALAIR", "GARAGETYPE", "MISCFEATURE",
        "SALETYPE", "SALECONDITION"
    ]


    disc_cols = df_clean.select_dtypes(include='int64').drop(
        columns=["ID"] + ord_cols + nom_cols, errors="ignore"
    )

    cont_cols = list(df_clean.select_dtypes(include='float64').drop(columns=["SALEPRICE"], errors="ignore").columns)
    bool_cols = list(df_clean.select_dtypes(include='bool').columns)

    if df_type == 0:
        target = df_clean["SALEPRICE"]
        return ord_cols, nom_cols, disc_cols, bool_cols, cont_cols, target
    if df_type == 1:
        return ord_cols, nom_cols, disc_cols, bool_cols, cont_cols

def _ord_cols_treatment(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_ord = df_clean.copy()
    drop_cols = ["UTILITIES"]
    add_cols = []

    # GRADES 1-10
    grade_map = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 2,
        8: 3,
        9: 3,
        10: 3,
    }
    df_ord["OVERALLCOND"] = df_ord["OVERALLCOND"].map(grade_map)
    df_ord["OVERALLQUAL"] = df_ord["OVERALLQUAL"].map(grade_map)

    # FIVE QUALITIES
    qual5_map = {
        "NA": 0,
        "PO": 0,
        "FA": 2,
        "TA": 2,
        "GD": 3,
        "EX": 3,
    }
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
    bsmt_fin_map = {
        "NA": 0,
        "UNF": 1,
        "LWQ": 2,
        "REC": 2,
        "BLQ": 2,
        "ALQ": 3,
        "GLQ": 3,
    }
    df_ord["BSMTFINTYPE1"] = df_ord["BSMTFINTYPE1"].map(bsmt_fin_map)
    df_ord["BSMTFINTYPE2"] = df_ord["BSMTFINTYPE2"].map(bsmt_fin_map)

    # SPECIAL CASES
    df_ord["LANDSLOPE"] = df_ord["LANDSLOPE"].map(
        {"NA": 0, "SEV": 3, "MOD": 2, "GTL": 1}
    )
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
            "SEV": 0,
            "MAJ2": 0,
            "MAJ1": 0,
            "MOD": 1,
            "MIN2": 1,
            "MIN1": 1,
            "TYP": 2,
            "NA": 0,
        }
    )

    # BOOLEAN CONVERSION
    df_ord = df_ord.rename(columns={"FENCE": "HASFENCE"})
    df_ord["HASFENCE"] = df_ord["HASFENCE"].astype("bool")
    drop_cols.append("FENCE")
    add_cols.append("HASFENCE")

    df_ord["PAVEDDRIVE"] = df_ord["PAVEDDRIVE"].map(
        {"NA": 0, "MNWW": 1, "GDWO": 1, "MNPRV": 1, "GDPRV": 1}
    )
    df_ord = df_ord.rename(columns={"PAVEDDRIVE": "HASPAVEDDRIVE"})
    df_ord["HASPAVEDDRIVE"] = df_ord["HASPAVEDDRIVE"].astype("bool")
    drop_cols.append("PAVEDDRIVE")
    add_cols.append("HASPAVEDDRIVE")


    df_ord["LOTSHAPE"] = df_ord["LOTSHAPE"].map(
        {"REG": 1, "IR1": 0, "IR2": 0, "IR3": 0}
    )
    df_ord = df_ord.rename(columns={"LOTSHAPE": "HASREGULARLOTSHAPE"})
    df_ord["HASREGULARLOTSHAPE"] = df_ord["HASREGULARLOTSHAPE"].astype("bool")
    drop_cols.append("LOTSHAPE")
    add_cols.append("HASREGULARLOTSHAPE")

    # UPDATE ORD_COLS
    ord_cols = list(set(ord_cols) - set(drop_cols))
    bool_cols = list(set(bool_cols).union(set(add_cols)))

    return df_ord

def _outlier_treatment(df: pd.DataFrame, cols: list, var_type: str) -> pd.DataFrame:
    QUANTILE_BOUNDS = {"cont": (0.2, 0.8), "disc": (0.2, 0.8)}
    ZERO_PCT_THRESH = 0.70

    q_low, q_high = QUANTILE_BOUNDS[var_type]

    zero_df = _zero_pct(df, cols)
    zero_df = zero_df[zero_df['ZERO_PCT'] > ZERO_PCT_THRESH]
    zero_heavy_cols = zero_df.index

    df_out = df.copy()

    for col in cols:

        if col in zero_heavy_cols:
            non_zero_values = df.loc[df[col] > 0, col]
            Q1 = non_zero_values.quantile(q_low)
            Q3 = non_zero_values.quantile(q_high)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            mask_outliers = (df_out[col] == 0) | ((df_out[col] > lower_bound) & (df_out[col] < upper_bound))
            df_out = df_out[mask_outliers]

        else:
            Q1 = df_out[col].quantile(q_low)
            Q3 = df_out[col].quantile(q_high)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            mask_outliers = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
            df_out = df_out[~mask_outliers]


    pct_removed = round((df.shape[0] - df_out.shape[0]) / df.shape[0] * 100, 2)
    print(f"{pct_removed}% removed")

    return df_out

def _cont_cols_treatment(df_ord: pd.DataFrame, cont_cols: list) -> pd.DataFrame:

    df_cont = df_ord.copy()
    skew_df = df_cont[cont_cols].skew().to_frame("SKEWNESS")

    sym_cols = skew_df[skew_df["SKEWNESS"].abs() < 0.5].index
    mod_skew_cols = skew_df[skew_df["SKEWNESS"].abs().between(0.5, 1.0)].index
    high_skew_cols = skew_df[skew_df["SKEWNESS"].abs() > 1.0].index


    zero_heavy_cols = [
        "3SSNPORCH",
        "LOWQUALFINSF",
        "BSMTFINSF2",
        "SCREENPORCH",
        "ENCLOSEDPORCH",
        "MASVNRAREA",
        "OPENPORCHSF",
        "BSMTFINSF1",
        "WOODDECKSF",
    ]

    high_skew_cols = list(set(high_skew_cols) - set(zero_heavy_cols))
    zero_df = _zero_pct(df_cont, zero_heavy_cols)

    pt_map = {}

    for col in high_skew_cols:
        pt = PowerTransformer()
        df_cont[col + "_TRANSFORMED"] = pt.fit_transform(df_cont[[col]])[:, 0]
        pt_map[col] = pt


    pt_cols = [c for c in df_cont.columns if "_TRANSFORMED" in c]


    drop_cols = []
    add_cols = []

    ZERO_PCT_THRESH = 70

    zero_flag_map = {
        col: "HAS" + col
        for col in zero_df[zero_df["ZERO_PCT"] > ZERO_PCT_THRESH].index
    }

    for old_col, new_col in zero_flag_map.items():
        df_cont = df_cont.rename(columns={old_col: new_col})
        df_cont[new_col] = np.where(df_cont[new_col] == 0, 0, 1).astype("bool")
        drop_cols.append(old_col)
        add_cols.append(new_col)


    zero_heavy_cols = list(set(zero_heavy_cols) - set(drop_cols))
    cont_cols = list(set(cont_cols) - set(drop_cols))
    bool_cols = list(set(bool_cols).union(set(add_cols)))

    # HANDLE REMAINING VARIABLES (moderate zero presence: bool flag + log1p)
    log_flag_map = {col: "HAS" + col for col in zero_heavy_cols}

    for old_col, new_col in log_flag_map.items():
        df_cont[new_col] = np.where(df_cont[old_col] == 0, 0, 1).astype("bool")
        df_cont[old_col + "_LOG"] = np.log1p(df_cont[old_col])
        add_cols.append(new_col)

    log_cols = [c for c in df_cont.columns if "_LOG" in c]
    bool_cols = list(set(bool_cols).union(set(add_cols)))

    high_skew_final = list(set(pt_cols).union(set(log_cols)))

    zero_df = _zero_pct(df_cont, mod_skew_cols)

    add_cols = []

    df_cont["HAS2NDFLRSF"] = np.where(df_cont["2NDFLRSF"] == 0, 0, 1).astype("bool")

    df_cont["2NDFLRSF_LOG"] = np.log1p(df_cont["2NDFLRSF"])
    add_cols.append("HAS2NDFLRSF")

    bool_cols = list(set(bool_cols).union(set(add_cols)))

    bsmt_unf_pt = PowerTransformer()
    df_cont["BSMTUNFSF_TRANSFORMED"] = bsmt_unf_pt.fit_transform(df_cont[['BSMTUNFSF']])[:, 0]

    log_cols = ['2NDFLRSF_LOG']
    pt_cols = ['BSMTUNFSF_TRANSFORMED']
    mod_skew_final = list(set(pt_cols).union(set(log_cols)))
    
    zero_df = _zero_pct(df_cont, sym_cols)

    sym_final = []
    add_cols = []

    scaler = StandardScaler()
    df_cont['GARAGEAREA_SCALED'] = scaler.fit_transform(df_cont[['GARAGEAREA']])[:, 0]
    sym_final.append('GARAGEAREA_SCALED')


    df_cont["LOTFRONTAGE_LOG"] = np.log1p(df_cont["LOTFRONTAGE"])
    sym_final.append("LOTFRONTAGE_LOG")

    df_cont["HASLOTFRONTAGE"] = np.where(df_cont["LOTFRONTAGE"] == 0, 0, 1).astype("bool")
    add_cols.append("HASLOTFRONTAGE")
    bool_cols = list(set(bool_cols).union(set(add_cols)))


    cont_cols = sym_final
    cont_cols = list(set(cont_cols).union(set(mod_skew_final)))
    cont_cols = list(set(cont_cols).union(set(high_skew_final)))

    df_cont = _outlier_treatment(df=df_cont, cols=cont_cols, var_type="cont")

    return df_cont

def _disc_cols_treatment(df_cont: pd.DataFrame, disc_cols: list) -> pd.DataFrame:
    df_disc = df_cont.copy()
    df_disc_zero = _zero_pct(df_disc, disc_cols.columns)
    df_disc_zero = df_disc_zero[df_disc_zero['ZERO_PCT'] > 0]

    drop_cols = []
    add_cols = []
    zero_flag_map = {}


    ZERO_PCT_THRESH = 0.70

    zero_flag_map = {
        col: "HAS" + col
        for col in df_disc_zero[df_disc_zero["ZERO_PCT"] > ZERO_PCT_THRESH].index
    }

    for old_col, new_col in zero_flag_map.items():
        print(old_col, new_col)
        df_disc = df_disc.rename(columns={old_col: new_col})
        df_disc[new_col] = np.where(df_disc[new_col] == 0, 0, 1).astype("bool")
        drop_cols.append(old_col)
        add_cols.append(new_col)


    disc_cols = list(set(disc_cols) - set(drop_cols))
    bool_cols = list(set(bool_cols).union(set(add_cols)))

    df_disc = _outlier_treatment(df=df_disc, cols=disc_cols, var_type="disc")

    return df_disc


