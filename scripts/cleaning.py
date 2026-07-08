import pandas as pd
import numpy as np


def _string_treatment(df_raw: pd.DataFrame) -> pd.DataFrame:
    """This function uppercases the DataFrame's column names and string values.
    It should be used JUST on cleaning().

    Args:
        df_raw (pd.DataFrame): original DataFrame

    Returns:
        pd.DataFrame: Dataframe with column names and string values uppercased
    """

    df_str_treat = df_raw.copy()
    df_str_treat.columns = df_str_treat.columns.str.upper()

    columns_str = df_str_treat.select_dtypes(include="string")

    for col in columns_str.columns:
        df_str_treat[col] = df_str_treat[col].str.upper()

    return df_str_treat


def _null_treatment(df_str_treat: pd.DataFrame, df_type: int) -> pd.DataFrame:
    """This function treats null values of DataFrame, filling or dropping them depending
    on the column, and also drops rows with inconsistent basement data.
    On the test DataFrame (df_type == 1) no rows may be dropped, since every
    row needs a prediction, so the rows that would be dropped on train are
    filled instead.
    It should be used JUST on cleaning().

    Args:
        df_str_treat (pd.DataFrame): Dataframe with string treated
        df_type (int): 0 for train, 1 for test

    Returns:
        pd.DataFrame: Dataframe with null values treated
    """

    df_null_treat = df_str_treat.copy()

    # GARAGE TREATMENT
    df_null_treat["HASGARAGE"] = np.where(
        df_null_treat["GARAGEYRBLT"].isnull(), 0, 1
    ).astype("bool")
    df_null_treat = df_null_treat.drop(columns="GARAGEYRBLT")

    df_null_treat["GARAGEQUAL"] = df_null_treat["GARAGEQUAL"].fillna("NA")
    df_null_treat["GARAGEFINISH"] = df_null_treat["GARAGEFINISH"].fillna("NA")
    df_null_treat["GARAGETYPE"] = df_null_treat["GARAGETYPE"].fillna("NA")
    df_null_treat["GARAGECOND"] = df_null_treat["GARAGECOND"].fillna("NA")

    # BASEMENT TREATMENT
    df_null_treat["BSMTCOND"] = df_null_treat["BSMTCOND"].fillna("NA")
    df_null_treat["BSMTQUAL"] = df_null_treat["BSMTQUAL"].fillna("NA")
    df_null_treat["BSMTFINTYPE1"] = df_null_treat["BSMTFINTYPE1"].fillna("NA")

    if df_type == 0:
        bsmt_columns = [
            "BSMTFINTYPE2",
            "BSMTEXPOSURE",
            "BSMTCOND",
            "BSMTQUAL",
            "BSMTFINTYPE1",
        ]
        df_null_treat_bsmt = df_null_treat[bsmt_columns].copy()

        mask = (df_null_treat_bsmt["BSMTCOND"].notnull()) & (
            (df_null_treat_bsmt["BSMTEXPOSURE"].isnull())
            | (df_null_treat_bsmt["BSMTFINTYPE2"].isnull())
        )
        filter = ~mask
        df_null_treat = df_null_treat[filter].reset_index(drop=True)

    df_null_treat["BSMTFINTYPE2"] = df_null_treat["BSMTFINTYPE2"].fillna("NA")
    df_null_treat["BSMTEXPOSURE"] = df_null_treat["BSMTEXPOSURE"].fillna("NA")

    # POOL TREATMENT

    df_null_treat["HASPOOL"] = np.where(df_null_treat["POOLQC"].isnull(), 0, 1).astype(
        "bool"
    )
    df_null_treat = df_null_treat.drop(columns=["POOLQC", "POOLAREA"])

    # FIREPLACE TREATMENT
    df_null_treat["FIREPLACEQU"] = df_null_treat["FIREPLACEQU"].fillna("NA")

    # OTHERS
    if df_type == 0:
        df_null_treat = df_null_treat.dropna(subset=["ELECTRICAL", "MASVNRAREA"])
    else:
        df_null_treat["ELECTRICAL"] = df_null_treat["ELECTRICAL"].fillna(
            df_null_treat["ELECTRICAL"].mode()[0]
        )
        df_null_treat["MASVNRAREA"] = df_null_treat["MASVNRAREA"].fillna(0)
    df_null_treat["MISCFEATURE"] = df_null_treat["MISCFEATURE"].fillna("NA")
    df_null_treat["ALLEY"] = df_null_treat["ALLEY"].fillna("NA")
    df_null_treat["FENCE"] = df_null_treat["FENCE"].fillna("NA")
    df_null_treat["MASVNRTYPE"] = df_null_treat["MASVNRTYPE"].fillna("None")
    df_null_treat["LOTFRONTAGE"] = df_null_treat["LOTFRONTAGE"].fillna(0)

    return df_null_treat


def _types_treatment(df_null_treat: pd.DataFrame, df_type: int) -> pd.DataFrame:
    """Corrects types of DataFrame. It should be used JUST on cleaning().

    Args:
        df_null_treat (pd.DataFrame): Dataframe with null values treated

    Returns:
        pd.DataFrame: Dataframe with types treated
    """

    df_type_treat = df_null_treat.copy()
    measure_varibles = [
        "LOTAREA",
        "BSMTFINSF1",
        "BSMTFINSF2",
        "BSMTUNFSF",
        "TOTALBSMTSF",
        "1STFLRSF",
        "2NDFLRSF",
        "LOWQUALFINSF",
        "GRLIVAREA",
        "GARAGEAREA",
        "WOODDECKSF",
        "OPENPORCHSF",
        "ENCLOSEDPORCH",
        "3SSNPORCH",
        "SCREENPORCH",
    ]

    df_type_treat[measure_varibles] = df_type_treat[measure_varibles].astype("float64")

    if df_type == 0:
        df_type_treat['SALEPRICE'] = df_type_treat['SALEPRICE'].astype("float64")
    
    return df_type_treat


def cleaning(df_raw: pd.DataFrame, df_type: int) -> pd.DataFrame:
    try:
        if df_type == 0:
            print("[!] STARTING DATA CLEANING PROCESS OF TRAIN DATAFRAME ...\n")
        else:
            print("[!] STARTING DATA CLEANING PROCESS OF TEST DATAFRAME ...\n")

        df_str_treat = _string_treatment(df_raw)
        print("[1/3] STRINGS TREATED")

        df_null_treat = _null_treatment(df_str_treat, df_type)
        print("[2/3] NULL VALUES TREATED")

        df_clean = _types_treatment(df_null_treat, df_type)
        print("[3/3] TYPES TREATED\n")

        if df_type == 0:
            print("[!] DATA CLEANING PROCESS OF TRAIN DATAFRAME  WAS SUCCESSFUL...\n")
        else:
            print("[!] DATA CLEANING PROCESS OF TEST DATAFRAME  WAS SUCCESSFUL...\n")

        print(75*'-', '\n')
        return df_clean
    
    except:
        print("[!] AN ERROR WAS OCURRED ON DATA CLEANING PROCESS. \n")