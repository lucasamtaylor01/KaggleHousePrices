"""Data cleaning functions, converted from noteboks/02_data_cleaning.ipynb."""

from functools import reduce
from typing import Callable

import numpy as np
import pandas as pd

MEASURE_VARIABLES = [
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


def uppercase_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.upper()
    columns_str = df.select_dtypes(include='string')

    for col in columns_str.columns:
        df[col] = df[col].str.upper()

    return df


def treat_garage_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['HASGARAGE'] = np.where(df['GARAGEYRBLT'].isnull(), 0, 1).astype('bool')
    df = df.drop(columns='GARAGEYRBLT')

    df['GARAGEQUAL'] = df['GARAGETYPE'].fillna('NA')
    df['GARAGEFINISH'] = df['GARAGETYPE'].fillna('NA')
    df['GARAGETYPE'] = df['GARAGETYPE'].fillna('NA')
    df['GARAGECOND'] = df['GARAGETYPE'].fillna('NA')

    return df


def treat_basement_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bsmt_columns = ['BSMTFINTYPE2', 'BSMTEXPOSURE', 'BSMTCOND', 'BSMTQUAL', 'BSMTFINTYPE1']
    df_bsmt = df[bsmt_columns].copy()

    df['BSMTCOND'] = df['BSMTCOND'].fillna('NA')
    df['BSMTQUAL'] = df['BSMTQUAL'].fillna('NA')
    df['BSMTFINTYPE1'] = df['BSMTFINTYPE1'].fillna('NA')

    mask = (df_bsmt['BSMTCOND'].notnull()) & (
        (df_bsmt['BSMTEXPOSURE'].isnull()) | (df_bsmt['BSMTFINTYPE2'].isnull())
    )
    df = df[~mask].reset_index(drop=True)

    df['BSMTFINTYPE2'] = df['BSMTCOND'].fillna('NA')
    df['BSMTEXPOSURE'] = df['BSMTQUAL'].fillna('NA')

    return df


def treat_pool_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_pool = df[['POOLQC', 'POOLAREA']].copy()
    mask_inconsistent_pool = (
        (df_pool['POOLQC'].isnull() & (df_pool['POOLAREA'] > 0))
        | (df_pool['POOLAREA'] < 0)
    )
    df_pool_inconsistent = df_pool[mask_inconsistent_pool]

    if not df_pool_inconsistent.empty:
        print(df_pool_inconsistent)
    else:
        print('There are not inconsistent values')

    df['HASPOOL'] = np.where(df['POOLQC'].isnull(), 0, 1).astype('bool')
    df = df.drop(columns=['POOLQC', 'POOLAREA'])

    return df


def treat_fireplace_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_fireplace = df[['FIREPLACES', 'FIREPLACEQU']].copy()
    fireplace_mask = (df_fireplace['FIREPLACEQU'].isnull()) & (df_fireplace['FIREPLACES'] > 0)
    df_fireplace = df_fireplace[fireplace_mask]

    if not df_fireplace.empty:
        print(df_fireplace)
    else:
        print('There are not inconsistent values')

    df['FIREPLACEQU'] = df['FIREPLACEQU'].fillna('NA')

    return df


def treat_general_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=['ELECTRICAL', 'MASVNRAREA'])
    df['MISCFEATURE'] = df['MISCFEATURE'].fillna('NA')
    df['ALLEY'] = df['MISCFEATURE'].fillna('NA')
    df['FENCE'] = df['MISCFEATURE'].fillna('NA')
    df['MASVNRTYPE'] = df['MASVNRTYPE'].fillna('None')
    df['LOTFRONTAGE'] = df['LOTFRONTAGE'].fillna(0)

    return df


def cast_measure_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[MEASURE_VARIABLES] = df[MEASURE_VARIABLES].astype('float64')

    return df


CLEANING_STEPS: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
    uppercase_strings,
    treat_garage_nulls,
    treat_basement_nulls,
    treat_pool_nulls,
    treat_fireplace_nulls,
    treat_general_nulls,
    cast_measure_variables,
]


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    return reduce(lambda df, step: step(df), CLEANING_STEPS, df_raw)
