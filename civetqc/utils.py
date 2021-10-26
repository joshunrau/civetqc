import pandas as pd


def print_df(df: pd.DataFrame) -> None:
    """ print all rows and columns in pandas dataframe """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)