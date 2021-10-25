import pandas as pd
    

def print_df(df: pd.DataFrame) -> None:
    """ print all rows and columns in pandas dataframe """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def check_unique(s: pd.Series) -> bool:
    """ return whether all values in series are unique """
    return len(s.unique()) == len(s)
