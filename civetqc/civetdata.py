import pandas as pd


class CivetData:

    def __init__(self, path_usrqc: str, path_civout: str) -> None:

        # Import user ratings and coerce all non-numeric values to NA
        self.usrqc = self.read_csv(path_usrqc, required_cols=["Full_ID", "QC_Rating"])
        self.usrqc["QC_Rating"] = pd.to_numeric(self.usrqc['QC_Rating'], errors='coerce')

        # All values must therefore be NA or 0, 1, or 2
        for i in self.usrqc['QC_Rating']:
            if i not in range(3) and not pd.isna(i):
                raise ValueError("All non-missing values in 'QC_Rating' must be between 0 and 2")
        
        # Import civet output
        self.civout = self.read_csv(path_civout)

        # Combine dataframes
        
        # Write csv file for documentation

    def __str__(self) -> str:
        return "\n".join((
            f"User QC: N={len(self.usrqc['Full_ID'])}",
            f"Civet Output: N={len(self.civout['ID'])}"
        ))
    
    def print_usrqc(self) -> None:
        self.print_df(self.usrqc)

    def print_civout(self) -> None:
        self.print_df(self.civout)

    @staticmethod
    def read_csv(path_csv: str, required_cols: list = []) -> pd.DataFrame:
        df = pd.read_csv(path_csv)
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Required field '{col}' not found in file {path_csv}")
        return df

    @staticmethod
    def print_df(df: pd.DataFrame) -> None:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)