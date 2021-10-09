import pandas as pd


class Subject:

    def __init__(self, prefix, subj_id) -> None:
        self.prefix = str(prefix)
        self.subj_id = str(subj_id)
        self.angles_png = f"{prefix}_{subj_id}_angles.png" # quality control image for mesh distortion angle between white and gray surfaces
        self.atlas_png = f"{prefix}_{subj_id}_atlas.png" # quality control image for surface registration and lobar segmentation
        self.clasp_png = f"{prefix}_{subj_id}_clasp.png" # quality control image for surface extraction
        self.converg_png = f"{prefix}_{subj_id}_converg.png" # quality control image for white/gray surface convergence
        self.gradient_png = f"{prefix}_{subj_id}_gradient.png" # quality control image for t1-gradient at position of calibrated white surface
        self.laplace_png = f"{prefix}_{subj_id}_laplace.png" # quality control image for gray surface expansion
        self.surfsurf_png = f"{prefix}_{subj_id}_surfsurf.png" # quality control image for surface-surface intersections
        self.verify_png = f"{prefix}_{subj_id}_verify.png" # quality control image for registration and classification
        self.classify_txt = f"{prefix}_{subj_id}_classify_qc.txt" # classified tissue percentages
        self.surface_txt = f"{prefix}_{subj_id}_surface_qc.txt" # error for white and gray surfaces
        self.civet_txt = f"{prefix}_{subj_id}_civet_qc.txt" # list of values of processing variables for populating the QC table
    
    def __str__(self) -> str:
        return f"{self.prefix} {self.subj_id}"
    
    @classmethod
    def import_csv(cls, path_csv) -> None:
        """ get data from csv file containing known QC ratings """

        cls.df = pd.read_csv(path_csv)
        for col in ["Full_ID", "QC_Rating"]:
            if col not in cls.df.columns:
                raise KeyError(f"Required field '{col}' not found in file {path_csv}")
        
        # All non-numeric values will be coerced to NA
        cls.df["QC_Rating"] = pd.to_numeric(cls.df['QC_Rating'], errors='coerce')

        # All values must therefore be NA or 0, 1, or 2
        for i in cls.df['QC_Rating']:
            if i not in range(3) and not pd.isna(i):
                raise ValueError("All non-missing values in 'QC_Rating' must be between 0 and 2")

    @staticmethod
    def print_df(df) -> None:
        """ used to print entire data frame for testing in jupyter notebook """
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
