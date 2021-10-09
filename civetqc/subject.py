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