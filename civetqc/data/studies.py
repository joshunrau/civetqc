import os


class Subject:

    def __init__(self, study_dir, subject_id) -> None:
        self.angles = os.path.join(study_dir, f"{subject_id}_angles.png")
        self.atlas = os.path.join(study_dir, f"{subject_id}_atlas.png")
        self.civet_qc = os.path.join(study_dir, f"{subject_id}_civet_qc.txt")
        self.clasp = os.path.join(study_dir, f"{subject_id}_clasp.png")
        self.classify_qc = os.path.join(study_dir, f"{subject_id}_classify_qc.txt")
        self.converg = os.path.join(study_dir, f"{subject_id}_converg.png")
        self.gradient = os.path.join(study_dir, f"{subject_id}_gradient.png")
        self.laplace = os.path.join(study_dir, f"{subject_id}_laplace.png")
        self.surface_qc = os.path.join(study_dir, f"{subject_id}_surface_qc.txt")
        self.surfsurf = os.path.join(study_dir, f"{subject_id}_surfsurf.png")
        self.verify = os.path.join(study_dir, f"{subject_id}_verify.png")


class Study:

    data_directory = "/Users/joshua/Developer/civetqc/data"
    raw_data_directory = "/Users/joshua/Developer/civetqc/data/raw"

    def __init__(self, name) -> None:

        self.name = name
        self.directory = os.path.join(self.data_directory, self.name)
        self.raw_data_directory = os.path.join(self.directory, "raw")
        
        list_ids = []
        for filename in os.listdir(self.directory):
            subject_id = "_".join(filename.split("_")[:2])
            if subject_id not in list_ids:
                list_ids.append(subject_id)
        
        print(list_ids)

    def organize_directory(self):
        for filename in os.listdir(self.directory):
            print(filename)


class Studies:

    names = ["FEP", "INSIGHT", "LAM", "NUSDAST", "TOPSY"]
    dir_path = Study.data_directory
    
    filepaths = {}
    for n in names:
        filepaths[n] = (os.path.join(dir_path, n, f"{n}_civet_data.csv"), os.path.join(dir_path, n, f"{n}_QC.csv"))


def main():
    for study in [Study(name) for name in ["LAM"]]:
        pass
        #study.organize_directory()

if __name__ == "__main__":
    main()