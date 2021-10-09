import sys
from civetqc.subject import Subject

if __name__ == "__main__":
    test_subj = Subject(sys.argv[1], sys.argv[2])
    print(test_subj)
