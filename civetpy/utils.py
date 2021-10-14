import pandas as pd


def import_data(filename: str, filter_by: str) -> pd.DataFrame:
    colnames = list(pd.read_csv(filename, dtype=str, nrows=0))
    include_vars = ["eid"] + [x for x in colnames if filter_by in x]
    return pd.read_csv(filename, dtype=str, usecols=include_vars)


def get_matching_ids(df: pd.DataFrame, colnames: list, s: str) -> list:
    df = df.astype(str)
    list_ids = []
    for i in colnames:
        for j in range(len(df[i])):
            value = df[i][j]
            subj_id = df["eid"][j]
            if s in value and subj_id not in list_ids:
                list_ids.append(subj_id)
        print(len(list_ids))
    return list_ids


def write_txt(my_list: list, filename: str) -> None:
    file_object = open(filename, 'w')
    for i in my_list:
        file_object.write(i + '\n')
    file_object.close()


if __name__ == "__main__":
    df = import_data("current.csv", "41202")
    df.to_csv("schizo_dx.csv", index=-False)
    list_ids = get_matching_ids(df, df.columns, "F2")
    write_txt(list_ids, "sz_patients.txt")