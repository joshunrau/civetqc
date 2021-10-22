import pandas as pd


def import_data(filename: str, contains: str) -> pd.DataFrame:
    """ import csv file including all column names containing a string """
    colnames = list(pd.read_csv(filename, dtype=str, nrows=0))
    include_vars = ["eid"] + [x for x in colnames if contains in x]
    return pd.read_csv(filename, dtype=str, usecols=include_vars)


def elements_in_str(s: str, l: list) -> bool:
    """ returns whether any element in 'l' is a substring of 's' """
    for element in l:
        if element in s:
            return True
    return False


def get_matching_ids(df: pd.DataFrame, colnames: list, identifier: str, contains: list) -> list:
    """ returns a list of ids with one or more values in any of 'colnames' containing any string in the list 'contains' """
    df = df.astype(str)
    list_ids = []
    for col in colnames:
        for row_index in range(len(df[col])):
            value, subj_id = df[col][row_index], df[identifier][row_index]
            if elements_in_str(value, contains) and subj_id not in list_ids:
                list_ids.append(subj_id)
        print(f"Number of IDs currently in list: {len(list_ids)}")
    return list_ids


def write_txt(l: list, filename: str) -> None:
    """ create file 'filename' with each element in list 'l' as a newline """
    file_object = open(filename, 'w')
    for i in l:
        file_object.write(i + '\n')
    file_object.close()


if __name__ == "__main__":
    df = import_data("current.csv", "41202")
    list_ids = get_matching_ids(df, df.columns, "eid", ["F2"])
