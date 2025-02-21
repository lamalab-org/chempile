import pandas as pd
import yaml
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list


def get_and_transform_data():
    # get raw data
    name = "herg_central"
    label_names = retrieve_label_name_list("herg_central")

    # select datasubset
    ln = label_names[0]  # herg_central_at_1uM
    print(ln)

    # get raw data
    splits = Tox(name=name, label_name=ln).get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_valid, df_test], axis=0)

    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=False)
    del df, df_train, df_valid, df_test

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = ["compound_id", "SMILES", "herg_central_at_1uM", "split"]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

if __name__ == "__main__":
    get_and_transform_data()
