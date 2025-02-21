import pandas as pd
import requests
import yaml


def get_and_transform_data():
    # get raw data
    data_path = (
        "https://datahub.hymarc.org/dataset/"
        "ad580d95-e7e2-4ef4-a7f6-3b2f91a96eba/resource/"
        "4ef1c494-366e-43a3-bed4-a3985de5c374/download/hydstormatdb-reversible_hydrides.csv"
    )
    fn_data_original = "data_original.txt"
    data = requests.get(data_path)
    with open(fn_data_original, "wb") as f:
        f.write(data.content)

    # create dataframe
    df = pd.read_csv(fn_data_original, sep=",")

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "material_type",
        "material_name",
        "chemical_formula",
        "keywords",
        "synthesis_method",
        "synthesis_conditions",
        "precursors",
        "activation",
        "principal_investigator",
        "entry_date",
        "institution",
        "reversible_capacity",
        "h_weight_density_theory",
        "h_weight_density_experiment",
        "h_weight_density_reference",
        "h_volume_density_theory",
        "h_volume_density_experiment",
        "h_volume_density_reference",
        "temperature_onset_release",
        "temperature_full_release",
        "temperature_release_reference",
    ]

    # clean data
    remove_columns = [
        "keywords",
        "activation",
        "principal_investigator",
        "institution",
        "reversible_capacity",
        "h_volume_density_theory",
        "h_volume_density_experiment",
        "h_volume_density_reference",
        "temperature_release_reference",
        "h_volume_density_reference",
        "entry_date",
        "precursors",
    ]
    df = df.drop(remove_columns, axis=1)

    df["synthesis_information"] = (
        df["synthesis_method"] + ": " + df["synthesis_conditions"]
    )
    df = df.drop(["synthesis_method", "synthesis_conditions"], axis=1)

    string_columns = list(df.select_dtypes(include=["object"]).columns)
    df[string_columns] = df[string_columns].apply(lambda x: x.str.strip())

    fn_data_csv = "data_clean.csv"
    # dropna    
    df = df.dropna()
    df.to_csv(fn_data_csv, index=False)


if __name__ == "__main__":
    get_and_transform_data()
