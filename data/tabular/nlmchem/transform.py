import os
import urllib.request
import zipfile

import pandas as pd
import yaml


def get_and_transform_data():
    url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/NLM-Chem-corpus.zip"
    local_zip_path = "NLM-Chem-corpus.zip"

    # Download the ZIP file
    urllib.request.urlretrieve(url, local_zip_path)

    # Open the ZIP file and extract the TSV file
    with zipfile.ZipFile(local_zip_path, "r") as z:
        with z.open("FINAL_v1/abbreviations.tsv") as f:
            df = pd.read_csv(f, sep="\t", header=None)

    # Set column names
    df.columns = ["MeSH_Identifier", "Abbreviation", "Full_Form"]

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Check duplicates
    assert not df.duplicated().sum(), "Found duplicate rows in the dataframe"

    # Save to CSV
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    print("Processed data saved to", fn_data_csv)

if __name__ == "__main__":
    get_and_transform_data()
