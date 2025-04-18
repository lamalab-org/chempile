import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from rdkit import Chem


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True

def canonicalize_smiles(smiles: str) -> str:
    """
    Return the canonical SMILES string.
    """

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def read_dataset():
    # hf_data = pd.read_csv(
    #     "https://huggingface.co/datasets/chemNLP/RedDB/raw/main/RedDBv2.csv"
    # )
    file = hf_hub_download(
        repo_id="chemNLP/RedDB", filename="RedDBv2.csv", repo_type="dataset"
    )
    hf_data = pd.read_csv(file)
    # apply canonical smiles
    hf_data["SMILES"] = hf_data.SMILES.apply(canonicalize_smiles)
    assert hf_data.SMILES.apply(is_valid_smiles).to_list() == [True] * len(hf_data)
    assert not hf_data.duplicated().sum()

    hf_data = hf_data.dropna()
    hf_data = hf_data.drop(columns=["Unnamed: 0", "id"])

    hf_data.to_csv("data_clean.csv")
    # create meta yaml
    meta = {
        "name": "RedDB",  # unique identifier, we will also use this for directory names
        "description": f"""RedDB: a computational database that contains {len(hf_data)} molecules
from two prominent classes of organic electroactive compounds, quinones and aza-aromatics,
has been presented. RedDB incorporates miscellaneous physicochemical property information
of the compounds that can potentially be employed as battery performance descriptors.
RedDBs development steps, including:
(i) chemical library generation,
(ii) molecular property prediction based on quantum chemical calculations,
(iii) aqueous solubility prediction using machine learning,
(iv) data processing and database creation, have been described.""",
        "targets": [
            {
                "id": "solubilityAqSolPred",
                "description": "Aqueous solubility prediction using machine learning",
                "units": "logS",
                "type": "continuous",
                "names": [{"noun": "ML-predicted aqueous solubility"}],
            },
            {
                "id": "molecularSurface",
                "description": "Total surface area of a molecule",
                "units": "\\AA^2",
                "type": "continuous",
                "names": [{"noun": "molecular surface area"}],
            },
            {
                "id": "reactionFieldEnergy",
                "description": "Energy associated with the interaction during a chemical reaction",
                "units": "kT",
                "type": "continuous",
                "names": [{"noun": "chemical reaction field energy"}],
            },
            {
                "id": "solventAccessSurface",
                "description": "Surface area of a molecule accessible to a solvent",
                "units": "\\AA^2",
                "type": "continuous",
                "names": [{"noun": "solvent-accessible surface area"}],
            },
            {
                "id": "cavityEnergy",
                "description": "Energy associated with the formation of cavities in a molecular structure",
                "units": "kT",
                "type": "continuous",
                "names": [
                    {"noun": "cavity formation energy at the PBE level of theory"}
                ],
            },
            {
                "id": "gasEnergy",
                "description": "Total energy of a molecule in the gas phase",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {"noun": "gas-phase molecular energy at the PBE level of theory"}
                ],
            },
            {
                "id": "gasHomo",
                "description": "Highest Occupied Molecular Orbital (HOMO) energy of a gas-phase molecule",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {"noun": "gaseous phase HOMO energy at the PBE level of theory"}
                ],
            },
            {
                "id": "gasLumo",
                "description": "Lowest Unoccupied Molecular Orbital (LUMO) energy of a gas-phase molecule",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {"noun": "gaseous phase LUMO energy at the PBE level of theory"}
                ],
            },
            {
                "id": "solutionEnergy",
                "description": "Total energy of a molecule in a solution",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {
                        "noun": "aqueous phase molecular energy at the PBE level of theory"
                    }
                ],
            },
            {
                "id": "solutionHomo",
                "description": "Highest Occupied Molecular Orbital (HOMO) energy in a solution",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {"noun": "aqueous phase HOMO energy at the PBE level of theory"}
                ],
            },
            {
                "id": "solutionLumo",
                "description": "Lowest Unoccupied Molecular Orbital (LUMO) energy in a solution",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {"noun": "aqueous phase LUMO energy at the PBE level of theory"}
                ],
            },
            {
                "id": "nuclearRepulsionEnergy",
                "description": "Electrostatic repulsion energy between atomic nuclei in a molecule",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {"noun": "nuclear repulsion energy at the PBE level of theory"}
                ],
            },
            {
                "id": "optGasEnergy",
                "description": "Total energy of an optimized gas-phase molecule",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {
                        "noun": "optimized gas-phase molecular energy at the PBE level of theory"
                    }
                ],
            },
            {
                "id": "optGasHomo",
                "description": "Highest Occupied Molecular Orbital (HOMO) energy of an optimized gas-phase molecule",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {
                        "noun": "optimized gas-phase HOMO energy at the PBE level of theory"
                    }
                ],
            },
            {
                "id": "optGasLumo",
                "description": "Lowest Unoccupied Molecular Orbital (LUMO) energy of an optimized gas-phase molecule",
                "units": "Hartree",
                "type": "continuous",
                "names": [
                    {
                        "noun": "optimized gas-phase LUMO energy calculated at the PBE level of theory"
                    },
                    {
                        "noun": "optimized gas-phase LUMO energy calculated with DFT at the PBE level of theory"
                    },
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",
                "description": "SMILES",  # description (optional, except for "Other")
            },
            {"id": "InChI", "type": "InChI", "description": "InChI"},
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1038/s41597-022-01832-2",
                "description": "corresponding publication",
            },
            {
                "url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F3QFSQ",
                "description": "Data source",
            },
        ],
        "num_points": len(hf_data),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Elif2022,
doi = {10.1021/ci300400a},
url = {https://doi.org/10.1038/s41597-022-01832-2},
year = {2022},
volume = {9},
number = {1},
author = {Elif Sorkun and Qi Zhang and Abhishek Khetan and Murat Cihan Sorkun and
Suleyman Er},
journal = {Nature Scientific Data}""",
        ],
        "templates": [
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {solubilityAqSolPred__names__noun} of {solubilityAqSolPred#} {solubilityAqSolPred__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {molecularSurface__names__noun} of {molecularSurface#} {molecularSurface__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {reactionFieldEnergy__names__noun} of {reactionFieldEnergy#} {reactionFieldEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {solventAccessSurface__names__noun} of {solventAccessSurface#} {solventAccessSurface__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {cavityEnergy__names__noun} of {cavityEnergy#} {cavityEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {gasEnergy__names__noun} of {gasEnergy#} {gasEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {gasHomo__names__noun} of {gasHomo#} {gasHomo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {gasLumo__names__noun} of {gasLumo#} {gasLumo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {solutionEnergy__names__noun} of {solutionEnergy#} {solutionEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {solutionLumo__names__noun} of {solutionLumo#} {solutionLumo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {nuclearRepulsionEnergy__names__noun} of {nuclearRepulsionEnergy#} {nuclearRepulsionEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {optGasEnergy__names__noun} of {optGasEnergy#} {optGasEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {optGasHomo__names__noun} of {optGasHomo#} {optGasHomo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} {#representation of |!}{SMILES#} has a {optGasLumo__names__noun} of {optGasLumo#} {optGasLumo__units}."""  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {solubilityAqSolPred__names__noun} of {solubilityAqSolPred#} {solubilityAqSolPred__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {molecularSurface__names__noun} of {molecularSurface#} {molecularSurface__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {reactionFieldEnergy__names__noun} of {reactionFieldEnergy#} {reactionFieldEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {solventAccessSurface__names__noun} of {solventAccessSurface#} {solventAccessSurface__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {cavityEnergy__names__noun} of {cavityEnergy#} {cavityEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {gasEnergy__names__noun} of {gasEnergy#} {gasEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {gasHomo__names__noun} of {gasHomo#} {gasHomo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {gasLumo__names__noun} of {gasLumo#} {gasLumo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {solutionEnergy__names__noun} of {solutionEnergy#} {solutionEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {solutionLumo__names__noun} of {solutionLumo#} {solutionLumo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {nuclearRepulsionEnergy__names__noun} of {nuclearRepulsionEnergy#} {nuclearRepulsionEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {optGasEnergy__names__noun} of {optGasEnergy#} {optGasEnergy__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {optGasHomo__names__noun} of {optGasHomo#} {optGasHomo__units}.""",  # noqa: E501
            """The {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} {#representation of |!}{InChI#} has a {optGasLumo__names__noun} of {optGasLumo#} {optGasLumo__units}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} based on the {#text |!}description{# below|!}.
Description: It has an {solubilityAqSolPred__names__noun} {solubilityAqSolPred#} {solubilityAqSolPred__units} and a {cavityEnergy__names__noun} of {cavityEnergy#} {cavityEnergy__units}.
      Result: {SMILES#}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} based on the {#text |!}description{# below|!}.
Description: It has an {solubilityAqSolPred__names__noun} {solubilityAqSolPred#} {solubilityAqSolPred__units} and a {cavityEnergy__names__noun} of {cavityEnergy#} {cavityEnergy__units}.
      Result: {InChI#}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description} based on the {#text |!}description{# below|!}.
Description: It has an {solutionLumo__names__noun} {solutionLumo#} {solutionLumo__units} and a {solutionHomo__names__noun} of {solutionHomo#} {solutionHomo__units}.
Result: {SMILES#}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule|compound|chemical|molecular species|chemical compound!} with the {InChI__description} based on the {#text |!}description{# below|!}.
Description: It has an {solutionLumo__names__noun} {solutionLumo#} {solutionLumo__units} and a {solutionHomo__names__noun} of {solutionHomo#} {solutionHomo__units}.
Result: {InChI#}""",  # noqa: E501
        ],
    }

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(
        str, str_presenter
    )  # to use with safe_dum
    fn_meta = "meta.yaml"
    #with open(fn_meta, "w") as f:
        #yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    read_dataset()
