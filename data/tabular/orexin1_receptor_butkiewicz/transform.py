import pandas as pd
import yaml
from tdc.single_pred import HTS


def get_and_transform_data():
    # get raw data
    label = "orexin1_receptor_butkiewicz"
    splits = HTS(name=label).get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test], axis=0)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = ["compound_id", "SMILES", "activity_orexin1", "split"]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "orexin1_receptor_butkiewicz",
        "description": """"GPCR Orexin 1 is relevant for behavioral plasticity,
the sleep-wake cycle, and gastric acid secretion.Three primary screens,
AID 485270, AID 463079, AID 434989, were performed. Validation assay
AID504701, AD492963. Counter screen 493232. More specific assay
AID504699. AID504701 and AID504699 were combined to identify 234 active
compounds excluding an overlap of 155 molecules.""",
        "targets": [
            {
                "id": "activity_orexin1",  # name of the column in a tabular dataset
                "description": "whether it is active against orexin1 receptor (1) or not (0).",
                "units": None,
                "type": "boolean",
                "names": [
                    {"noun": "orexin 1 inhibitor"},
                    {"noun": "a orexin 1 receptor antagonist"},
                    {"gerund": "inhibiting orexin 1 receptor"},
                    {"adjective": "orexin-1 inhibitory"},
                ],
                "pubchem_aids": [485270, 463079, 434989, 504701, 493232, 504699],
                "uris": ["http://purl.bioontology.org/ontology/SNOMEDCT/838464006"],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
        ],
        "benchmarks": [
            {
                "name": "TDC",
                "link": "https://tdcommons.ai/",
                "split_column": "split",
            }
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://tdcommons.ai/single_pred_tasks/hts/#butkiewicz-et-al",
                "description": "original dataset",
            },
            {
                "url": "https://doi.org/10.3390/molecules18010735",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1093/nar/gky1033",
                "description": "corresponding publication",
            },
            {
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5962024/",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Butkiewicz2013,
doi = {10.3390/molecules18010735},
url = {https://doi.org/10.3390/molecules18010735},
year = {2013},
month = jan,
publisher = {{MDPI} {AG}},
volume = {18},
number = {1},
pages = {735--756},
author = {Mariusz Butkiewicz and Edward Lowe and Ralf Mueller and
Jeffrey Mendenhall and Pedro Teixeira and C. Weaver and Jens
Meiler},
title = {Benchmarking Ligand-Based Virtual High-Throughput
Screening with the {PubChem} Database},
journal = {Molecules}}""",
            """@article{Kim2018,
doi = {10.1093/nar/gky1033},
url = {https://doi.org/10.1093/nar/gky1033},
year = {2018},
month = oct,
publisher = {Oxford University Press ({OUP})},
volume = {47},
number = {D1},
pages = {D1102--D1109},
author = {Sunghwan Kim and Jie Chen and Tiejun Cheng and Asta
Gindulyte and Jia He and Siqian He and Qingliang Li and Benjamin
A Shoemaker and Paul A Thiessen and Bo Yu and Leonid Zaslavsky
and Jian Zhang and Evan E Bolton},
title = {{PubChem} 2019 update: improved access to chemical data},
journal = {Nucleic Acids Research}}""",
            """@article{Butkiewicz2017,
doi = {},
url = {https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5962024/},
year = {2017},
publisher = {Chem Inform},
volume = {3},
number = {1},
author = {Butkiewicz, M. and Wang, Y. and Bryant, S. H. and Lowe,
E. W. and Weaver, D. C. and Meiler, J.},
title = {{H}igh-{T}hroughput {S}creening {A}ssay {D}atasets from
the {P}ub{C}hem {D}atabase}},
journal = {Chemical Science}}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}.",  # noqa: E501
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {activity_orexin1#not &NULL}{activity_orexin1__names__gerund}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule has {activity_orexin1#no &NULL}{activity_orexin1__names__noun} {#properties|characteristics|features!}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that is {activity_orexin1#not &NULL}identified as {activity_orexin1__names__adjective}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_orexin1__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional|extra!} words.
Result: {activity_orexin1#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_orexin1__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {activity_orexin1__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|figure out|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {activity_orexin1__names__adjective}?
Assistant: {activity_orexin1#No&Yes}, this molecule is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {activity_orexin1__names__adjective}?
Assistant: {activity_orexin1#No&Yes}, it is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}?
Assistant: This is a molecule that is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: This sounds {#very exciting. |very interesting. | very curious. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {activity_orexin1#not &NULL}be {activity_orexin1__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {activity_orexin1#not &NULL}be {activity_orexin1__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {activity_orexin1__names__adjective}:<EOI>{activity_orexin1#no&yes}",  # noqa: E501 for the benchmarking setup <EOI>separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_orexin1__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI>{activity_orexin1#False&True}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {activity_orexin1__names__adjective}.
Result:<EOI>{SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {activity_orexin1__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{activity_orexin1%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {activity_orexin1__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{activity_orexin1%}
Answer:<EOI>{%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%activity_orexin1%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_orexin1#not &NULL}{activity_orexin1__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional|extra!} words.
Options:
{SMILES%activity_orexin1%}
Answer:<EOI>{%multiple_choice_result}""",  # noqa: E501,
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
    get_and_transform_data()
