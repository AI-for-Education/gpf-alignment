# %%
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fdllm import get_caller
from fdllm.extensions import general_query
from fdllm.llmtypes import LLMMessage
from fdllm.chat import ChatController
from fdllm.sysutils import register_models
from tqdm import tqdm
from joblib import Parallel, delayed
import yaml
import json
import sys

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
register_models(HERE.parent / "custom_models.yaml")
TABLEPATH = HERE / "grade-specs/appendix_b_tables"
DATADIR = HERE.parent / "data/stimuli"
sys.path.append(str(HERE.parent / "GPF-Reading"))
from GPF.item_generation import DataLoader, Item, Question

# %%
# load appendix b grade description tables
#
feature_table_nos_by_grade = {2: 16, 3: 17, 6: 19, 9: 21}
feature_tables_by_grade = {
    k: pd.read_csv(TABLEPATH / f"Table{v}.csv").set_index("Feature")
    for k, v in feature_table_nos_by_grade.items()
}
grades = list(feature_table_nos_by_grade.keys())
features = feature_tables_by_grade[2].index.unique().to_list()

#%%
# build a csv of feautures
# feature_df = pd.DataFrame(features, columns=["Features"]).set_index("Features")
concat_list = []
for g in grades:
    concat_list.append(
        feature_tables_by_grade[g]
        .loc[:, ["Scope", "Elaboration"]]
        .rename(columns={"Scope": f"G{g} Scope", "Elaboration": f"G{g} Elaboration"}),
    )
feature_df = pd.concat(concat_list,axis=1)
with open('gpf_grade_features.csv','w') as file:
    feature_df.transpose().to_csv(file,header=True)
    
    
#%%
# format into JSON grade by grade
# heirarchical or arrays with properties?
feature_descriptors = {}
for f in features:
    grade_json = {}
    for g in grades:
        grade_json.update(
            {
                f"grade-{g}::This describes the defining properties of texts at grade {g} in terms of {f}"
                "": feature_tables_by_grade[g]
                .loc[f, ["Scope", "Elaboration"]]
                .to_dict()
            }
        )
    feature_descriptors.update(
        {f"{f}::This describes the {f} of texts at different grades" "": grade_json}
    )

features_json = {
    "features_of_text_by_grade::"
    "These descriptions define features of text materials at different grades according to the GPF"
    "": feature_descriptors
}

# %%
# build a csv of feautures
# feature_df = pd.DataFrame(features, columns=["Features"]).set_index("Features")
concat_list = []
for g in grades:
    concat_list.append(
        feature_tables_by_grade[g]
        .loc[:, ["Scope", "Elaboration"]]
        .rename(columns={"Scope": f"G{g} Scope", "Elaboration": f"G{g} Elaboration"}),
    )
feature_df = pd.concat(concat_list,axis=1)
with open('gpf_grade_features.csv','w') as file:
    feature_df.transpose().to_csv(file,header=True)

#%% load amys edits
INDIR = HERE / "grade-specs/appendix_b_interpolate/gpt-4o-amy"

for f in features:
    df = pd.read_csv(INDIR / f"{f}.csv")
    suggested_col = next((col for col in df.columns if col.startswith("Suggested")), None)
    if suggested_col:
        df["Elaboration"] = df.apply(
            lambda row: row[suggested_col] if pd.notna(row[suggested_col]) else row["Elaboration"],
            axis=1
        )
        df.drop(columns=[suggested_col], inplace=True)
    df.to_csv(INDIR / f"{f}.csv", index=False)
# %%
