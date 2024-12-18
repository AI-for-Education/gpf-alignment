# %%
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict
from copy import deepcopy

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fdllm import get_caller
from fdllm.chat import ChatController
from fdllm.extensions import general_query
from fdllm.llmtypes import LLMMessage

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
TABLEPATH = HERE / "grade-specs/appendix_b_tables"
DATADIR = HERE.parent / "data/stimuli"
OUTDIR = HERE / "grade-specs/appendix_b_interpolate"
OUTDIR.mkdir(parents=True, exist_ok=True)

# %%
# load appendix b grade description tables
#
feature_table_nos_by_grade: Dict[str, pd.DataFrame] = {2: 16, 3: 17, 6: 19, 9: 21}
feature_tables_by_grade = {
    k: pd.read_csv(TABLEPATH / f"Table{v}.csv").set_index("Feature")
    for k, v in feature_table_nos_by_grade.items()
}
grades = list(feature_table_nos_by_grade.keys())
features = feature_tables_by_grade[2].index.unique().to_list()

# %%
feature_tables_by_feature: Dict[str, pd.DataFrame] = {}
for feature in features:
    df_data = defaultdict(list)
    for grade in range(2, 10):
        df_data["grade"].append(grade)
        df_data["feature"].append(feature)
        if grade in grades:
            df_data["scope"].append(feature_tables_by_grade[grade].Scope[feature])
            df_data["elaboration"].append(
                feature_tables_by_grade[grade].Elaboration[feature]
            )
            df_data["contextualization"].append(
                feature_tables_by_grade[grade].Contextualization[feature]
            )
        else:
            for key in ["scope", "elaboration", "contextualization"]:
                df_data[key].append("")

    feature_tables_by_feature[feature] = pd.DataFrame(df_data).fillna("")
# %%
for feature in features:
    feature_tables_by_feature[feature].to_csv(OUTDIR / "raw"/ f"{feature}.csv")

# %%
feature_descriptors = {}
for f in features:
    grade_json = {}
    for g in grades:
        grade_json.update(
            {
                f"Grade {g}"
                "": feature_tables_by_grade[g]
                .loc[f, ["Scope", "Elaboration"]]
                .to_dict()
            }
        )
    feature_descriptors[f] = grade_json

features_json_in = {
    "features_of_text_by_grade::"
    "These descriptions define features of text materials at different grades "
    " according to the Global Proficiency Framework (GPF) for reading."
    "They should vary smoothly between the different grades"
    "No feature alone defines a grade definitively, the grade of a text is evaluated by a holistic consideration of these different features."
    "Please complete Scope and Elaboration entries for grades where they are missing."
    "Try to avoid using exactly the same wording for Elaboration across two grades, each should be slightly different."
    "Scope can be repeated if appropriate."
    "This is about reading standards for school children in a global context so please take that into account."
    "Please make sure you respect the sequential difficulty of the grades."
    # "For example, grade 8 descriptors should represent easier texts than grade 9"
    # "Consider each new grade description in terms of the closest give both above and below. "
    # "For example, if Grade 9 Scope is 'Moderate to substantial' then Grade 8 scope should not be 'Substantial' as this is a stronger scope than at grade 9"
    "": feature_descriptors
}

# %%
# incldue all the grades in json out

if False:
    features_out = {}
    for f in features:
        grade_json = {}
        for g in range(2, 10):
            if g not in grades:
                grade_json.update({f"Grade {g}": {"Scope": "", "Elaboration": ""}})
        feature_values = grade_json | feature_descriptors[f]
        features_out[f] = {k: feature_values[k] for k in sorted(feature_values)}

    features_json_out = {"features_of_text_by_grade": features_out}

# %%

features_out = {}
for f in features:
    # load robins manual Scopes
    df = pd.read_csv(OUTDIR / "robin-manual-scope" / f"{f}.csv")
    grade_json = {}
    for g in range(2, 10):
        if g not in grades:
            grade_json.update(
                {
                    f"Grade {g}": {
                        "Scope": df.loc[df.grade == g, "scope"].squeeze(),
                        "Elaboration": "",
                    }
                }
            )
    features_out[f] = grade_json

features_json_out = {"features_of_text_by_grade": features_out}

# %%
jsonin = features_json_in
jsonout = features_json_out
# %%
model = "claude-3-5-sonnet-20241022"
# model = "gpt-4o"
# model = "gpt-4o-mini"
# model = "o1-mini"
caller = get_caller(model)
# caller = get_caller("gpt-4o")

out = general_query(
    jsonin,
    jsonout,
    caller=caller,
    role="user",
    # response_format={"type": "json_object"},
    # temperature=1,
)
print(json.dumps(out, indent=2, ensure_ascii=False))

# %%
(OUTDIR / model).mkdir(exist_ok=True)
full_descriptions = deepcopy(feature_descriptors)
for f in features:
    existing = {
        k: (v | {"Synthesised": False}) for k, v in feature_descriptors[f].items()
    }
    synthesised = {
        k: (v | {"Synthesised": True})
        for k, v in out["features_of_text_by_grade"][f].items()
    }
    full_descriptions[f] = existing | synthesised

for f in features:
    pd.DataFrame(full_descriptions[f]).transpose().sort_index().to_csv(
        OUTDIR / model / f"{f}.csv"
    )

# %%
