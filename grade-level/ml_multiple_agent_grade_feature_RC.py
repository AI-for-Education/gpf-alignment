#%%
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
from fdllm.sysutils import register_models
HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
TABLEPATH = HERE / "grade-specs/appendix_b_tables"
DATADIR = HERE.parent / "data/stimuli"
CRIDIR = HERE / "grade-specs/appendix_b_tables"
CACHDIR = HERE / "grade-feature-RC-caches"
register_models(HERE.parent / "custom_models.yaml")
#%%
def read_and_transform_table(grade_json:dict ) -> dict:
    tables = grade_json["tables"]
    if not tables:
        return grade_json
    
    for table_name, table_file in tables.items():
        table_file_path = TABLEPATH / table_file
        table = pd.read_csv(table_file_path)
        table_str = table.to_string(index=False).replace("\n", " ").replace("  ", " ")
        table_str = table_str.replace("  ", " ")
        grade_json["tables"][table_name] = table_str
    return grade_json

grade_jsons = list(CRIDIR.glob("*.json"))

grade_description = {}
for file in grade_jsons:
    print(file)
    grade = file.stem.split("_")[-1]
    with open(file, "r") as f:
        grade_json = json.load(f)
        print(grade_json.keys())
        grade_json = read_and_transform_table(grade_json)
    # transform the dict to a formatted string
    grade_description[grade] = json.dumps(grade_json, indent=4)
with open(CACHDIR / "grade_description.json", "w") as f:
    json.dump(grade_description, f, indent=4)

# %%

single_grade_prompt ="""
    features_of_text_by_grade::
    These descriptions define features of text materials at different grades 
    according to the Global Proficiency Framework (GPF) for reading. There are direct descriptions of the features of text materials at each grade level. 
    And also examples. Some grades have more info such as tables. 
    They should vary smoothly between the different grades
    Please provide a description of the features of text materials at each grade level. The description will be used to evaluate the reading level of text materials.
    Another AI model will use this information to evaluate the reading level of text materials.
    This is about reading standards for school children in a global context so please take that into account.
"""

# %%
model = "gpt-4o-2024-08-06"
caller_feature_summary = get_caller(model)
feature_dict = {}
for grade, grade_json in grade_description.items():
    prompt = single_grade_prompt + "Grade " + grade + "::" + grade_json
    msg = LLMMessage(Role="user", Message=prompt)
    output_text = caller.call(msg).Message
    feature_dict[grade] = output_text
    
with open(CACHDIR / "feature_dict.json", "w") as f:
    json.dump(feature_dict, f, indent=4)

# %%
import re

sample_feature_dict = {k: feature_dict[k] for k in list(feature_dict.keys())[:3]}
sample_feature_json = json.dumps(sample_feature_dict, indent=4)  # a smaller subset of `feature_dict`

guideline_prompt = """
I need guidelines to ensure the descriptions of text material features are consistent, clear, and grade-appropriate.
Each grade level should have its own guideline to ensure that descriptions reflect a gradual progression in difficulty, consistent terminology, and clarity across grades. 

This is a global reading proficiency framework for school children, and the descriptions should reflect that:
1. Maintain clarity and consistency while differentiating difficulty levels.
2. Ensure a smooth progression of difficulty across grades. So a new model can classify text materials based on these descriptions.
3. Avoid exact repetitions but keep similar structure across grades for coherence.
4. Ensure the descriptions align with a global context.
Here is a sample of these descriptions for a few grade levels:
{}

Please return the prompt so I can apply to all grade levels.
""".format(sample_feature_json)
guide = LLMMessage(Role="user", Message=guideline_prompt)
output_text = caller_feature_summary.call(guide).Message
json_text = re.search(r'```json\n({.*})\n```', output_text, re.DOTALL).group(1)

# Step 2: Load JSON string into a Python dictionary
guidelines = json.loads(json_text)
with open(CACHDIR / "guidelines.json", "w") as f:
    json.dump(guidelines, f, indent=4)
#%%
refined_feature_dict = deepcopy(feature_dict)
for grade, grade_json in grade_description.items():
    prompt = "Please refine the feature descriptions for Grade " + grade + "based on the guidelines provided. Only return the feature. Nothing else." + json.dumps(guidelines, indent=4)
    prompt += grade_json
    
    msg = LLMMessage(Role="user", Message=prompt)
    output_text = caller.call(msg).Message
    refined_feature_dict[grade] = output_text
with open(CACHDIR / "refined_feature_dict.json", "w") as f:
    json.dump(refined_feature_dict, f, indent=4)
#%% Now we need to classify the text materials based on the refined feature descriptions

