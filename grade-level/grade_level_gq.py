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
from gpf.item_generation import DataLoader, Item, Question

# %%
# load interpolated appendix b grade description tables
#
feature_table_nos_by_grade = {2: 16, 3: 17, 6: 19, 9: 21}
feature_tables_by_grade = {
    k: pd.read_csv(TABLEPATH / f"Table{v}.csv").set_index("Feature")
    for k, v in feature_table_nos_by_grade.items()
}
grades = list(feature_table_nos_by_grade.keys())
features = feature_tables_by_grade[2].index.unique().to_list()

# load amys edits
feature_descriptors = {}
for f in features:
    feature_json = {}
    df = pd.read_csv(HERE / "grade-specs/appendix_b_interpolate/gpt-4o-amy" / f"{f}.csv",index_col=0)
    df["Combined"] = df["Scope"].str.strip() + " : " + df["Elaboration"].str.strip()

    feature_json.update(
        df.loc[:,'Combined'].to_dict()
    )

    feature_descriptors.update(
        {f"{f}::This describes the {f} of texts at different grades" "": feature_json}
    )

features_json = {
    "features_of_text_by_grade::"
    "These descriptions define features of text materials at different grades according to the GPF"
    "": feature_descriptors
}

# %%
# load appendix b grade stimulus type tables
#
type_table_nos_by_grade = {3: 18, 6: 20, 9: 22}
type_tables_by_grade = {
    k: pd.read_csv(TABLEPATH / f"Table{v}.csv").set_index("Text type")
    for k, v in type_table_nos_by_grade.items()
}
text_type_descriptions = {}
for g in type_tables_by_grade.keys():
    text_type_descriptions.update(
        {
            f"grade-{g}::This describes the types of text items that can be used for grade {g}"
            "": type_tables_by_grade[g].transpose().to_dict()
        }
    )
text_types_json = {
    "types_of_text_by_grade::"
    "These descriptions define the types of text items that can be used for assessments at some example grades"
    "": text_type_descriptions
}
# %%
# grade item text descriptions
grade_text_descriptions = {
    "grade-2::This describes texts used for grade 2": "At grade two, texts are so short that they are mainly simple descriptions. Texts typically have a single character engaged in a simple action, or a very brief description of a single object or event.",
    "grade-4::This describes texts used for grade 4": "Grade four texts are typically slightly longer than grade three texts and contain more detail. However, greater complexity in one factor may be balanced by less complexity in another. For example, a shorter text may contain some less familiar content, or some less common vocabulary.",
    "grade-5::This describes texts used for grade 5": "Grade five texts may be of varying lengths and are mainly narrative (stories) and informational. Some instructional texts may also be used. Simple non-continuous texts such as lists and tables are introduced at this level. There may be some non-conventional genre elements in the texts.  Narrative texts include details such as some limited character development, or a simple description of the setting. Information texts may include basic paratextual features: for example, subheadings or captions.  Vocabulary includes a wide range of familiar words describing concrete and abstract concepts as well as less familiar words where the context strongly supports the meaning. For example, a common technical or discipline-specific term may be used where the meaning can be inferred from prominent clues.",
    "grade-7::This describes texts used for grade 7": "Grade 7 texts are of varying lengths, with longer texts typically being straightforward and shorter texts a little more complex. A range of familiar text types, including narrative (stories), informational, persuasive, and instructional texts, are used at this grade level. A range of simple, non-continuous formats includes tables, diagrams, maps, and graphs.  Texts typically include several minor complexities such as unfamiliar content that is clearly explained, less common vocabulary supported in context, significant implied ideas, or a less familiar structure.",
    "grade-8::This describes texts used for grade 8": "Texts may be somewhat longer and more complex than grade seven texts. Text types that include narrative, informational, persuasive, and instruction are used at this grade level. A range of non-continuous formats includes tables, diagrams, maps, and graphs.  Texts typically include several minor complexities such as unfamiliar content that is clearly explained, less common vocabulary supported in context, significant implied ideas, or a less familiar structure.",
}
grade_descriptions_json = {
    "descriptions_of_grade_texts::"
    "These descriptions give a description of texts at different grades"
    "": grade_text_descriptions
}

# %%
# load GPF text items / stimuli
#
loader = DataLoader()
all_domains = [loader.get_domain(d) for d in ["C", "R", "D"]]
items = loader.get_items()
gpf_example_items_json = [
    json.loads(
        i.model_dump_json(include=["text", "genre", "grade_appropriate", "explanation"])
        .replace('"grade_appropriate":', '"grade":')
        .replace('"genre":', '"type":')
    )
    for i in items
]

gpf_example_text_json = {
    "example_texts::"
    "These are example texts which are graded according to the GPF, along with explanations for that grading"
    "": gpf_example_items_json
}


# %%
jsonin = {}
jsonin.update(text_types_json)
jsonin.update(grade_descriptions_json)
jsonin.update(gpf_example_text_json)
jsonin.update(features_json)

# %%
#

jsonout = {
    "text_to_classify::"
    """
    You are pedagogical expert with experience of teaching at both primary and secondary levels in an international context. 
    classify the grade-level of the provided text, based on the features described. 
    When assessing features such as the commonality of words, please keep in mind that this refers
    to exposure pupils will have had to those words in educational materials at the appropriate grade level, and not necessarily the frequency of the words in overall adult language use. 
    Please bear in mind that a particular text might appear to fit different grade levels depending on different features. 
    The features provided are a guide, but the final rating should be a holistic consideration which trades off the different aspects encapsulated in the features. 
    """
    "": {
        #         "text": """
        #             Ama moves to a new town with her family. She gets ready for her first day at a new school. Ama
        #     is scared and excited at the same time. Ama makes new friends. She also learns new things at her
        #     school. One of the things she learns about herself is that she is good at Maths, thanks to the
        #     support of her new teacher. Sometimes change is a good thing!
        # """
        #     "text": """
        # Africa is a beautiful continent with many wild animals. Elephants, lions, tigers and zebras roam the
        # savannahs.
        # Mountain gorillas live in the rainforests. The African grey parrot flies across the sky. In the rivers,
        # crocodiles and hippopotamuses live. Which of Africa's animals is most beautiful to you? Some say lions
        # are the most beautiful. Others are amazed by the elephants.
        # Some wild animals are dangerous. They attack humans and destroy crops. Some people kill wild animals
        # and sell their parts for money.
        # It is important to preserve wildlife and keep them safe. If not, Africa will lose many animals to poachers,
        # who hunt them. Animals also sometimes die from harsh natural conditions like drought. Animals keep
        # ecosystems balanced.
        # Learning about wild animals will help humans preserve them. It will also help humans and animals to live
        # together in peace. Preserving wildlife means these beautiful animals will remain for many years to come.
        #     """
        # "text": " Are you afraid of sharks? Some sharks are harmless. The dwarf lantern shark cannot hurt you.  You might think sharks are large but this one is not. It is so small you can hold it in one hand.  Another unusual thing about dwarf lantern sharks is that they glow in the dark. They live at the bottom of very deep oceans. There is no light where they live. They make their own light."
        "text": "My name is Aba. I like to play with toy cars. My sister likes to play with dolls.  After school, we do our homework. Then, it is play time!",
        "grade:: (Int)": None,
        "explanation": {
            **{f: None for f in features},
            "overall::here put an overall explanation for the grade rating, which weighs up the different features and tries to give the best rating on balance": None,
        },
        "confidence:: (1-10)": None,
    }
}

# %%
jsonout = {
    "grade_texts::"
    """
    Please generate **three** new texts of the type described, appropriate for the grade-level specified
    Be careful to generate valid JSON. Please escape any quotation marks used inside the generated texts. 
    """: [
        {
            "grade": 8,
            # "type": "Information (descriptions)",
            "type": "Narrative (Stories)",
            # "type": "Information (non-continuous text), a recipe",
            "text": None,
        }
    ]
}

# %%
# caller = get_caller("gpt-4o-2024-08-06")
caller = get_caller("claude-3-5-sonnet-20241022")
# caller = get_caller("claude-3-opus-20240229")
# caller = get_caller("gpt-4o-mini-2024-07-18")
# caller = get_caller("or-llama-3.2-3b-instruct")

out = general_query(
    jsonin,
    jsonout,
    caller=caller,
    role="user",
    # response_format={"type": "json_object"},
    temperature=0,
)
print(json.dumps(out, indent=2, ensure_ascii=False))

# %%
caller = get_caller("gpt-4o-2024-08-06")
msg = LLMMessage(Role="user", Message=json.dumps(jsonin))
print(len(caller.tokenize([msg])))

# %%
#!%%timeit
caller = get_caller("claude-3-5-sonnet-20240620")
# caller = get_caller("gpt-4o-2024-08-06")


# %%
#
#
# Cross-validate over all stimuli (rising and gpf)
#

# %%
# load rising items
#
with open(DATADIR / "Reading_questions_stimuli.yaml", "r") as file:
    stimuli_full = yaml.safe_load(file)

stimuli = [
    {"grade": s["question_level"], "title": s["title"], "text": s["text"]}
    for s in stimuli_full
]
for i, stim in enumerate(stimuli_full):
    if "gpf_explanation" in stim:
        stimuli[i]["explanation"] = stim["gpf_explanation"]
    # remove grade 1
    if stim["question_level"]==1:
        stimuli[i]["grade"]==2

gt_grade = [s["grade"] for s in stimuli]
# %%
# leave-one-out cv loop
# preserve order of examples

# model = "gpt-4o-2024-08-06"
model = "claude-3-5-sonnet-20241022"
caller = get_caller(model)

# could order of examples be a factor?
# currently increasing grade level
# randomise or reverse order?
# for stim in stimuli:
cv_res = []
cv_grade = []
for stim in tqdm(stimuli):
    cvstimuli = [
        {k: v for k, v in s.items() if k != "title"}
        for s in stimuli
        if s["title"] != stim["title"]
    ]
    cv_example_json = {
        "example_texts::"
        "These are example texts which are graded according to the GPF, along with explanations for that grading, if available"
        "": json.dumps(cvstimuli)
    }

    jsonin = {}
    jsonin.update(text_types_json)
    jsonin.update(grade_descriptions_json)
    jsonin.update(cv_example_json)
    jsonin.update(features_json)

    jsonout = {
        "text_to_classify::"
        """
        classify the grade-level of the provided text, based on the features described. 
        When assessing features such as the commonality of words, please keep in mind that this refers
        to exposure pupils will have had to those words in educational materials at the appropriate grade level, and not necessarily the frequency of the words in overall adult language use. 
        Please take care to return valid JSON (escape quotation marks if necessary). 
        """
        "": {
            "text": json.dumps(stim["text"]),
            "explanation": {
                **{f: None for f in features},
                "overall::here put an overall explanation for the grade rating, which weighs up the different features and tries to give the best rating on balance": None,
            },
            "grade:: (Int)": None,
        }
    }
    out = general_query(
        jsonin,
        jsonout,
        caller=caller,
        role="user",
        # response_format={"type": "json_object"},
        temperature=0,
    )
    cv_res.append(out)
    cv_grade.append(out["text_to_classify"]["grade"])
    # print(json.dumps(out, indent=2, ensure_ascii=False))


# %%


model = "gpt-4o-2024-08-06"
# model = "claude-3-5-sonnet-20241022"
caller = get_caller(model)

prompt_version = 3


def cvloop(stim):
    res = {}
    cvstimuli = [
        {k: v for k, v in s.items() if k != "title"}
        for s in stimuli
        if s["title"] != stim["title"]
    ]
    cv_example_json = {
        "example_texts::"
        "These are example texts which are graded according to the GPF, along with explanations for that grading, if available"
        "": json.dumps(cvstimuli)
    }

    jsonin = {}
    jsonin.update(text_types_json)
    jsonin.update(grade_descriptions_json)
    jsonin.update(cv_example_json)
    jsonin.update(features_json)

    jsonout = {
        "text_to_classify::"
        """
        You are pedagogical expert with experience of teaching at both primary and secondary levels in an international context. 
        Your task is to classify the grade-level of the provided text, based on the GPF as described in the input. 
        When assessing features such as the commonality of words, please keep in mind that this refers
        to exposure pupils will have had to those words in educational materials at the appropriate grade level, and not necessarily the frequency of the words in overall adult language use. 
        Please bear in mind that a particular text might appear to fit different grade levels depending on different features. 
        The features provided are a guide, but the final rating should be a holistic consideration which trades off the different aspects encapsulated in the features. 
        When assessing features such as the commonality of words, please keep in mind that this refers
        to exposure pupils will have had to those words in educational materials at the appropriate grade level, and not necessarily the frequency of the words in overall adult language use. 
        Please take care to return valid JSON (escape quotation marks if necessary).         """
        "": {
            "text": json.dumps(stim["text"]),
            "explanation": {
                **{f: None for f in features},
                "overall::here put an overall explanation for the grade rating, which weighs up the different features and tries to give the best rating on balance": None,
            },
            "grade:: (Int)": None,
        }
    }
    out = general_query(
        jsonin,
        jsonout,
        caller=caller,
        role="user",
        # response_format={"type": "json_object"},
        temperature=0,
    )
    res["out"] = out
    res["grade"] = out["text_to_classify"]["grade"]
    return res


cv_res_jl = Parallel(n_jobs=10, prefer="threads")(delayed(cvloop)(s) for s in stimuli)

prompt_cache = Path(f"cv_cache/prompt_v{prompt_version}.json")
if not prompt_cache.exists():
    with open(prompt_cache, "w") as file:
        json.dump({"jsonin": jsonin, "jsonout": jsonout}, file, indent=2)
with open(f"cv_cache/{model}_v{prompt_version}.json", "w") as file:
    json.dump(cv_res_jl, file, indent=2)
# %%
cv_grade = [r["grade"] for r in cv_res_jl]
x = np.array([cv_grade, gt_grade]).T
error = x[:, 1] - x[:, 0]
mse = ((error) ** 2.0).mean()
rmse = np.sqrt(mse)
print(np.c_[x, error])
cv_df = pd.DataFrame(np.c_[x, error], columns=["CV", "GT", "Error"])
print(cv_df.groupby("GT").Error.mean().to_markdown())
# %%
# gpt4o
# |   GT |     Error |
# |-----:|----------:|
# |    1 | -0.875    |
# |    2 | -0.125    |
# |    3 |  0.142857 |
# |    4 |  0.7      |
# |    5 | -0.5      |
# |    6 |  0        |
# |    7 |  1.6      |
# |    8 |  2        |
# |    9 |  0        |
#
# claude-3.5-sonnet
# |   GT |     Error |
# |-----:|----------:|
# |    1 | -1        |
# |    2 | -0.125    |
# |    3 |  0        |
# |    4 |  0.4      |
# |    5 | -1        |
# |    6 |  0        |
# |    7 |  0.4      |
# |    8 |  0.666667 |
# |    9 |  0        |


# %%
