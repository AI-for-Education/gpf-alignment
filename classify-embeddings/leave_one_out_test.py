# %%
# %%
#! %matplotlib qt
from pathlib import Path
import json

from dotenv import load_dotenv

load_dotenv(override=True)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent

from gpf.item_generation import (
    DataLoader,
    Item,
    Question,
    get_code_structure,
    get_unique_grades,
    get_unique_skills,
    get_question_data,
    get_all_items_flat
)
from gpf.utils import hash_stimulus, hash_question

training_data_path = HERE / "training_data"
example_data_path = training_data_path / "example_data"


# %%


print(get_unique_skills())
dataLoader = DataLoader()
skill_codes = get_unique_skills()

all_domains = [dataLoader.get_domain(d) for d in ["C", "R", "D"]]

all_items_flat = get_all_items_flat()


# %%
# normalize function from https://platform.openai.com/docs/guides/embeddings/use-cases
def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


# load the embeddings
with open(example_data_path / "domain_embeddings_small.json") as f:
    domain_embeddings = json.load(f)
with open(example_data_path / "subconstruct_embeddings_small.json") as f:
    subconstruct_embeddings = json.load(f)
with open(example_data_path / "construct_embeddings_small.json") as f:
    construct_embeddings = json.load(f)
with open(example_data_path / "skill_embeddings_small.json") as f:
    skill_embeddings = json.load(f)
with open(example_data_path / "stimulus_embeddings_small.json") as f:
    stimulus_embeddings = json.load(f)
with open(example_data_path / "question_embeddings_small.json") as f:
    question_embeddings = json.load(f)
with open(example_data_path / "answer_embeddings_small.json") as f:
    answer_embeddings = json.load(f)

nc = 128
# use the first nc of the embedding for the trainning data
# then normalize the embedding
dict_list = [
    domain_embeddings,
    construct_embeddings,
    subconstruct_embeddings,
    skill_embeddings,
    stimulus_embeddings,
    question_embeddings,
    answer_embeddings,
]
for d in dict_list:
    for k, v in d.items():
        v_cut = v[:nc]
        v_normalized = normalize_l2(v_cut)
        d[k] = v_normalized


# %% permute all the constructs part with each item
X_domains = []
X_constructs = []
X_subconstructs = []
X_skills = []
X_items = []
Ys = []
# use multiple output here -
stimulus_list = []
qhash_list = []
stihash_list = []
question_list = []
item_context_list = []
target_context_list = []
for code, year, stimulus, question in all_items_flat:
    stimhash = hash_stimulus(stimulus)
    qhash = hash_question(question)
    stimulus_embedding = stimulus_embeddings[stimhash]
    question_embedding = question_embeddings[qhash]
    answer_embedding = answer_embeddings[qhash]
    skill_code_target = code
    domain_code_target, construct_code_target, subconstruct_code_target, _ = (
        get_code_structure(skill_code_target)
    )
    # none of the items have more than one code,
    # the code for each item is also fixed.

    # get the corresponding skill-code for the items.
    # use code to seperate it from the skill-code for the structure information
    for skill_code in skill_codes:
        # the skill_codes is the list of all unique skill-code.
        domain_code, construct_code, subconstruct_code, _ = get_code_structure(
            skill_code
        )
        domain_embedding = domain_embeddings[domain_code]
        construct_embedding = construct_embeddings[construct_code]
        subconstruct_embedding = subconstruct_embeddings[subconstruct_code]
        skill_embedding = skill_embeddings[skill_code]
        out_put_Y = [
            domain_code == domain_code_target,
            construct_code == construct_code_target,
            subconstruct_code == subconstruct_code_target,
            skill_code == skill_code_target,
            *[i == year for i in range(2, 10)],
        ]

        out_put_Y = [int(i) for i in out_put_Y]
        out_put_Y = np.array(out_put_Y)
        print(skill_code)
        print(skill_code_target)
        print(out_put_Y)

        X_domains.append(domain_embedding)
        X_constructs.append(construct_embedding)
        X_subconstructs.append(subconstruct_embedding)
        X_skills.append(skill_embedding)
        X_items.append(
            [
                *stimulus_embedding,
                *question_embedding,
                *answer_embedding,
                len(stimulus.text),
                len(stimulus.text.split()),
            ]
        )

        Ys.append(out_put_Y)

        stimulus_list.append(stimulus.text)
        question_list.append(question.text)
        item_context_list.append(skill_code)
        target_context_list.append(skill_code_target)
        stihash_list.append(stimhash)
        qhash_list.append(qhash)

X_domains = np.array(X_domains)
X_constructs = np.array(X_constructs)
X_subconstructs = np.array(X_subconstructs)
X_skills = np.array(X_skills)
X_items = np.array(X_items)
Ys = np.array(Ys)

# use train_test_split to split the data and train a multiple logistic regression model
# X = np.concatenate([X_domains, X_constructs, X_subconstructs, X_skills], axis=1)
X = np.concatenate(
    [X_domains, X_constructs, X_subconstructs, X_skills, X_items], axis=1
)
# test the accuracy of the model among 100 time of shuffling
# %%
# try different models

acc_table = {}


# model = LogisticRegression()
# cross validate, give the model, calc the score from that .
# or ask for multiple scores.
#
# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_prob=None):
    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["f1"] = f1_score(y_true, y_pred, average="macro")
    results["precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    results["recall"] = recall_score(y_true, y_pred, average="macro")
    # results["roc_auc"] = roc_auc_score(y_true, y_pred)
    # if y_prob is not None:
    #     results['roc_auc'] = roc_auc_score(y_true, y_prob)
    #     results['log_loss'] = log_loss(y_true, y_prob)
    return results


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)

# remove the other domain and used only the R domaiin
#

import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.svm import SVC  # clustering.
from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)  # try the size of the layer

model = MLPClassifier(max_iter=500)


# %%
stimhash_list_unique = list(set(stihash_list))
train_res = []
nfeature = Ys.shape[1]
# leave one out test
for stimhash in stimhash_list_unique:
    # get the index of the stimulus
    idx = [i for i, x in enumerate(stihash_list) if x == stimhash]
    idx = np.array(idx)
    idx_remain = [i for i in range(len(stihash_list)) if i not in idx]
    idx_remain = np.array(idx_remain)

    X_train = X[idx_remain]
    Y_train = Ys[idx_remain]
    X_test = X[idx]
    Y_test = Ys[idx]

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    for i in range(1, nfeature):

        metrics = calculate_metrics(Y_test[:, i], Y_pred[:, i])
        this_row = {
            "stimhash": stimhash,
            "model": model,
            "feature": i,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            # "roc_auc": metrics["roc_auc"],
        }
        train_res.append(this_row)
# %%
train_res = pd.DataFrame(train_res)
train_res
# %%
import matplotlib.pyplot as plt
import seaborn as sns

train_res_to_plot = train_res[["feature", "f1", "accuracy", "precision", "recall"]]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(x="feature", y="f1", data=train_res_to_plot, ax=ax)
ax.plot([0, 11], [0.5, 0.5], "r--")
ax.set_ylim([0, 1])
ax.set_title("F1 score")
fig.savefig(training_data_path / "res" / "f1_score_leaveoneout.png")

# %%
qhash_list_unique = list(set(qhash_list))
train_res = []
nfeature = Ys.shape[1]
# leave one out test
for qhash in qhash_list_unique:
    # get the index of the stimulus
    idx = [i for i, x in enumerate(qhash_list) if x == qhash]
    idx = np.array(idx)
    idx_remain = [i for i in range(len(qhash)) if i not in idx]
    idx_remain = np.array(idx_remain)

    X_train = X[idx_remain]
    Y_train = Ys[idx_remain]
    X_test = X[idx]
    Y_test = Ys[idx]

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    for i in range(1, nfeature):

        metrics = calculate_metrics(Y_test[:, i], Y_pred[:, i])
        this_row = {
            "stimhash": stimhash,
            "model": model,
            "feature": i,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            # "roc_auc": metrics["roc_auc"],
        }
        train_res.append(this_row)
# %%
train_res = pd.DataFrame(train_res)
train_res_to_plot = train_res[["feature", "f1", "accuracy", "precision", "recall"]]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(x="feature", y="f1", data=train_res_to_plot, ax=ax)
ax.plot([0, 11], [0.5, 0.5], "r--")
ax.set_ylim([0, 1])
ax.set_title("F1 score")
fig.savefig(training_data_path / "res" / "f1_score_leaveoneout_question.png")

# %%
# %%
# get all the unique skill code for the performance test

# %%
