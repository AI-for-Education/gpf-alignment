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
    results["roc_auc"] = roc_auc_score(y_true, y_pred)
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

models = [
    LogisticRegression(solver="lbfgs", max_iter=2000),
    SVC(probability=True),  # balance options
    MLPClassifier(max_iter=500),
    GaussianNB(),
    # MLPRegressor(activation="tanh", solver="lbfgs", max_iter=2000)
]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# stratified kfold - check this out.
#%%
for model in models:

    multi_target_model = MultiOutputClassifier(model)
    # multi_target_model = MultiOutputRegressor(model)
    res_df = []
    fold = 0
    for train_index, test_index in kfold.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Ys[train_index], Ys[test_index]

        # Fit the model on the training data
        multi_target_model.fit(X_train, Y_train)

        # Predict on the test data
        Y_pred = multi_target_model.predict(X_test)
        # Y_prob = multi_target_model.predict_proba(
        #     X_test
        # )  # This might need adjustment based on classifier capabilities

        # Calculate metrics for each label

        for i in range(Ys.shape[1]):
            metric_results = calculate_metrics(
                Y_test[:, i], Y_pred[:, i]  # , Y_prob[i][:, 1] if Y_prob else None
            )
            # calculate the mean for each feature and get list of the scores across features
            # add this list to the dictionary for each metric
            this_row = metric_results.copy()
            this_row["feature"] = i
            this_row["fold"] = fold
            res_df.append(this_row)

        fold += 1
    res_df = pd.DataFrame(res_df)
    res_df_mean = res_df.groupby("feature").agg("mean").reset_index()
    print("model:", model)
    print(res_df_mean)

# %%
test = np.concatenate([Y_test, Y_pred], axis=1)
print(test.tolist())
# close
# print out wrong one and comapre to the correct one
# use the correct grade
# add the grade for each context and try add the embedding for the grade info.

# 100 new data coming in,
# haramony mean

# %%
for i in range(4):
    print("total number of positive samples:", np.sum(Ys[:, i]), "out of", len(Ys))

# %%
# focus on the reason that lead to the right prediction from LogisticRegression
model = MLPClassifier()
multi_target_model = MultiOutputClassifier(model)

nfeature = Ys.shape[1]
final_result_correct = {}
final_result_wrong = {}
final_value_correct = {}
final_value_wrong = {}
for i in range(Ys.shape[1]):
    correct_index_original = []
    wrong_index_original = []
    correct_value_all = []
    wrong_value_all = []

    for train_index, test_index in kfold.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Ys[train_index], Ys[test_index]

        # Fit the model on the training data
        multi_target_model.fit(X_train, Y_train)

        # Predict on the test data
        Y_pred = multi_target_model.predict(X_test)


        correct_index = np.where(Y_test[:, i] == Y_pred[:, i])[0]
        wrong_index = np.where(Y_test[:, i] != Y_pred[:, i])[0]
        correct_value = Y_test[correct_index, i]
        wrong_value = Y_test[wrong_index, i]
        correct_value_all.append(correct_value)
        wrong_value_all.append(wrong_value)
        
        correct_index_original.append(test_index[correct_index])
        wrong_index_original.append(test_index[wrong_index])
    correct_index_original = np.concatenate(correct_index_original)
    wrong_index_original = np.concatenate(wrong_index_original)
    final_result_correct[i] = correct_index_original
    final_result_wrong[i] = wrong_index_original
    final_value_correct[i] = np.concatenate(correct_value_all)
    final_value_wrong[i] = np.concatenate(wrong_value_all)




# %%
for key, value in final_result_wrong.items():
    print(key, len(value))
    
# %%
# check for the feature 1, 2, 3
this_index = final_result_wrong[3]
stimulus_list_wrong = [stimulus_list[idx] for idx in this_index]
question_list_wrong = [question_list[idx] for idx in this_index]
item_context_list_wrong = [item_context_list[idx] for idx in this_index]
target_context_list_wrong = [target_context_list[idx] for idx in this_index]
# %%
unique_stimulus = list(set(stimulus_list_wrong))
# %%
wrong_count = {}
wrong_codepair = {}
for stim in unique_stimulus:
    wrong_count[stim] = len([i for i in stimulus_list_wrong if i == stim])
    this_index = [idx for idx, i in enumerate(stimulus_list_wrong) if i == stim]
    this_target = [target_context_list_wrong[idx] for idx in this_index]
    this_item = [item_context_list_wrong[idx] for idx in this_index]
    wrong_codepair[stim] = list(zip(this_target, this_item))
#%%
for key, value in wrong_codepair.items():
    print(key, value)
#%%
for x in unique_stimulus:
    print(x)
    print(wrong_codepair[x])
    print('-----')
#%%
unique_question = list(set(question_list_wrong))
print(len(unique_question))
for x in unique_question:
    print(x)
    print('-----')




#%%
# all the stimulus that have more than 1 wrong question
print('all the stimulus that have more than 1 wrong question')
# %%
print("The trial for the wrong prediction")
for key, value in final_value_wrong.items():
    print(key)
    print('-----')
    count_matched = np.sum(value)
    count_mismatched = len(value) - count_matched
    print('match:', count_matched)
    print('no match:', count_mismatched)
print('Mostly wronly think the matched code is not matches between code and target')

# %%
print("The trial for the correct prediction")
for key, value in final_value_correct.items():
    print(key)
    print('-----')
    count_matched = np.sum(value)
    count_mismatched = len(value) - count_matched
    print('match:', count_matched)
    print('no match:', count_mismatched)
#%% get the confusion matrix for the model
for featurei in range(4):
    print('-'*10)
    print(f'feature {featurei}')
    print('-'*10)
    print(f'there are {len(Ys)} samples in total for feature {featurei}')
    print(f'{len(Ys) - np.sum(Ys[:, featurei])} samples are mismatched')
    print(f'{np.sum(Ys[:, featurei])} samples are matched')
    corr_trial_this = final_value_correct[featurei]
    count_matched = np.sum(corr_trial_this)
    count_mismatched = len(corr_trial_this) - count_matched
    wrong_trial_this = final_value_wrong[featurei]
    count_matched_wrong = np.sum(wrong_trial_this)
    count_mismatched_wrong = len(wrong_trial_this) - count_matched_wrong

    print('------')
    print(f'for the {np.sum(Ys[:, featurei])} matched samples')
    print(f'only {count_matched} matched samples are correctly predicted')
    print(f'most of matched samples, {count_matched_wrong}, are not correctly predicted')

    print('------')
    print(f'for the {len(Ys)-np.sum(Ys[:, featurei])} mismatched samples')
    print(f'only {count_mismatched} mismatched samples are correctly predicted')
    print(f'very little of mismatched samples, {count_mismatched_wrong}, are not correctly predicted')
    print('------')



# %%
