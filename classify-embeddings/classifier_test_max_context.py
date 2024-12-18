# %%
#! %matplotlib qt
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from gpf.item_generation import (
    get_unique_skills,
    get_all_items_flat,
    get_all_items_nfer_flat,
)
from gpf_alignment.classify import (
    gen_target_X,
    gen_Xy,
    classifier_test,
    targeted_folds_test,
    classifier_results,
    get_gpf_contexts,
)

HERE = Path(__file__).resolve().parent
training_data_path = HERE / "training_data"
example_data_path = training_data_path / "example_data"

###########################################


# %%
# load GPF and NFER items

all_items_flat = get_all_items_flat()

# only test on NFER items that do not share stimuli with GPF
all_items_unique_nfer_flat = get_all_items_nfer_flat(
    gpf_items=all_items_flat, only_unique=True
)

# for combo use all NFER items
all_items_nfer_flat = get_all_items_nfer_flat(
    gpf_items=all_items_flat, only_unique=False
)

# %%
# number of embeddings dimensions to use
nc = 512
skill_codes = get_unique_skills()
use_stimlen = True
use_qlen = True

### uncomment this if no embeddings in cache
# get_all_embedding()


X_context, codes_context = get_gpf_contexts(skill_codes=skill_codes, nc=nc)

# permute all the constructs part with each item
X, Ys, stimhash_list, qhash_list = gen_Xy(
    all_items_flat,
    nc=nc,
    skill_codes=skill_codes,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

X_unq_nfer, Ys_unq_nfer, stimhash_unq_nfer_list, qhash_unq_nfer_list = gen_Xy(
    all_items_unique_nfer_flat,
    suffix="nfer",
    nc=nc,
    skill_codes=skill_codes,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

X_nfer, Ys_nfer, stimhash_nfer_list, qhash_nfer_list = gen_Xy(
    all_items_nfer_flat,
    suffix="nfer",
    nc=nc,
    skill_codes=skill_codes,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

X_combo, Ys_combo, stimhash_combo_list, qhash_combo_list = gen_Xy(
    all_items_flat + all_items_unique_nfer_flat,
    suffix=["", "nfer"],
    nc=nc,
    skill_codes=skill_codes,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

# get the list of targeted skill code, and corresponding item
(
    X_items_sti,
    stimhash_list_sti,
    qhash_list_sti,
    skill_code_target_list_sti,
    year_target_list_sti,
) = gen_target_X(
    all_items_flat,
    nc=nc,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

(
    X_items_nfer_sti,
    stimhash_nfer_list_sti,
    qhash_nfer_list_sti,
    skill_code_target_nfer_list_sti,
    year_target_nfer_list_sti,
) = gen_target_X(
    all_items_nfer_flat,
    suffix="nfer",
    nc=nc,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

(
    X_items_unq_nfer_sti,
    stimhash_unq_nfer_list_sti,
    qhash_unq_nfer_list_sti,
    skill_code_target_unq_nfer_list_sti,
    year_target_unq_nfer_list_sti,
) = gen_target_X(
    all_items_unique_nfer_flat,
    suffix="nfer",
    nc=nc,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

(
    X_items_combo_sti,
    stimhash_combo_list_sti,
    qhash_combo_list_sti,
    skill_code_target_combo_list_sti,
    year_target_combo_list_sti,
) = gen_target_X(
    all_items_flat + all_items_unique_nfer_flat,
    suffix=["", "nfer"],
    nc=nc,
    use_stimlen=use_stimlen,
    use_qlen=use_qlen,
)

# %%
### train / test on GPF items, never mixing stimuli between train and test

model = MLPClassifier(
    max_iter=1000,
    random_state=930523,
    solver="lbfgs",
    hidden_layer_sizes=(256,),
)
train_res = targeted_folds_test(
    X,
    Ys,
    X_context,
    X_items_sti,
    skill_code_target_list_sti,
    year_target_list_sti,
    [stimhash_list],
    [stimhash_list_sti],
    codes_context=codes_context,
    verbose=1,
    model=model,
)
train_res.to_csv(training_data_path / "res" / "leave_one_out_test_context.csv")

classifier_results(
    train_res,
    perm=True,
    metric=f1_score,
    average="macro",
    figure_path=training_data_path,
)

# %%
### train / test on combined GPF and NFER items, never mixing stimuli between train and test

model = MLPClassifier(
    max_iter=1000,
    random_state=930523,
    solver="lbfgs",
    hidden_layer_sizes=(256,),
)
train_res = targeted_folds_test(
    X_combo,
    Ys_combo,
    X_context,
    X_items_combo_sti,
    skill_code_target_combo_list_sti,
    year_target_combo_list_sti,
    [stimhash_combo_list],
    [stimhash_combo_list_sti],
    codes_context=codes_context,
    model=model,
    verbose=1,
)
# train_res.to_csv(training_data_path / "res" / "leave_one_out_test_context.csv")

classifier_results(
    train_res,
    perm=True,
    figure_path=training_data_path,
    metric=f1_score,
    average="macro",
)

# %%
### train on GPF items / test on unique NFER items (NFER items that don't contain GPF stimuli)

model = MLPClassifier(
    max_iter=1000,
    random_state=930523,
    solver="lbfgs",
    hidden_layer_sizes=(256,),
)
train_res = classifier_test(
    X,
    Ys,
    X_context,
    np.arange(X.shape[0]),
    np.arange(len(X_items_unq_nfer_sti)),
    X_items_unq_nfer_sti,
    skill_code_target_unq_nfer_list_sti,
    year_target_unq_nfer_list_sti,
    codes_context=codes_context,
    model=model,
)
# train_res.to_csv(training_data_path / "res" / "leave_one_out_test_context.csv")

classifier_results(
    pd.DataFrame(train_res),
    perm=True,
    metric=f1_score,
    average="macro",
    figure_path=training_data_path,
)

# %%
### train on unique NFER items (NFER items that don't contain GPF stimuli) / test on GPF items

train_res = classifier_test(
    X_unq_nfer,
    Ys_nfer,
    X_context,
    np.arange(X_unq_nfer.shape[0]),
    np.arange(len(X_items_sti)),
    X_items_sti,
    skill_code_target_list_sti,
    year_target_list_sti,
    codes_context=codes_context,
    # model=model,
)
# train_res.to_csv(training_data_path / "res" / "leave_one_out_test_context.csv")

classifier_results(
    pd.DataFrame(train_res),
    perm=True,
    metric=f1_score,
    average="macro",
    figure_path=training_data_path,
)


# %%
# train_res_q = leave_one_out_test(X, Ys, [qhash_list], [qhash_list_sti])
# train_res_q.to_csv(training_data_path / "res" / "leave_one_out_test_context_Q.csv")

# classifier_results(train_res_q)

# # %%
# train_res_both = leave_one_out_test(
#     X, Ys, [stimhash_list, qhash_list], [stimhash_list_sti, qhash_list_sti]
# )
# train_res_both.to_csv(
#     training_data_path / "res" / "leave_one_out_test_context_both.csv"
# )

# classifier_results(train_res_both)

# # %%
# # %%
# train_res = leave_one_out_test_perm(
#     X, Ys, [stimhash_list], [stimhash_list_sti], nperms=5
# )
# # train_res.to_csv(training_data_path / "res" / "leave_one_out_test_context.csv")

# # leave_one_out_results(train_res, perm=True)
