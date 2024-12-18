import re
from itertools import product
from pathlib import Path

from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from .item_generation import (
    get_code_structure,
)
from .utils import hash_stimulus, hash_question
from .embeddings import (
    load_embeddings_items,
    load_embeddings_context,
)

HERE = Path(__file__).resolve().parent

restrs = [r"([RCD]).*", r"([RCD]\d).*", r"([RCD]\d\.\d).*", r"([RCD]\d\.\d\.\d)"]
reqs = [re.compile(restr) for restr in restrs]

# year_map = {
#     2: "2-3",
#     3: "2-3",
#     4: "4-6",
#     5: "4-6",
#     6: "4-6",
#     7: "7-9",
#     8: "7-9",
#     9: "7-9",
# }

year_map = {i: i for i in range(2, 10)}


###########################################


def _gen_X(
    stimulus_embedding,
    question_embedding,
    answer_embedding,
    stimulus,
    question,
    use_stimlen,
    use_qlen,
):
    X = [
        *stimulus_embedding,
        *question_embedding,
        *answer_embedding,
    ]
    if use_stimlen:
        X.extend(
            [
                len(stimulus.text),
                len(stimulus.text.split()),
            ]
        )
    if use_qlen:
        X.extend(
            [
                len(question.text),
                len(question.text.split()),
            ]
        )
    return X


def gen_Xy(
    items_flat,
    skill_codes,
    suffix="",
    nc=128,
    use_stimlen=True,
    use_qlen=True,
    verbose=0,
):
    (
        domain_embeddings,
        construct_embeddings,
        subconstruct_embeddings,
        skill_embeddings,
    ) = load_embeddings_context(nc=nc)

    stimulus_embeddings, question_embeddings, answer_embeddings = load_embeddings_items(
        suffix=suffix, nc=nc
    )

    X_domains = []
    X_constructs = []
    X_subconstructs = []
    X_skills = []
    X_items = []
    Ys = []
    # use multiple output here -
    stimulus_list = []
    qhash_list = []
    stimhash_list = []
    question_list = []
    item_context_list = []
    target_context_list = []
    for code, year, stimulus, question in items_flat:
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
            if verbose:
                print(skill_code)
                print(skill_code_target)
                print(out_put_Y)

            X_domains.append(domain_embedding)
            X_constructs.append(construct_embedding)
            X_subconstructs.append(subconstruct_embedding)
            X_skills.append(skill_embedding)
            X_items.append(
                _gen_X(
                    stimulus_embedding,
                    question_embedding,
                    answer_embedding,
                    stimulus,
                    question,
                    use_stimlen,
                    use_qlen,
                )
            )

            Ys.append(out_put_Y)

            stimulus_list.append(stimulus.text)
            question_list.append(question.text)
            item_context_list.append(skill_code)
            target_context_list.append(skill_code_target)
            stimhash_list.append(stimhash)
            qhash_list.append(qhash)

    X_domains = np.array(X_domains)
    X_constructs = np.array(X_constructs)
    X_subconstructs = np.array(X_subconstructs)
    X_skills = np.array(X_skills)
    X_items = np.array(X_items)
    Ys = np.array(Ys)

    X = np.concatenate(
        [X_domains, X_constructs, X_subconstructs, X_skills, X_items], axis=1
    )

    return X, Ys, stimhash_list, qhash_list


###########################################


def gen_target_X(
    items_flat,
    suffix="",
    nc=128,
    use_stimlen=True,
    use_qlen=True,
):
    stimulus_embeddings, question_embeddings, answer_embeddings = load_embeddings_items(
        suffix=suffix, nc=nc
    )

    X_items_sti = []
    stimhash_list_sti = []
    qhash_list_sti = []
    skill_code_target_list_sti = []
    year_target_list_sti = []
    for code, year, stimulus, question in items_flat:
        stimhash = hash_stimulus(stimulus)
        qhash = hash_question(question)
        stimulus_embedding = stimulus_embeddings[stimhash]
        question_embedding = question_embeddings[qhash]
        answer_embedding = answer_embeddings[qhash]
        skill_code_target = code
        stimhash_list_sti.append(stimhash)
        qhash_list_sti.append(qhash)
        X_items_sti.append(
            _gen_X(
                stimulus_embedding,
                question_embedding,
                answer_embedding,
                stimulus,
                question,
                use_stimlen,
                use_qlen,
            )
        )
        skill_code_target_list_sti.append(skill_code_target)  #
        year_target_list_sti.append(year)

    return (
        X_items_sti,
        stimhash_list_sti,
        qhash_list_sti,
        skill_code_target_list_sti,
        year_target_list_sti,
    )


##################################


def get_gpf_contexts(skill_codes, nc=128, verbose=0):
    #### load embddings
    (
        domain_embeddings,
        construct_embeddings,
        subconstruct_embeddings,
        skill_embeddings,
    ) = load_embeddings_context(nc=nc)

    # get the list of the skill code, and corresponding item
    X_domains_test = []
    X_constructs_test = []
    X_subconstructs_test = []
    X_skills_test = []
    codes_context = []
    for skill_code in skill_codes:
        # the skill_codes is the list of all unique skill-code.
        domain_code, construct_code, subconstruct_code, _ = get_code_structure(
            skill_code
        )
        domain_embedding = domain_embeddings[domain_code]
        construct_embedding = construct_embeddings[construct_code]
        subconstruct_embedding = subconstruct_embeddings[subconstruct_code]
        skill_embedding = skill_embeddings[skill_code]
        codes_context.append(skill_code)
        if verbose:
            print(skill_code)

        X_domains_test.append(domain_embedding)
        X_constructs_test.append(construct_embedding)
        X_subconstructs_test.append(subconstruct_embedding)
        X_skills_test.append(skill_embedding)
    X_domains_test = np.array(X_domains_test)
    X_constructs_test = np.array(X_constructs_test)
    X_subconstructs_test = np.array(X_subconstructs_test)
    X_skills_test = np.array(X_skills_test)

    X_context = np.concatenate(
        [X_domains_test, X_constructs_test, X_subconstructs_test, X_skills_test], axis=1
    )

    return X_context, codes_context


###########################################################################
###########################################################################


def predict_function_for_item(X_item, X_context, model, codes_context, scaler=None):
    code_parts = [
        np.array(codel)
        for codel in zip(
            *[[req.match(code).groups()[0] for req in reqs] for code in codes_context]
        )
    ]
    code_part_levels = [
        {val: codel == val for val in sorted(set(codel))} for codel in code_parts
    ]
    X_item = np.array(X_item)
    X_item = np.repeat(X_item[np.newaxis, :], X_context.shape[0], axis=0)
    X = np.concatenate([X_context, X_item], axis=1)
    if scaler is not None:
        X = scaler.transform(X)
    Y_pred = model.predict_log_proba(X)
    # return the maximum index for the axis=0
    level_pred = []
    high_level_filt = np.ones(Y_pred.shape[0], dtype=bool)
    for i, cpl in enumerate(code_part_levels):
        pred_dict = {}
        for val, idx in cpl.items():
            filt = idx & high_level_filt
            if filt.sum():
                pred_dict[val] = np.exp(Y_pred[filt, i]).mean()
        max_level = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
        level_pred.append(max_level)
        high_level_filt = high_level_filt & cpl[max_level]
    year_pred = np.exp(Y_pred[:, len(code_part_levels) :]).max(axis=0)
    level_pred.append(year_map[np.argmax(year_pred) + 2])
    return level_pred


#######
def targeted_folds_test(
    X,
    Ys,
    X_context,
    X_items_sti,
    skill_code_target_list_sti,
    year_target_list_sti,
    hash_lists,
    hash_lists_sti,
    codes_context,
    model=None,
    verbose=0,
):
    hash_lists_unique = [sorted(set(hash_list)) for hash_list in hash_lists]

    train_res = []
    for hashes in tqdm(
        product(*hash_lists_unique),
        total=np.prod([len(hlu) for hlu in hash_lists_unique]),
    ):
        if verbose:
            print(hashes)
        idx = [
            i
            for i, x in enumerate(zip(*hash_lists))
            if any(x_ == hash_ for x_, hash_ in zip(x, hashes))
        ]
        if not idx:
            continue

        idx_remain = np.array([i for i in range(len(hash_lists[0])) if i not in idx])

        idx_sti = [
            i
            for i, x in enumerate(zip(*hash_lists_sti))
            if any(x_ == hash_ for x_, hash_ in zip(x, hashes))
        ]

        train_res_ = classifier_test(
            X,
            Ys,
            X_context,
            idx_train=idx_remain,
            idx_sti=idx_sti,
            X_items_sti=X_items_sti,
            skill_code_target_list_sti=skill_code_target_list_sti,
            year_target_list_sti=year_target_list_sti,
            codes_context=codes_context,
            model=model,
            verbose=verbose,
        )
        for d in train_res_:
            d["hash"] = "_".join(hashes)
        train_res.extend(train_res_)

    return pd.DataFrame(train_res)


###############


def classifier_test(
    X,
    Ys,
    X_context,
    idx_train,
    idx_sti,
    X_items_sti,
    skill_code_target_list_sti,
    year_target_list_sti,
    codes_context,
    model=None,
    verbose=0,
):

    X_train = X[idx_train]
    Y_train = Ys[idx_train]

    scaler = StandardScaler().fit(X_train)

    if model is None:
        model = MLPClassifier(max_iter=1000, random_state=930523, solver="lbfgs")
    with np.errstate(divide="ignore"):
        model.fit(scaler.transform(X_train), Y_train)

    item_sti = [X_items_sti[i] for i in idx_sti]
    skill_code_target = [skill_code_target_list_sti[i] for i in idx_sti]
    year_target = [year_target_list_sti[i] for i in idx_sti]

    train_res = []
    for i in range(len(item_sti)):
        with np.errstate(divide="ignore"):
            (
                domain_pred,
                construct_pred,
                subconstruct_pred,
                skill_pred,
                year_pred,
            ) = predict_function_for_item(
                item_sti[i],
                X_context,
                model,
                codes_context=codes_context,
                scaler=scaler,
            )
        Y_true = skill_code_target[i]
        year_true = year_map[year_target[i]]
        domain_true, construct_true, subconstruct_true, skill_true = get_code_structure(
            Y_true
        )

        if verbose:
            print(construct_true, construct_pred)
            print(year_true, year_pred)

        this_row = {
            "item_idx": i,
            "model": model,
            "domain_true": domain_true,
            "construct_true": construct_true,
            "subconstruct_true": subconstruct_true,
            "skill_true": skill_true,
            "year_true": year_true,
            "domain_pred": domain_pred,
            "construct_pred": construct_pred,
            "subconstruct_pred": subconstruct_pred,
            "skill_pred": skill_pred,
            "year_pred": year_pred,
        }
        train_res.append(this_row)

    return train_res


##########


def classifier_results(
    res,
    doplot=False,
    perm=False,
    seed=863356,
    metric=f1_score,
    figure_path=HERE,
    **metric_kwargs,
):
    def eval(true_vals, pred_vals, doplot=False):
        if context_item not in ["year"]:
            if doplot:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            # confusion matrix
            out = metric(true_vals, pred_vals, **metric_kwargs)
            if doplot:
                sns.heatmap(
                    pd.crosstab(true_vals, pred_vals), annot=True, fmt="d", ax=ax
                )
                ax.set_title(f"{context_item} f1 score: {out}")
                fig.savefig(
                    figure_path
                    / "res"
                    / f"confusion_matrix_{context_item}_test_context.png"
                )
        else:
            out = np.mean(np.abs(pred_vals - true_vals))
        return out

    nperm = 1000
    rng = np.random.default_rng(seed)
    context_list = ["domain", "construct", "subconstruct", "skill", "year"]
    for context_item in context_list:
        pred_vals = res[context_item + "_pred"]
        true_vals = res[context_item + "_true"]
        out = eval(true_vals, pred_vals, doplot=doplot)
        print(out)
        if perm:
            perm_out = []
            for _ in range(nperm):
                pred_vals_perm = rng.permuted(pred_vals)
                perm_out.append(eval(true_vals, pred_vals_perm))
            print(np.percentile(perm_out, [2.5, 50, 97.5]))
