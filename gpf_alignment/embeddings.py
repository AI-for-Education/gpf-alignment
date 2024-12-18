from pathlib import Path
import json

from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
import numpy as np

from .item_generation import (
    DataLoader,
    get_code_structure,
    get_unique_grades,
    get_unique_skills,
    get_all_items_flat,
    get_all_items_nfer_flat,
)
from .utils import hash_stimulus, hash_question

HERE = Path(__file__).resolve().parent
training_data_path = HERE.parent / "GPF-Reading/training_data"
example_data_path = training_data_path / "example_data"
example_data_path.mkdir(exist_ok=True, parents=True)


# set api_key = "" if needed
client = OpenAI()


# text_emdedding_3-large and shrunk it to 128 / 256


def generate_item_text(item):
    title = item.title  # remove
    genre = item.genre  # remove
    text = item.text
    out = f"title: {title}\ngenre: {genre}\ntext: {text}"
    return out


def get_embedding(
    text,
    model="text-embedding-3-small",
):
    if not isinstance(text, list):
        text = [text]

    response = client.embeddings.create(input=text, model=model)
    return [item.embedding for item in response.data]


def get_item_embedding(items_flat, suffix=""):
    stimulus_text = {}
    question_text = {}
    answer_text = {}
    for code, year, stimulus, question in items_flat:
        stimhash = hash_stimulus(stimulus)
        if stimhash not in stimulus_text:
            stimulus_text[stimhash] = generate_item_text(stimulus)
        qhash = hash_question(question)
        if qhash not in question_text:
            question_text[qhash] = question.text
            answer_text[qhash] = question.answer
    embedding_types = [
        ("stimulus", stimulus_text),
        ("question", question_text),
        ("answer", answer_text),
    ]
    for embedding_type, texts in embedding_types:
        if suffix:
            prefix = f"{embedding_type}_{suffix}"
        else:
            prefix = embedding_type
        outfile = example_data_path / f"{prefix}_embeddings_small.json"
        if not outfile.exists():
            embeddings = get_embedding(list(texts.values()))
            out = dict(zip(texts.keys(), embeddings))
            with open(outfile, "w") as f:
                json.dump(out, f)


# normalize function from https://platform.openai.com/docs/guides/embeddings/use-cases
def _normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def _process_embeddings_list(embs_list, nc=128):
    embs_list = [d.copy() for d in embs_list]
    for d in embs_list:
        for k, v in d.items():
            v_cut = v[:nc]
            v_normalized = _normalize_l2(v_cut)
            d[k] = v_normalized

    return embs_list


def load_embeddings_context(nc=128):
    # load the embeddings
    with open(example_data_path / "domain_embeddings_small.json") as f:
        domain_embeddings = json.load(f)
    with open(example_data_path / "subconstruct_embeddings_small.json") as f:
        subconstruct_embeddings = json.load(f)
    with open(example_data_path / "construct_embeddings_small.json") as f:
        construct_embeddings = json.load(f)
    with open(example_data_path / "skill_embeddings_small.json") as f:
        skill_embeddings = json.load(f)

    return _process_embeddings_list(
        [
            domain_embeddings,
            construct_embeddings,
            subconstruct_embeddings,
            skill_embeddings,
        ],
        nc=nc,
    )


def load_embeddings_items(suffix="", nc=128):
    if not isinstance(suffix, list):
        suffix = [suffix]
    item_embedding_types = [
        [f"{et}_{suff}" if suff else et for et in ["stimulus", "question", "answer"]]
        for suff in suffix
    ]

    item_embeddings = []
    for iet in item_embedding_types:
        ie = []
        for et in iet:
            with open(example_data_path / f"{et}_embeddings_small.json") as f:
                ie.append(json.load(f))
        item_embeddings.append(_process_embeddings_list(ie, nc=nc))

    out = [{} for _ in range(len(item_embeddings[0]))]
    for ie in item_embeddings:
        for i, embs in enumerate(ie):
            out[i] = {**out[i], **embs}
    
    return out
    


def load_embeddings(suffix="", nc=128):
    return [
        *load_embeddings_context(nc=nc),
        *load_embeddings_items(suffix=suffix, nc=nc),
    ]


def get_all_embedding():
    all_items_flat = get_all_items_flat()
    all_items_nfer_flat = get_all_items_nfer_flat()
    skill_codes = get_unique_skills()
    dataLoader = DataLoader()

    # generate embeddings
    domain_codes = []
    construct_codes = []
    subconstruct_codes = []
    for skill_code in skill_codes:
        domain_code, construct_code, subconstruct_code, _ = get_code_structure(
            skill_code
        )
        domain_codes.append(domain_code)
        construct_codes.append(construct_code)
        subconstruct_codes.append(subconstruct_code)

    all_grades = get_unique_grades()

    # domain embeddings
    embedding_types = [
        ("domain", domain_codes, dataLoader.get_domain, "description"),
        ("construct", construct_codes, dataLoader.get_construct, "description"),
        (
            "subconstruct",
            subconstruct_codes,
            dataLoader.get_subconstruct,
            "description",
        ),
        (
            "skill",
            skill_codes,
            lambda s: dataLoader.get_skill(skill_code=s, grades=all_grades),
            "description",
        ),
    ]

    for embedding_type, codes, get_func, attr in embedding_types:
        outfile = example_data_path / f"{embedding_type}_embeddings_small.json"
        if not outfile.exists():
            codes_unique = list(set(codes))
            texts = [getattr(get_func(code), attr) for code in codes_unique]
            embeddings = get_embedding(texts)
            out = dict(zip(codes_unique, embeddings))
            with open(outfile, "w") as f:
                json.dump(out, f)

    get_item_embedding(all_items_flat)
    get_item_embedding(all_items_nfer_flat, suffix="nfer")
