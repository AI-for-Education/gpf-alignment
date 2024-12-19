import json

import yaml
from fuzzywuzzy import process
from gpf.item_generation import get_all_items_flat
from gpf.gpf_item import Question, Item

from . import DATADIR

NFER_PATH = DATADIR / "nfer"
rising_items_file = NFER_PATH / "rising_nfer_aligned_questions.json"
rising_stimuli_file = NFER_PATH / "Reading_questions_stimuli.yaml"


def get_all_items_nfer_flat(
    domains=["C", "R", "D"], gpf_items=None, only_unique=False, debug=False
):
    if gpf_items is None:
        gpf_items = get_all_items_flat()

    gpf_texts = sorted(set([it[2].text for it in gpf_items]))

    with open(rising_items_file) as f:
        rising_questions = json.load(f)

    with open(rising_stimuli_file) as f:
        rising_stimuli = yaml.safe_load(f)

    # flatten the list of list of items
    all_items_flat = []
    all_stimulus_flat = []
    for qdict in rising_questions:
        question = Question.model_validate(qdict)
        title = question.itemtitle
        stimdicts = [sd for sd in rising_stimuli if sd["title"] == title]
        if len(stimdicts) != 1:
            raise
        stimdict = stimdicts[0]
        if stimdict.get("in_gpf", False):
            if only_unique:
                continue
            else:
                nearest_text, score = process.extractOne(stimdict["text"], gpf_texts)
                if score >= 90:
                    stimulus = [
                        it[2] for it in gpf_items if it[2].text == nearest_text
                    ][0]
                else:
                    if debug:
                        print(stimdict["text"])
                        print(nearest_text)
                        print(score)
                    stimulus = Item(title=title, text=stimdict["text"])
        else:
            stimulus = Item(title=title, text=stimdict["text"])
        code = question.code
        year = stimdict["question_level"]
        stimval = (code, year, stimulus)
        if stimval not in all_stimulus_flat:
            all_stimulus_flat.append((code, year, stimulus))
        all_items_flat.append((code, year, stimulus, question))
    return all_items_flat
