# %%
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

from . import PROJECT_ROOT

TABLEPATH = PROJECT_ROOT / "grade-level/grade-specs/appendix_b_tables"
MBSSE_DIR = PROJECT_ROOT / "data/mbsse"
MBSSE_JSON = MBSSE_DIR / "mbsseKP_files_lessonplans_parsed_cleaned.json"


def load_mbsse_extras():
    # load extras for GPF grades: primary 1-6, JSS 1-3
    with open(MBSSE_JSON, "r") as f:
        mbsse = json.load(f)

    extras = []
    for file in mbsse:
        # only want GPF grades 1-9 so can skip SSS
        level = file["file_meta"]["Level"]
        subject = file["file_meta"]["Subject"]
        if level == "sss":
            continue
        if subject == "mathematics":
            continue

        for plan in file["plans"]:
            if plan["extra"]:
                for extra in plan["extra"]:
                    if extra["markdown"]:
                        if "Part" in file["file_meta"]:
                            file_meta = {
                                "part": file["file_meta"]["Part"],
                            }
                        else:
                            file_meta = {
                                "year": file["file_meta"]["Year"],
                                "term": file["file_meta"]["Term"],
                            }
                        file_meta["level"] = level
                        file_meta["subject"] = subject

                        if level == "primary":
                            file_meta["gpf_grade"] = file_meta["year"]
                        elif level == "jss":
                            file_meta["gpf_grade"] = file_meta["year"] + 6

                        extras.append(
                            {
                                **extra,
                                **file_meta,
                                "class_level": plan["Class/Level"],
                                "theme": plan["Theme"],
                                "lesson_title": plan["Lesson Title"],
                                "lesson_number": plan["Lesson Number"],
                                "filename": file["filename"],
                            }
                        )

    # add an arbitrary ID number based on load order
    for i, x in enumerate(extras):
        x["id"] = i
    return extras


# %%
# investigate all extras (loaded all not excluding SSS)
# 4019 extras items
# 1465 unique "Headings"
# SSS 4 has Part in (1,2) instead of Year/Term
# year 1,2,3,4,5,6
# print(set(x["year"] for x in extras if x["level"]=="primary"))
# {1, 2, 3, 4, 5, 6}

# print(set(x["year"] for x in extras if x["level"]=="jss"))
# {1, 2, 3}

# print(set(x["year"] for x in extras if x["level"]=="sss" and "year" in x))
# {1, 2, 3}

# print(set(x["class_level"] for x in extras if x["level"]=="sss" and "part" in
# x)) print(set(x["part"] for x in extras if x["level"]=="sss" and "part" in x))
# {1, 2}
