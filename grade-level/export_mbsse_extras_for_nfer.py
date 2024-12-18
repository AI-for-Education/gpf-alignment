# %%
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from mbsse import MBSSE_DIR, load_mbsse_extras
import subprocess

HERE = Path(__file__).resolve().parent
# %%
# write out grade extras for NFER
extras = load_mbsse_extras()
with open(MBSSE_DIR / "is_reading_stim_ri_20241120.json") as f:
    keep = json.load(f)

OUT_DIR = HERE.parent / "data/nfer/mbsse_extas"
EXCEL_OUT = OUT_DIR / "excel"
EXCEL_OUT.mkdir(exist_ok=True, parents=True)
DOC_OUT = OUT_DIR / "doc"
DOC_OUT.mkdir(exist_ok=True, parents=True)
for grade in range(1, 10):
    extras_grade = [
        x for i, x in enumerate(extras) if x["gpf_grade"] == grade and keep[i]
    ]

    # Excel table from pandas df
    data = defaultdict(list)
    cols = ["id", "heading", "markdown", "gpf_grade"]
    for col in cols:
        data[col] = [x[col] for x in extras_grade]
    df = pd.DataFrame(data)
    df.to_excel(EXCEL_OUT / f"grade_{grade}.xlsx", index=False)

    # Format a document
    md_file = DOC_OUT / f"grade_{grade}.md"
    with open(md_file,"w") as f:
        for x in extras_grade:
            f.write(f"**Item: {x['id']}, Grade: {x['gpf_grade']}**\n\n")
            f.write(f"Heading: {x['heading']}\n\n")
            f.write(x["markdown"])
            f.write("\n\n\n")
    docx_file = md_file.with_suffix('.docx')
    subprocess.run(['pandoc', str(md_file), '-o', str(docx_file)])

# %%
