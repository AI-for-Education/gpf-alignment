#%%
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fdllm import get_caller
from fdllm.chat import ChatController
from PIL import Image
from dotenv import load_dotenv
# pymupdf
import tabula

HERE = Path(__file__).resolve().parent
GPFFILE = HERE.parent / 'GPF-Reading/data-source/GPF-Reading-Final.pdf'
load_dotenv(HERE.parent / '.env')
TABLEPATH = HERE / 'grade-specs/appendix_b_tables'

#%%
def concat_ignore_nan(series):
    return ' '.join(str(x) for x in series.dropna())

def clean_df(df):
    return df.fillna('').groupby(df.index.to_series().ffill()).agg(' '.join)

def combine_rows(df, row_heights):
    cum_rows = [0] + list(np.cumsum(row_heights))
    cdf = pd.DataFrame([
        df.iloc[start:end].agg(concat_ignore_nan)
        for start, end in zip(cum_rows[:-1], cum_rows[1:])
    ])
    return cdf

# %%
# table16
dfs = tabula.read_pdf(GPFFILE, pages='86')
cdf = clean_df(dfs[0].set_index("Feature"))
cdf.to_csv(TABLEPATH / 'Table16.csv')
# %%
# table 17
dfs = tabula.read_pdf(GPFFILE, pages='88')
cdf = clean_df(dfs[0].set_index("Feature"))
cdf.to_csv(TABLEPATH / 'Table17.csv')
# %%
# table 18
dfs = tabula.read_pdf(GPFFILE, pages='88')
cdf = combine_rows(dfs[1], [3, 3]).set_index("Text type")
cdf.to_csv(TABLEPATH / 'Table18.csv')
# %%
# table 19
dfs = tabula.read_pdf(GPFFILE, pages='97')
df = dfs[0].drop('Scope',axis=1).rename(columns={"Unnamed: 0": "Scope"})
cdf = clean_df(df.set_index("Feature"))
cdf.to_csv(TABLEPATH / 'Table19.csv')
# %%
# table 20
dfs = tabula.read_pdf(GPFFILE, pages='97')
cdf = combine_rows(dfs[1], [3,4,2,2]).set_index("Text type")
cdf.to_csv(TABLEPATH / 'Table20.csv')
#%%
# tabula can't extract table 21 and table 22 so I did these by hand


