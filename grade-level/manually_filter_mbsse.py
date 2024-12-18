# %%
import json
from gpf_alignment.mbsse import load_mbsse_extras, MBSSE_DIR

extras = load_mbsse_extras()

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__ # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if __name__ == "__main__" and not is_notebook():
    keep = []
    fname = input("filename to save:\n").lower()
    for i, ex in enumerate(extras):
        print(f"Item: {i} from lesson {ex['lesson_number']}")
        print(f"GPF GRADE {ex['gpf_grade']}")
        print("--------------\n")
        print(ex['heading'])
        print(ex['markdown'])
        print("\n\n")
        choice = input("Use this one? y/n\n").lower()
        if choice == "y":
            keep.append(True)
        else:
            keep.append(False)
    with open(fname, "w") as f:
        json.dump(keep, f)

#%%

# RI manual run 20/11/2024
file = MBSSE_DIR / "is_reading_stim_ri_20241120.json"
with open(file) as f:
    keep = json.load(f)
# %%
# keep[461] = True
# keep[669] = True
# with open(file,"w") as f:
#     json.dump(keep, f)
# %%
