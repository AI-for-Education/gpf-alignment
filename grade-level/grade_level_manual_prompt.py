# %%
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fdllm import get_caller
from fdllm.chat import ChatController
from fdllm.extensions import general_query
from fdllm.llmtypes import LLMMessage
from fdllm.sysutils import register_models
from joblib import Parallel, delayed
from hashlib import md5
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
register_models(HERE.parent / "custom_models.yaml")
TABLEPATH = HERE / "grade-specs/appendix_b_tables"
DATADIR = HERE.parent / "data/stimuli"
sys.path.append(str(HERE.parent / "GPF-Reading"))
from GPF.item_generation import DataLoader, Item, Question

# %%
#
# Manually prompting
#
# %%
grade_level_text_background = """
You are an expert teacher with a deep understanding of foundational literacy and numeracy, particularly assessment of reading skills at primary and secondary levels. 

The Global Proficiency Framework (GPF) for Reading relies heavily on "grade-level text". This paper aims to support that definition by describing a continuum of text complexity and examples of texts at designated grade levels. In this
context, the term “text” applies to written or printed artifacts, whether paper-based or digital, that comprise language arranged in sentences and paragraphs
(continuous texts) or other meaningful structures such as lists, tables, or labeled diagrams (non-continuous texts). While Grade 1 is included in the GPF, it is
not included in this description of text complexity because the Grade 1 focus is on single words, rather than longer continuous or non-continuous texts.

A Continuum of Text Complexity
MANY FACTORS
Evaluating text complexity requires complex judgments based on consideration of many factors that can make reading a text with comprehension more or
less difficult. The text itself—the length, the structure, the vocabulary, the extent of the challenge involved in interpretation—need to be considered. The
student’s context also matters, as what is familiar, whether through formal teaching or through general background knowledge, influences the extent to which
students will find it easier or harder to understand the text.
This annex provides broad guidelines about key factors that affect the complexity of a text at various grade levels. Sample texts are offered for illustration.

GRADE-APPROPRIATE
The assumption is that a grade-appropriate text is one that most students in that grade would be able to read independently and largely understand. That is,
they would understand the main ideas and important details, but may not understand every aspect of the text. (Note that in the early years of school, students’
aural comprehension will be considerably more advanced than the texts they are able to read independently.) In order for text complexity to be reflected in
assessment results, the items must address the main ideas and important details, so that student understanding of the overall text is assessed. A further
important assumption is that, in general, the complexity of the text will be reflected in the difficulty of the items; that is, simple texts will support easy items
and complex texts will have items that require students to think carefully about the meaning of the text

ON-BALANCE JUDGMENTS
As texts become more complex, the factors that affect how difficult the text is to comprehend also become more complex. This is not a uniform trajectory.
The overall complexity of a text must be an on-balance judgment, based on consideration of the interplay of all of the factors mentioned above, including the
students’ context.
The intention in this annex is to describe the key factors that affect complexity when these are relatively evenly balanced within a text. This helps to support
differentiating text complexity between grade levels, but many texts may not exhibit such even balance, especially as texts become more complex. Some
factors in a text may be easier than those suggested at a grade level and others may be harder. An on-balance judgment is required about where the text best
fits.
The intention here is also to describe and illustrate an average text that sits within a designated grade and would be considered on balance, too easy for most
students in the grade above and too hard for most students in the grade below. An average text is positioned, as much as possible, in the middle of a continuum
of text complexity for a grade. There is no hard boundary between grade levels for text complexity, and there will be many texts that are borderline and fall
into grey areas of being possibly suitable for many students in two adjacent grades. Some parts of a text may be simple and some parts more complex.
Considered judgements are required about overall complexity and the extent to which this is appropriate for most students in a given grade.

CONTINUUM AND MPLS
There are many clear differences between a grade two-level text, a grade three-level text, and a grade four-level text, making it reasonably straightforward
to describe and differentiate texts at each of these grades. However, it becomes increasingly difficult to make fine, between-grade level distinctions above
grade four. From grade five on, there is an increasing number of ways in which each of the factors that affect complexity (for example, length, familiarity of
content or vocabulary) might be made more challenging and the interplay of factors also becomes more complex. The wider range of text types that students
are expected to encounter as they become more proficient readers also makes comparisons of text complexity more challenging. It is more meaningful to
make broader distinctions. Accordingly, because the focus of the MPLs is on grades two/three, end of primary (typically grade six), and end of lower secondary
(typically grade nine), this document focuses on the factors that affect text complexity at grade two, grade three, grade six, and grade nine. Sample texts at
these levels are described in terms of the key factors affecting text complexity. Additional texts are located along the continuum—at the intermediate grades,
grades four and five, and grades seven and eight—but no descriptions of the factors affecting text complexity are provided for these grades. The intermediary
grade texts have been ranked based on on-balance judgements

MAKING COMPARISONS
Ranking through pairwise comparison of texts is strongly recommended as a strategy to support allocating a text to a grade level of complexity.

A new text can be compared with sample texts at a grade level within this document, making a judgment each time about whether the new text is harder or
easier than the sample texts. If it is generally harder than the texts at one level, the new text can be compared with texts at the next level and so on, until an
appropriate position is identified in the continuum of complexity.

CONTEXT RELEVANCE
This document is intended to provide guidance about determining text complexity with the important caveat that guidance should always be adjusted according
to the language and context.
Text length, which is of critical importance in grades two and three, is only specified approximately. An indicative word count is given in English on the
understanding that languages with longer words may adopt a shorter word count. Similarly, where a sentence count is given, this is on the understanding that
more very short sentences, or fewer longer sentences, might also be appropriate. The sample texts provide guidance about the scope of the content that is
expected to be covered in a grade-level text.
Familiarity is of critical importance at all grades. Content, structure, and vocabulary should be very familiar at lower grades, and the degree of familiarity will
depend on what has been taught as well as personal experience, at home and in the local community. As texts become more complex, most factors start to
become less familiar. Again, what “less familiar” means will depend on what has been taught and what most students are likely to have encountered outside
school.
"""

feature_table_nos_by_grade = {2: 16, 3: 17, 6: 19, 9: 21}
feature_tables_by_grade = {
    k: pd.read_csv(TABLEPATH / f"Table{v}.csv").set_index("Feature")
    for k, v in feature_table_nos_by_grade.items()
}
grades = list(feature_table_nos_by_grade.keys())
features = feature_tables_by_grade[2].index.unique().to_list()

grade_lines = lambda f: "\n".join(
    [
        f"GRADE {g} : {feature_tables_by_grade[g].loc[f].Scope.strip()}, {feature_tables_by_grade[g].loc[f].Elaboration.strip()}"
        for g in grades
    ]
)

feature_prompts = [
    f"""
    <feature>
    **{f}**
   
    {grade_lines(f)}
    </feature>
    """
    for f in features
]
# print("\n".join(feature_prompts))
feature_prompts = (
    "\n".join(feature_prompts)
    + """
    **Grade 2**
    At grade two, texts are so short that they are mainly simple descriptions. Texts typically have a single character engaged in a simple action, or a very brief description of a single object or event.

    **Grade 4**
    Grade four texts are typically slightly longer than grade three texts and contain more detail. However, greater complexity in one factor may be balanced by less complexity in another. For example, a shorter text may contain some less familiar content, or some less common vocabulary.

    **Grade 5**
    Grade five texts may be of varying lengths and are mainly narrative (stories) and informational. Some instructional texts may also be used. Simple non-continuous texts such as lists and tables are introduced at this level. There may be some non-conventional genre elements in the texts.  Narrative texts include details such as some limited character development, or a simple description of the setting. Information texts may include basic paratextual features: for example, subheadings or captions.  Vocabulary includes a wide range of familiar words describing concrete and abstract concepts as well as less familiar words where the context strongly supports the meaning. For example, a common technical or discipline-specific term may be used where the meaning can be inferred from prominent clues.

    **Grade 7**
    Grade 7 texts are of varying lengths, with longer texts typically being straightforward and shorter texts a little more complex. A range of familiar text types, including narrative (stories), informational, persuasive, and instructional texts, are used at this grade level. A range of simple, non-continuous formats includes tables, diagrams, maps, and graphs.  Texts typically include several minor complexities such as unfamiliar content that is clearly explained, less common vocabulary supported in context, significant implied ideas, or a less familiar structure.
    
    **Grade 8**
    Texts may be somewhat longer and more complex than grade seven texts. Text types that include narrative, informational, persuasive, and instruction are used at this grade level. A range of non-continuous formats includes tables, diagrams, maps, and graphs.  Texts typically include several minor complexities such as unfamiliar content that is clearly explained, less common vocabulary supported in context, significant implied ideas, or a less familiar structure.
    """
)

evaluation_features = (
    """
Please structure your evaluation based on the following features. Details are given for grade-levels 2, 3, 6 and 9, for other grades you will need to interpolate. 
"""
    + feature_prompts
)

# print(evaluation_features)


# %%
# load GPF text items / stimuli
#
loader = DataLoader()
all_domains = [loader.get_domain(d) for d in ["R"]]
items = loader.get_items()
gpf_example_items_json = [
    json.loads(
        i.model_dump_json(include=["text", "genre", "grade_appropriate", "explanation"])
        .replace('"grade_appropriate":', '"grade":')
        .replace('"genre":', '"type":')
    )
    for i in items
]

gpf_example_text_json = {
    "example_texts::"
    "These are example texts which are graded according to the GPF, along with explanations for that grading"
    "": gpf_example_items_json
}

# load rising items
#
with open(DATADIR / "Reading_questions_stimuli.yaml", "r") as file:
    stimuli_full = yaml.safe_load(file)

stimuli = [
    {"grade": s["question_level"], "title": s["title"], "text": s["text"]}
    for s in stimuli_full
]
for i, stim in enumerate(stimuli_full):
    if "gpf_explanation" in stim:
        stimuli[i]["explanation"] = stim["gpf_explanation"]
gt_grade = [s["grade"] for s in stimuli]

# %%
task_intro = """
Below you will see a text. Your task is to allocate a grade-level to this text, together with a short explanation modelled after the examples provided. 

To support your decision, below you will find a list of example texts with their grades. For some of these an explanation is provided.  
"""


task_item = """
Here is the text for you to classify. Please return the grade inside <grade> tags, and an explanation inside <explanation> tags, which is based on the features above. 
"""

# %%


query = """
    <text>
    Salt is something we use every day. We add it to our food to make it taste better. But did you
    know that salt is important in many other ways?
    Salt is very important for your body to work. Your body uses salt to make your muscles move and
    to help your blood flow. Salt also helps your body use the food you eat. If you have too little salt in
    you, you may feel dizzy and tired. But watch out, too much salt can also make you sick!
    Salt is also used for cleaning. Some people use it to clean soot from chimneys or mix it in water to
    clean burned pots and pans.
    Salt is also used to keep food from spoiling. For example, you can add salt to fresh meat or fish to
    dry it out so it will keep for later.
    Salt has many uses and is important for people to survive!
    </text>
"""

query = """
<text>
The Sevan trout only lives in Lake Sevan in Armenia. It has been in danger of becoming extinct for quite some time.
One reason is that about 50 years ago, whitefish, goldfish, and crayfish were put in the lake to provide more fish for people to catch and
eat. The problem was that the new fish ate a lot of the food that the Sevan trout used to eat. Another problem was that more people
came to the lake to catch the new fish and they also caught a lot of Sevan trout.
The government banned fishing in the lake and this has helped, but the fish are still endangered because there is often not enough
water in the lake for them to breed. The water levels in the lake have dropped because farmers need the lake water for their crops and
towns need water for industry and household use. We still need to find a way to save the Sevan trout.
</text>"""

query = """
<text>
Do our teeth become cleaner and cleaner the longer and harder we brush them?
British researchers say no. They have actually tried out many different alternatives, and
ended up with the perfect way to brush your teeth. A 2-minute brush, without brushing too
hard, gives the best result. If you brush hard, you harm your tooth enamel and your gums
without loosening food remnants or plaque.
Bente Hansen, an expert on tooth brushing, says that it is a good idea to hold the toothbrush
the way you hold a pen. “Start in one corner and brush your way along the whole row,” she says. “Don’t forget your tongue either! It can
actually contain loads of bacteria that may cause bad breath.”
OECD (2010), PISA 2009 Results: What Students Know and Can Do: Student Per
</text>"""

query = """
<text>
Lazy Rabbit never did any work. He had not dug the fields for his wife to sow their vegetable crop. Finally, his wife chased him out of
their house and would not let him back. Lazy Rabbit thought of a plan.
He found Big Elephant and started to tease him. “I’m so fast that you could never catch me,” he called out as he darted in between the
elephant’s legs and round and round his feet. Big Elephant was very bad tempered by the time he finally caught Lazy Rabbit’s little
white tail under his foot.
“Now, I’m going to stamp on you,” roared Big Elephant.
But Lazy Rabbit was thinking fast.
“You have to lift your foot to stamp on me and then I will run away,” cried out the crafty rabbit. “We
should have a competition to see who is the strongest. I will try to pull you into the sea. If I can’t do it then I will lie here nice and still and
you can stamp on me all you like.”
Big Elephant thought he would easily win, so he let Lazy Rabbit tie a red rope around his middle. Lazy Rabbit took one end of the red
rope and ran through the forest to his fields and tied the red rope to his plough. Then he got another rope, a blue one, and tied it to the
other end of the plough and ran over his fields to the sea.
“Hey, Giant Whale,” he called out, “I’m so strong I bet I could pull you out of the sea.” Giant Whale was furious. He swam to the shore to
teach Lazy Rabbit a lesson. He let Lazy Rabbit tie the other end of the blue rope around him and then he swam off as fast as he could.
Suddenly, to Giant Whale’s surprise, the blue rope pulled tight and no matter how hard he swam he could not pull Lazy Rabbit into the
sea.
In the forest, Big Elephant was pulling on the red rope with all his might. He was amazed by how strong Lazy Rabbit was. All day and all
night the whale and the elephant pulled and pulled. First the elephant pulled the red rope and the plough dug through the fields towards
the forest. Then the whale pulled on the blue rope and the plough dug back through the fields towards the sea. As the whale and the
elephant pulled backwards and forwards, the plough was pulled up and down the field, digging up the earth.
Finally, in the morning, Big Elephant and Giant Whale gave up. They were so embarrassed that each quietly untied his end of the rope
and slunk away. They both hoped that no one had seen them being beaten by a rabbit.
Meanwhile, Lazy Rabbit went home and proudly showed his wife their fields that were all nicely dug up and ready for planting.
</text>"""

query = """
<text>
    Jojo was walking down the stairs at home when he slipped. He fell all the way to the bottom. When
    he looked at his leg, he could see it was bent in a strange position.
    Mum came running. She touched Jojo's leg very gently, but it still hurt him. There was no blood,
    but his ankle was swelling up fast.
    Call the ambulance,' Mum said to Dad.
    Mum and Dad sat with Jojo on the stairs while they waited for the ambulance to arrive. Dad told
    Jojo not to move in case he made it worse.
</text>
"""

# query = """
# <text>
# Fatu's birthday is on Saturday. She will be six years old.
#     Fatu's mother buys her a cake and five pink balloons.
#     </text>"""

# query = """
# <text>
#     Tim and Kofi sit under a big tree. They read a book about clouds and have a lot of fun. The bell
#     rings. Tim and Kofi walk to class.
# </text>"""
prompt = (
    grade_level_text_background + evaluation_features + task_intro + task_item + query
)

#
# o1-mini gives a blank response
# 
model = "o1-mini"
# model = "claude-3-5-sonnet-20241022"
# model = "gpt-4o-2024-08-06"
# chatter = ChatController(Caller=get_caller("gpt-4o"))
chatter = ChatController(Caller=get_caller(model))
inmsg, outmsg = chatter.chat(prompt)
print(outmsg.Message)
# %%

model = "gpt-4o-2024-08-06"
# model = "claude-3-5-sonnet-20241022"
# model = "gpt-4o-mini-2024-07-18"
# model = "o1-mini-2024-09-12"

grade_level_text_background = """
You are an expert teacher with a deep understanding of foundational literacy and numeracy, particularly assessment of reading skills at primary and secondary levels. 

Your task is to rate the grade-level of a short piece of text according to the Global Proficiency Framework. You can also use the Salford reading test grade level. 

"""
evaluation_features = """When evaluating the grade level you might want to consider features such as challenge, familiary, length, predictability and sentence structure."""

def extract_tag_contents(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, text, re.DOTALL)


def cvloop(stim):
    res = {}
    cvstimuli = [
        {k: v for k, v in s.items() if k != "title"}
        for s in stimuli
        if s["title"] != stim["title"]
    ]
    cv_example_xml = [
        "<example>\n"
        + "\n".join([f"<{k}>{v}</{k}>" for k, v in x.items() if k != "title"])
        + "\n</example>"
        for x in cvstimuli
    ]

    query = "<text>\n" + json.dumps(stim["text"]) + "\n</text>"

    prompt = (
        grade_level_text_background
        + evaluation_features
        + task_intro
        + "\n".join(cv_example_xml)
        + task_item
        + query
    )
    chatter = ChatController(Caller=get_caller(model),Keep_History=False)
    inmsg, outmsg = chatter.chat(prompt)
    out = outmsg.Message

    res["out"] = out
    res["grade"] = extract_tag_contents(out, "grade")[0]
    return res


cv_res_jl = Parallel(n_jobs=10, prefer="threads")(delayed(cvloop)(s) for s in stimuli)

prompt_template = (
    grade_level_text_background
    + evaluation_features
    + task_intro
    + "\n\n<CV_EXAMPLES>\n\n"
    + task_item
    + "\n<QUERY_TEXT>\n"
)
prompt_version = md5(prompt_template.encode()).hexdigest()

prompt_cache_dir = Path(f"cv_cache/{model}/{prompt_version}")
prompt_cache_dir.mkdir(parents=True,exist_ok=True)

prompt_template_file = prompt_cache_dir / "prompt.txt"
if not prompt_template_file.exists():
    with open(prompt_template_file, "w") as file:
        file.write(prompt_template)

with open(prompt_cache_dir / "results.json", "w") as file:
    json.dump(cv_res_jl, file, indent=2)

all_example_xml = [
    "<example>\n"
    + "\n".join([f"<{k}>{v}</{k}>" for k, v in x.items() if k != "title"])
    + "\n</example>"
    for x in stimuli
]
full_prompt = (
    grade_level_text_background
    + evaluation_features
    + task_intro
    + "\n".join(all_example_xml)
    + task_item
)
with open(prompt_cache_dir / "full_prompt.txt", "w") as file:
    file.write(full_prompt)

# %%
caller = get_caller(model)
msg = LLMMessage(Role="user", Message=full_prompt)
print(len(caller.tokenize([msg])))
# %%
import numpy as np
cv_grade = [int(r["grade"]) for r in cv_res_jl]
gt_grade = [s["grade"] for s in stimuli]
x = np.array([cv_grade, gt_grade]).T
error = x[:, 1] - x[:, 0]
mse = ((error) ** 2.0).mean()
rmse = np.sqrt(mse)
print(np.c_[x, error])
#%%
cv_df = pd.DataFrame(np.c_[x, error], columns=["CV", "GT", "Error"])
print(cv_df.groupby("GT").Error.mean().to_markdown())

# %%
# 'gpt-4o-2024-08-06' 
# |   GT |    Error |
# |-----:|---------:|
# |    1 | 0        |
# |    2 | 0.25     |
# |    3 | 0.428571 |
# |    4 | 0.5      |
# |    5 | 0.5      |
# |    6 | 0.25     |
# |    7 | 0.8      |
# |    8 | 2.33333  |
# |    9 | 1.66667  |
# 
# 'claude-3-5-sonnet-20241022'
# |   GT |    Error |
# |-----:|---------:|
# |    1 | 0        |
# |    2 | 0.5      |
# |    3 | 0.428571 |
# |    4 | 0.7      |
# |    5 | 0.333333 |
# |    6 | 1.125    |
# |    7 | 1.8      |
# |    8 | 2        |
# |    9 | 1.66667  |


# 'claude-3-5-sonnet-20241022' - just examples
# |   GT |    Error |
# |-----:|---------:|
# |    1 | 0        |
# |    2 | 0.625    |
# |    3 | 0.571429 |
# |    4 | 0.8      |
# |    5 | 0.666667 |
# |    6 | 1        |
# |    7 | 1.2      |
# |    8 | 1.66667  |
# |    9 | 1.66667  |
# 'gpt-4o-2024-08-06'  - just examples
# |   GT |    Error |
# |-----:|---------:|
# |    1 | 0        |
# |    2 | 0.375    |
# |    3 | 0.285714 |
# |    4 | 0.4      |
# |    5 | 0        |
# |    6 | 0.375    |
# |    7 | 1.2      |
# |    8 | 1.66667  |
# |    9 | 2        |