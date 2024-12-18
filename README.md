# Automating alignment to the Global Proficiency Framework

The [Global Proficiency Framework](https://www.edu-links.org/resources/global-proficiency-framework-reading-and-mathematics) describes structured framework of specific skills in foundational literacy and numarcy in reading and mathematics across grades 2 to 9. This is presented in the form of a detailed PDF document outlining the structure of the framework with example assessment items.

This repository provides interactive code and scripts which build on [our digitization of the GPF](https://github.com/AI-for-Education/global-proficiency-framework/). Code is included for a few experimental application. 

### How to use this repository

We provide details for a conda environment, which allows you to get more of the intermediate data and source material. To set up the environment:
- `conda env create -f environment.yml`
- `conda activate gpf`
- `dvc pull`
The conda environment includes [DVC](https://dvc.org/) which is used to version control data files within the repository. DVC pull will add these files to the `data/` folder. 

If you are using a different Python environment you will need to run `pip install -e .` to install the functions from this repository and ensure the `gpf` package is installed. 

## Code sections

### Classifying assessment questions according to GPF skill with an approach based on embeddings

This approach uses LLM-based embeddings of the stimulus text, the question text and answers, and the skill, construct and subconstruct descriptions. The folder [classify-embeddings](classify-embeddings/) contains the code for this method. The embeddings are combined and a non-linear classifier is trained to match skills to questions (including the stimulus item). This is tested with a cross-validation approach that splits on the level of stimulus texts i.e. questions from a specific text are either all in the training set, or all in the test set. Individual stimulus texts are never used in both training and test data. 


### Determining grade-level of text 

Determining the grade-level of the assessment text is crucial determining the GPF alignment of an assessment, but is somewhat seperate from the problem of determining the individual skill covered by a specific question, since the grade-level is a property of the text rather than the question. The folder [grade-level](grade-level/) contains our work prompting LLMs with details from the GPF about the features of texts and different grade-levels, and then asking the LLM to evaluate a new text item. 

The key script for this approach is [grade_level_gq.py](grade-level/grade_level_gq.py). We use a leave-one-text-out cross-validation approach. The final cross-validated performance in terms of mean square error for different grades is shown in comments at the end of this file. 