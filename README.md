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

