# Bachelor Thesis: Active Learning and its applications to reduce text classification labeling effort
This repository contains the code used for my bachelor thesis on active learning. Which you can find here:

## Installation
Python 3 is required to run this project.
Moreover, use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following dependencies:
```bash
pip install dill hydra matplotlib nltk numba pandas sentence_transformers sklearn scipy torch torchtext transformers
```
## Usage
For the experiment that compared active learning and random sampling, run:
```bash
    python main.py -m dataset=sst,sst,sst +heuristic=random
    python main.py -m dataset=sst,sst,sst query_function=variation_ratio,predictive_entropy,mutual_information
    python plot_data.py 1
    python compute_deficiencies.py 1
```

For the experiment that compared different query sizes, run:
```bash
    python main.py -m dataset=sst,sst,sst parameters.Q=85,42,425 metric_file=scaling
    python plot_data.py 2
    python compute_deficiencies.py 2
```

For the third experiment that examined the performance of different heuristics, run:
```bash
    python main.py -m dataset=sst,sst,sst 
    python main.py -m dataset=sst,sst,sst +heuristic=ret,rect,sud metric_file=heuristics
    python plot_data.py 3
    python compute_deficiencies.py 3
```