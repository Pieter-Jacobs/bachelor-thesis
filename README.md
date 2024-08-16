# Active Learning and its applications to reduce text classification labeling effort
This repository contains the code used for my bachelor thesis on active learning. The thesis can be found on https://arxiv.org/abs/2109.04847, whereas the later publication can be found on https://research.rug.nl/en/publications/active-learning-for-reducing-labeling-effort-in-text-classificati-2. The BNAIC publication includes a more extensive discussion as well as a related work section. 

## Installation
Python 3 is required to run this project.
Moreover, use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following dependencies:
```bash
pip install dill hydra-core matplotlib nltk numba pandas sentence_transformers sklearn scipy torch torchtext transformers
```
## Usage
For the experiment that compared active learning and random sampling, run:
```bash
    python main.py -m dataset=sst +heuristic=random,random,random
    python main.py -m dataset=sst query_function=variation_ratio,variation_ratio,variation_ratio,predictive_entropy,predictive_entropy,predictive_entropy,predictive_entropy,mutual_information,mutual_information,mutual_information
    python plot_data.py 1
    python compute_deficiencies.py 1
```

For the experiment that compared different query sizes, run:
```bash
    python main.py -m dataset=sst parameters.Q=85,85,85,42,42,42,425,425,425 metric_file=scaling
    python plot_data.py 2
    python compute_deficiencies.py 2
```

For the third experiment that examined the performance of different heuristics, run:
```bash
    python main.py -m query_function=variation_ratio,variation_ratio,variation_ratio
    python main.py -m dataset=sst +heuristic=ret,ret,ret,rect,rect,rect,sud,sud,sud metric_file=heuristics
    python plot_data.py 3
    python compute_deficiencies.py 3
```
