# Replication Data for: *"An Ever-changing World? A Forecasting Approach to Assessing Temporal Heterogeneity in the Causal Process of Conflict"*

Author: Luuk Boekestein (luuk.boekestein@gmail.com)

Master's thesis, Department of Peace and Conflict Research
Uppsala University, 2026

---

This repository contains all replication data, code and documentation necessary to reproduce the results of the master's thesis. 

### Structure of the repository

The respository is structured as follows:
- `data/`: contains all data used in the thesis, including the raw VIEWS data as well as a csv file linking country_ids to country names.
- `docs/`: contains a codebook describing the data and code in more detail.
- `figures/`: contains all figures seen in the main thesis and in the appendices.
- `src/`: contains all source code used for the main parts of the analysis, which are referenced in the main replication file
- `replication.ipynb`: the main replication file, which contains all code necessary to reproduce the results of the thesis. This file references the source code in `src/` and uses the data in `data/` to produce the figures in `figures/`.

### Replication instructions

To replicate the results of the thesis, `Python 3.11` or higher is required, as well as the packages listed in the `requirements.txt` file. 

To set up the environment, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

Then, to run the replication, simply run the `replication.ipynb` file in a Jupyter notebook environment. This will produce all figures seen in the main thesis and in the appendices, which will be saved in the `figs/` folder.