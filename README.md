# Replication Data for: *"An Ever-changing World? A Forecasting Approach to Assessing Temporal Heterogeneity in the Causal Process of Conflict"*

Author: Luuk Boekestein (luuk.boekestein@gmail.com)

Master's thesis, Department of Peace and Conflict Research

Uppsala University, 2026

---

This repository contains all replication data, code and documentation necessary to reproduce the results of the master's thesis. 

### Repository structure

```
├── data/
│   ├── auxiliary             # Mapping of country_ids to country names
│   ├── views_data/           # VIEWS data (querysets as parquet files)
├── docs/
│   ├── codebook.xlsx         # Codebook describing the data and code in more detail
├── figs/                     # All figures seen in the main thesis and in the appendices
├── tables/                   # All tables seen in the main thesis and in the appendices
├── src/
│   ├── dynamic.py            # Main code for the forecasting approach and dynamic model manager
│   └── hurdleregression.py   # Implementation of HurdleRegression model
├── README.md                 # This file
├── replication.ipynb         # Main replication file, which runs all models and produces all figures
└── requirements.txt          # pip requirements
```

### Replication instructions

To replicate the results of the thesis, `Python 3.11` or higher is required, as well as the packages listed in the `requirements.txt` file. 

To set up the environment, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

Then, to run the replication, simply run the `replication.ipynb` file in a Jupyter notebook environment. This will produce all figures seen in the main thesis and in the appendices, which will be saved in the `figs/` folder, as well as the table with descriptive statistics as seen in the main thesis, which will be saved in the `tables/` folder. The expected runtime of the replication is around 5-10 minutes for the main results (on Apple M3) and around 20-30 minutes for the full notebook including the robustness checks.

### Documentation

#### Models, algorithms and features

The five models used in this thesis are replicated constituent models from the VIEWS fatalities002 model, which can be found on the PRIO-data GitHub repository [here](https://github.com/prio-data/viewsforecasting/tree/main). Specific details on constituent models can be found in the model definitions file [here](https://github.com/prio-data/viewsforecasting/blob/main/SystemUpdates/ModelDefinitions.py), as well as information on their [querysets](https://github.com/prio-data/viewsforecasting/blob/main/Tools/cm_querysets.py) and [features](https://github.com/lboekestein/masterthesis/tree/main/viewsforecasting/Documentation). 

More information on the specific replication details of the models is summarized in the table below:
| Model name | VIEWS model name | Algorithm | Queryset | Features |
| --- | --- | --- | --- | --- |
| Narrow | [`fatalities002_joint_narrow_xgb`](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/SystemUpdates/ModelDefinitions.py#L191) | `XGBRFRegressor(n_estimators=250)` | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Tools/cm_querysets.py#L1853) | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Documentation/Model_fatalities002_joint_narrow.md) |
| Broad | [`fatalities002_joint_broad_rf`](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/SystemUpdates/ModelDefinitions.py#L166) | `XGBRFRegressor(n_estimators=250)` | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Tools/cm_querysets.py#L3137) | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Documentation/Model_fatalities002_joint_broad.md)|
| Conflict History | [`fatalities002_conflict_history_rf`](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/SystemUpdates/ModelDefinitions.py#L49) | `XGBRFRegressor(n_estimators=250)` | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Tools/cm_querysets.py#L3081) | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Documentation/Model_fatalities002_conflict_history.md) |
| WDI | [`fatalities002_wdi_rf`](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/SystemUpdates/ModelDefinitions.py#L115) | `XGBRFRegressor(n_estimators=250)` | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Tools/cm_querysets.py#L3123) | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Documentation/Model_fatalities002_wdi_short.md) |
|VDEM |[`fatalities002_vdem_hurdle_xgb`](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/SystemUpdates/ModelDefinitions.py#L102) | `HurdleRegression()`| [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Tools/cm_querysets.py#L3109) | [link](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Documentation/Model_fatalities002_vdem_short.md) |

The `HurdleRegression` model is implemented in the [`src/hurdleregression.py`](src/hurdleregression.py) file, and is an adaptation of the class defined by VIEWS [here](https://github.com/prio-data/viewsforecasting/blob/29a347786511656a2b9c712e995f5393ec4ecab0/Tools/ViewsEstimators.py#L22), which is in turn a derivative of code provided by Geoff Ruddock [here](https://geoffruddock.com/building-a-hurdle-regression-estimator-in-scikit-learn/). 

Using these model specifications, the models are then implemented in the `replication.ipynb` notebook. More information of the underlying features of each models can also be found in the `docs/codebook.xlsx` file, which contains a codebook describing the data and the transformations of each feature in more detail. The data for each model is loaded from the `data/views_data/` folder, which contains the querysets as parquet files.

#### Forecasting approach

The core part of the analysis is the forecasting approach and is implemented in the `src/dynamic.py` file, which contains the `DynamicModelManager` class. This class takes care of all model training, forecasting and evaluation over different test sets, and is used in the `replication.ipynb` notebook to run all models. This DynamicModelManager class manages multiple DynamicModel instances, which are then trained on the sliding sets of training data, and evaluated on blocks of testing data. 

#### Evaluation

The evaluation of all models is done using the `get_disaggregated_metrics` method of the `DynamicModelManager` class, which calculates the Mean Squared Log Error for each model-test set combination. The skill score is then calculated using the `calculate_skill_score` function, which is defined in the replication notebook. 

### References

- VIEWS. 2022. Forecasting Fatalities. Department of Peace and Conflict Research, PRIO. https://uu.diva-portal.org/smash/get/diva2:1667048/FULLTEXT01.pdf.

- Hegre, Håvard, Sofia Nordenving, Michael Colaresi, Mihai Croicu, and James Dale. 2023. Ensembling and Calibration in VIEWS, Version Fatalities002. Model Documentation Series. VIEWS. https://viewsforecasting.org/wp-content/uploads/VIEWS_documentation_Ensembling_Fatalities002.pdf.

- Hegre, Håvard, Sofia Nordenving, and James Dale. 2024. Models in VIEWS, Version Fatalities002. Model Documentation Series. VIEWS. https://viewsforecasting.org/wp-content/uploads/VIEWS_documentation_models_Fatalities002.pdf.