# Reproducibility Materials for Dual-Pathway Analysis of Occupational Accident Outcomes

## Overview

This repository contains the data files, rule-constrained large language model
(LLM) coding instructions, and Python scripts supporting the manuscript
**"Dual-pathway analysis of severe occupational accident outcomes using
LLM-assisted narrative coding and explainable machine learning."**

The study distinguishes two outcome pathways:

- **Fatality pathway:** fatal versus non-fatal occupational accidents.
- **Severe-injury pathway:** severe versus minor injury among non-fatal accidents.

The deposited scripts reproduce the analyses based on the prepared modeling
datasets, including Firth logistic regression, machine-learning performance
comparison, SHAP variable-importance analysis, regression-SHAP comparison, and
scenario-activity co-occurrence analysis.

## Repository Contents

### Data files

| File | Description |
|---|---|
| `CSDataset source dataset.xlsx` | Source accident records extracted from the Construction Safety Dataset (CSDataset) before study-specific cleaning and variable reconstruction. |
| `Final modeling dataset.xlsx` | Final study dataset containing structured accident attributes and text-derived scenario-exposure and work-activity variables. |
| `dual_pathway_machine_learning_data.xlsx` | Analysis-ready dataset used by the machine-learning, SHAP, and scenario-activity co-occurrence scripts. |
| `severe_injury_firth_regression_data.xlsx` | Non-fatal accident dataset used for severe-injury-pathway Firth logistic regression. |
| `regression_shap_comparison_input.xlsx` | Consolidated regression and SHAP inputs used for the variable-level comparison reported in Table . |

### Coding instructions

| File | Description |
|---|---|
| `LLM_Coding_Instructions.txt` | Rule-constrained instructions, fixed labels, evidence rules, prohibited evidence, and JSON output format used to derive scenario-exposure and work-activity variables from `event_keyword` and `abstract`. |

### Analysis scripts

| File | Main output |
|---|---|
| `firth_logistic_regression.py` | Table 6: severe-injury-pathway Firth logistic regression |
| `machine_learning_model_comparison.py` | Table 7: predictive performance of the pathway-specific models |
| `shap_importance_analysis.py` | Table 8 and pathway-specific SHAP variable-importance figures |
| `regression_shap_comparison.py` | Table 9: comparison of regression and SHAP findings |
| `scenario_activity_cooccurrence_analysis.py` | Table 10 and the scenario-activity co-occurrence heatmap |
| `requirements.txt` | Python packages required by the analysis scripts |

## Core Variables

| Variable | Definition |
|---|---|
| `death_binary` | 1 = fatality; 0 = non-fatal outcome |
| `severe_binary` | Within non-fatal cases, 1 = severe injury and 0 = minor injury; structurally missing for fatal cases where applicable |
| `scene_*` variables | Binary multi-label scenario-exposure variables derived from accident narratives |
| `task_*` variables | Binary multi-label work-activity variables derived from accident narratives |

Definitions and evidence rules for all scenario and activity labels are provided
in `LLM_Coding_Instructions.txt`.

## Software Requirements

A Python 3 environment is required. Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The scripts require the following packages:

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- openpyxl
- SHAP
- tqdm

Keep the scripts and their input workbooks in the same directory unless the file
paths defined at the beginning of each script are changed.

## Reproduction Workflow

Run the scripts from the repository root:

```bash
python firth_logistic_regression.py
python machine_learning_model_comparison.py
python shap_importance_analysis.py
python regression_shap_comparison.py
python scenario_activity_cooccurrence_analysis.py
```

The scripts are independent once their listed input files are present.
`regression_shap_comparison.py` uses the supplied
`regression_shap_comparison_input.xlsx`, which consolidates the Model 5
regression estimates and SHAP rankings used in Table 9.

## Generated Outputs

| Script | Generated file(s) |
|---|---|
| `firth_logistic_regression.py` | `table6_severe_injury_firth_logistic.xlsx` |
| `machine_learning_model_comparison.py` | `table7_machine_learning_performance.xlsx` |
| `shap_importance_analysis.py` | `table8_shap_variable_importance.xlsx`, `figure_fatality_pathway_shap_importance.png`, and `figure_severe_injury_pathway_shap_importance.png` |
| `regression_shap_comparison.py` | `table9_regression_shap_comparison.xlsx` |
| `scenario_activity_cooccurrence_analysis.py` | `table10_scenario_activity_cooccurrence.xlsx` and `figure_scenario_activity_cooccurrence_heatmap.png` |

## Key Reproducibility Settings

- Machine-learning models use a stratified 70/30 training-test split with
  `random_state=42`.
- The default classification threshold is 0.5.
- The machine-learning comparison uses class weighting for imbalanced outcomes.
- The SHAP analysis uses Extra Trees and evaluates up to 5,000 test observations
  in batches of 50.
- Scenario-activity risk classification requires at least 100 co-occurring
  observations for lift-based risk labeling.
- Firth logistic regression uses a maximum of 200 iterations and a convergence
  tolerance of `1e-7`.

Additional model specifications are documented directly in the scripts.

## Scope of Reproducibility

The analysis scripts begin with prepared modeling datasets. The repository
provides the source dataset, the final modeling dataset, and the coding
instructions so that the data transformation logic can be inspected.

The current package does not automatically call an external LLM service or
rerun the original narrative-coding process. The resulting scenario-exposure
and work-activity variables are already included in the final modeling data.
Model/provider details and validation procedures for LLM-assisted coding are
reported in the manuscript.

## Data Provenance

The source records were derived from the Construction Safety Dataset
(CSDataset), which was developed from publicly available Occupational Safety
and Health Administration records:

> Ou, Z., Li, D., Tan, Z., Li, W., Liu, H., and Song, S. (2025). Building
> Safer Sites: A Large-Scale Multi-Level Dataset for Construction Safety
> Benchmark. *Proceedings of the 34th ACM International Conference on
> Information and Knowledge Management*, 6508-6512.
> https://doi.org/10.1145/3746252.3761652

Users of the source data should comply with the terms and attribution
requirements of the original data provider.

## Review Status and Anonymity

This repository is prepared for double-anonymized peer review. Author names,
affiliations, contact details, acknowledgements, and funding information are
therefore omitted. These details may be restored in the archival version after
peer review.

## Licence and Citation

No separate licence is assigned in this anonymous review package. Before public
release, the repository should include an appropriate software licence and
clarify the reuse terms for derived data without overriding the terms of the
source dataset.

When citing these materials, cite the associated manuscript after its
publication and the original CSDataset reference above.

