# -*- coding: utf-8 -*-
"""
Table 8: Extra Trees SHAP variable-importance analysis for the fatality and
severe-injury pathways.

Input:
    dual_pathway_machine_learning_data.xlsx

Outputs:
    table8_shap_variable_importance.xlsx
    figure_fatality_pathway_shap_importance.png
    figure_severe_injury_pathway_shap_importance.png
"""
# ============================================================
#
# ============================================================

import math
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier


# ============================================================
#
# ============================================================

DATA_FILE = "dual_pathway_machine_learning_data.xlsx"

OUTPUT_EXCEL = "table8_shap_variable_importance.xlsx"
OUTPUT_DEATH_FIG = "figure_fatality_pathway_shap_importance.png"
OUTPUT_SEVERE_FIG = "figure_severe_injury_pathway_shap_importance.png"

#
#
#
SHAP_MAX_N = 5000

#
#
SHAP_BATCH_SIZE = 50


# ============================================================
# 3. Load data
# ============================================================

print("\n==================== Load data ====================", flush=True)

df = pd.read_excel(DATA_FILE)

print("Original row count: ", len(df), flush=True)
print("Original column count: ", len(df.columns), flush=True)


# ============================================================
#
# ============================================================

categorical_vars = [
    "age_group_reg",
    "sex_group_reg",
    "Major_categories_num",
    "Month",
    "main_num",
    "event_type",
]


scene_vars = [
    "height_fall_scene",
    "same_level_surface_scene",
    "vehicle_mobile_equipment_scene",
    "lifting_fallingload_scene",
    "machinery_entrapment_scene",
    "electrical_energy_scene",
    "pressure_gas_chemical_scene",
    "fire_heat_explosion_scene",
    "natural_environment_scene",
    "violence_scene",
    "medical_or_insufficient_scene",
]


task_vars = [
    "maintenance_installation_task",
    "cleaning_inspection_task",
    "material_handling_task",
    "cutting_processing_task",
    "demolition_task",
    "driving_operating_task",
]


#
#
predictors = categorical_vars + scene_vars + task_vars

required_cols = [
    "death_binary",
    "severe_binary",
] + predictors


# ============================================================
#
# ============================================================

print("\n==================== Check variables ====================", flush=True)

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(f"The following columns are missing; check the Excel column names: {missing_cols}")

print("Required-column check passed.", flush=True)


# ============================================================
#
# ============================================================

def clean_user_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SPSS user-missing codes and special Excel missing values to NaN.

    Note:
        severe_binary may be missing for fatal cases;
        severe_binary is used only for the severe-injury pathway;
        and such cases must not be removed from the full sample.
    """

    d = data.copy()

    d = d.replace(
        ["#NULL!", "NULL", "null", "", " ", "NaN", "nan"],
        np.nan,
    )

    for col in required_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d.loc[d["age_group_reg"].isin([0]), "age_group_reg"] = np.nan
    d.loc[d["sex_group_reg"].isin([3]), "sex_group_reg"] = np.nan
    d.loc[d["Major_categories_num"].isin([8, 15]), "Major_categories_num"] = np.nan
    d.loc[d["main_num"].isin([8]), "main_num"] = np.nan

    return d


print("\n==================== Missing-value processing ====================", flush=True)

df_clean = clean_user_missing(df)

print("Missing-value processing completed.", flush=True)


# ============================================================
#
# ============================================================

def build_path_data(data: pd.DataFrame, path: str):
    """
    Construct the modeling dataset for the selected pathway.

    path = "death":
        Fatality pathway; 
        the sample comprises all eligible records;
        the outcome variable is death_binary.

    path = "severe":
        Severe-injury pathway; 
        first restrict the data to non-fatal cases with death_binary=0;
        then use severe_binary as the outcome variable.
    """

    d = data.copy()

    if path == "death":
        target = "death_binary"

        use_cols = [target] + predictors
        d_model = d[use_cols].dropna().copy()

    elif path == "severe":
        target = "severe_binary"

        d = d[d["death_binary"] == 0].copy()

        use_cols = [target] + predictors
        d_model = d[use_cols].dropna().copy()

    else:
        raise ValueError("path must be either 'death' or 'severe'.")

    d_model[target] = d_model[target].astype(int)

    for col in categorical_vars:
        d_model[col] = d_model[col].astype(int).astype(str)

    for col in scene_vars + task_vars:
        d_model[col] = d_model[col].astype(int)

    X = d_model[predictors]
    y = d_model[target]

    return X, y, d_model, target


print("\n==================== Construct pathway datasets ====================", flush=True)

X_death, y_death, death_model_data, death_target = build_path_data(
    df_clean,
    path="death",
)

X_severe, y_severe, severe_model_data, severe_target = build_path_data(
    df_clean,
    path="severe",
)


print("\nValid fatality-pathway sample size: ", len(death_model_data), flush=True)
print(y_death.value_counts().sort_index(), flush=True)

print("\nValid severe-injury-pathway sample size: ", len(severe_model_data), flush=True)
print(y_severe.value_counts().sort_index(), flush=True)


# ============================================================
#
# ============================================================

print("\n==================== Configure models ====================", flush=True)

try:
    onehot = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )
except TypeError:
    onehot = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse=False,
    )

preprocess = ColumnTransformer(
    transformers=[
        ("cat", onehot, categorical_vars),
        ("num", "passthrough", scene_vars + task_vars),
    ]
)

extra_trees_model = ExtraTreesClassifier(
    n_estimators=50,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=10,
    n_jobs=-1,
)

print("Model configuration completed.", flush=True)


# ============================================================
#
# ============================================================

def get_encoded_feature_names(fitted_preprocess: ColumnTransformer):
    """
    Return feature names generated by the ColumnTransformer.

    Name categorical features as 'variable=category';
    retain the original names of scenario and work-activity variables.
    """

    encoder = fitted_preprocess.named_transformers_["cat"]

    encoded_names = []

    for var, categories in zip(categorical_vars, encoder.categories_):
        categories = list(categories)

        #
        used_categories = categories[1:]

        for cat in used_categories:
            encoded_names.append(f"{var}={cat}")

    feature_names = encoded_names + scene_vars + task_vars

    return feature_names


def map_encoded_to_original(encoded_feature: str) -> str:
    """Map one-hot encoded feature names to their original variables."""

    if "=" in encoded_feature:
        return encoded_feature.split("=")[0]

    return encoded_feature


# ============================================================
#
# ============================================================

def get_positive_class_shap_values(shap_values):
    """
    Handle SHAP output formats used by different SHAP versions.

    Objective:
        Extract SHAP values for the positive class (class 1).

    Fatality pathway:1 = Fatality
    Severe-injury pathway:1 = Severe injury
    """

    if isinstance(shap_values, list):
        return shap_values[1]

    shap_values = np.asarray(shap_values)

    #
    if shap_values.ndim == 3:
        if shap_values.shape[2] == 2:
            return shap_values[:, :, 1]

        if shap_values.shape[0] == 2:
            return shap_values[1, :, :]

    #
    if shap_values.ndim == 2:
        return shap_values

    raise ValueError(f"Unrecognized SHAP output shape: {shap_values.shape}")


# ============================================================
#
# ============================================================

def compute_shap_in_batches(explainer, X_array, batch_size, desc):
    """Calculate SHAP values in batches and display a tqdm progress bar."""

    n_samples = X_array.shape[0]
    n_batches = math.ceil(n_samples / batch_size)

    shap_list = []

    print(
        f"\n{desc}:{n_samples} samples will be processed in {n_batches} batches.",
        flush=True,
    )

    for start in tqdm(
        range(0, n_samples, batch_size),
        total=n_batches,
        desc=desc,
        ncols=100,
        file=sys.stdout,
    ):
        end = min(start + batch_size, n_samples)

        X_batch = X_array[start:end]

        shap_values_raw = explainer.shap_values(X_batch)
        shap_values_pos = get_positive_class_shap_values(shap_values_raw)

        shap_list.append(shap_values_pos)

    shap_values_all = np.vstack(shap_list)

    return shap_values_all


# ============================================================
#
# ============================================================

def run_shap_for_path(
    path_name: str,
    positive_event: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_fig: str,
):
    """
    Train Extra Trees for one pathway and calculate SHAP variable importance.

    Outputs:
        1. SHAP importance at the original-variable level;
        2. SHAP importance at the one-hot category level;
        3. a bar chart of the top 15 variables.
    """

    print(
        f"\n==================== Start pathway: {path_name} ====================",
        flush=True,
    )

    start_time = time.time()

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    print(f"{path_name}:Splitting training and test sets...", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print(f"{path_name}:Training sample size = {len(X_train)}", flush=True)
    print(f"{path_name}:Test sample size = {len(X_test)}", flush=True)

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    print(f"{path_name}:Training the Extra Trees model...", flush=True)

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", extra_trees_model),
    ])

    pipe.fit(X_train, y_train)

    print(f"{path_name}:Extra Trees training completed.", flush=True)

    fitted_preprocess = pipe.named_steps["preprocess"]
    fitted_model = pipe.named_steps["model"]

    feature_names = get_encoded_feature_names(fitted_preprocess)

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    print(f"{path_name}:Transforming test-set features...", flush=True)

    X_test_encoded = fitted_preprocess.transform(X_test)

    if SHAP_MAX_N is not None and X_test_encoded.shape[0] > SHAP_MAX_N:
        rng = np.random.default_rng(42)

        sample_idx = rng.choice(
            X_test_encoded.shape[0],
            size=SHAP_MAX_N,
            replace=False,
        )

        X_shap = X_test_encoded[sample_idx]

        print(
            f"{path_name}:The test set is large; randomly sampled {SHAP_MAX_N} samples for SHAP.",
            flush=True,
        )

    else:
        X_shap = X_test_encoded

        print(
            f"{path_name}:Using the full test set of {X_shap.shape[0]} samples for SHAP.",
            flush=True,
        )

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    print(f"{path_name}:Creating the SHAP explainer...", flush=True)

    explainer = shap.TreeExplainer(fitted_model)

    print(
        f"{path_name}:Starting batched SHAP calculation; see the progress bar below.",
        flush=True,
    )

    shap_values_pos = compute_shap_in_batches(
        explainer=explainer,
        X_array=X_shap,
        batch_size=SHAP_BATCH_SIZE,
        desc=f"{path_name} SHAP progress",
    )

    print(f"{path_name}:SHAP calculation completed.", flush=True)

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    print(f"{path_name}:Preparing category-level SHAP results...", flush=True)

    encoded_importance = pd.DataFrame({
        "Pathway": path_name,
        "Positive event": positive_event,
        "Feature": feature_names,
        "Original variable": [
            map_encoded_to_original(f)
            for f in feature_names
        ],
        "MeanAbsSHAP": np.abs(shap_values_pos).mean(axis=0),
    })

    encoded_importance = encoded_importance.sort_values(
        "MeanAbsSHAP",
        ascending=False,
    ).reset_index(drop=True)

    encoded_importance["Rank"] = encoded_importance.index + 1

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    print(f"{path_name}:Aggregating SHAP values to the original-variable level...", flush=True)

    variable_importance = (
        encoded_importance
        .groupby(
            ["Pathway", "Positive event", "Original variable"],
            as_index=False,
        )["MeanAbsSHAP"]
        .sum()
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )

    variable_importance["Rank"] = variable_importance.index + 1

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    print(f"{path_name}:Plotting the SHAP bar chart...", flush=True)

    top15 = variable_importance.head(15).copy()
    top15_plot = top15.sort_values("MeanAbsSHAP", ascending=True)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(9, 6))

    plt.barh(
        top15_plot["Original variable"],
        top15_plot["MeanAbsSHAP"],
    )

    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Variable")
    plt.title(f"{path_name} SHAP variable importance (Top 15)")

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{path_name}:Figure saved -> {output_fig}", flush=True)

    end_time = time.time()

    print(
        f"{path_name}:Completed,elapsed time {round(end_time - start_time, 2)} seconds.",
        flush=True,
    )

    return variable_importance, encoded_importance


# ============================================================
#
# ============================================================

death_variable_importance, death_encoded_importance = run_shap_for_path(
    path_name="Fatality pathway",
    positive_event="Fatality",
    X=X_death,
    y=y_death,
    output_fig=OUTPUT_DEATH_FIG,
)


severe_variable_importance, severe_encoded_importance = run_shap_for_path(
    path_name="Severe-injury pathway",
    positive_event="Severe injury",
    X=X_severe,
    y=y_severe,
    output_fig=OUTPUT_SEVERE_FIG,
)


# ============================================================
#
# ============================================================

print("\n==================== Generate Table 8 ====================", flush=True)

death_top15 = death_variable_importance.head(15).copy()
severe_top15 = severe_variable_importance.head(15).copy()


table8 = pd.DataFrame({
    "Rank": range(1, 16),
    "Fatality pathway variable": death_top15["Original variable"].values,
    "Fatality pathway Mean|SHAP|": death_top15["MeanAbsSHAP"].round(6).values,
    "Severe-injury pathway variable": severe_top15["Original variable"].values,
    "Severe-injury pathway Mean|SHAP|": severe_top15["MeanAbsSHAP"].round(6).values,
})


# ============================================================
#
# ============================================================

sample_info = pd.DataFrame([
    {
        "Pathway": "Fatality pathway",
        "Outcome variable": "death_binary",
        "Positive event": "Fatality",
        "Valid N": len(death_model_data),
        "Class 0 N": int((y_death == 0).sum()),
        "Class 1 N": int((y_death == 1).sum()),
        "SHAP explanation model": "Extra Trees",
        "SHAP sample setting": (
            "Full test set"
            if SHAP_MAX_N is None
            else f"Random test-set sample: {SHAP_MAX_N}"
        ),
        "SHAP_BATCH_SIZE": SHAP_BATCH_SIZE,
    },
    {
        "Pathway": "Severe-injury pathway",
        "Outcome variable": "severe_binary",
        "Positive event": "Severe injury",
        "Valid N": len(severe_model_data),
        "Class 0 N": int((y_severe == 0).sum()),
        "Class 1 N": int((y_severe == 1).sum()),
        "SHAP explanation model": "Extra Trees",
        "SHAP sample setting": (
            "Full test set"
            if SHAP_MAX_N is None
            else f"Random test-set sample: {SHAP_MAX_N}"
        ),
        "SHAP_BATCH_SIZE": SHAP_BATCH_SIZE,
    },
])


# ============================================================
#
# ============================================================

print("\n==================== Save results ====================", flush=True)

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

    table8.to_excel(
        writer,
        sheet_name="Table8_SHAP_Importance",
        index=False,
    )

    death_variable_importance.to_excel(
        writer,
        sheet_name="Fatality_Variable_SHAP",
        index=False,
    )

    severe_variable_importance.to_excel(
        writer,
        sheet_name="Severe_Variable_SHAP",
        index=False,
    )

    death_encoded_importance.to_excel(
        writer,
        sheet_name="Fatality_Category_SHAP",
        index=False,
    )

    severe_encoded_importance.to_excel(
        writer,
        sheet_name="Severe_Category_SHAP",
        index=False,
    )

    sample_info.to_excel(
        writer,
        sheet_name="Sample_Sizes",
        index=False,
    )


# ============================================================
#
# ============================================================

print("\n==================== Completed ====================", flush=True)

print("\nTable 8: SHAP variable-importance ranking")
print(table8)

print(f"\nExcel results saved to: {OUTPUT_EXCEL}", flush=True)
print(f"Fatality-pathway figure saved to: {OUTPUT_DEATH_FIG}", flush=True)
print(f"Severe-injury-pathway figure saved to: {OUTPUT_SEVERE_FIG}", flush=True)
