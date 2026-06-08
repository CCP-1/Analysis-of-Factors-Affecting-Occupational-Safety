# -*- coding: utf-8 -*-
"""
Table 7: Predictive performance comparison of machine-learning models for
the fatality and severe-injury pathways.

Input:
    dual_pathway_machine_learning_data.xlsx

The fatality pathway models death_binary in the full sample. The
severe-injury pathway models severe_binary among non-fatal cases. User-defined
missing codes are converted to NaN, and weekend is excluded from all models.
"""
# ============================================================
#
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)


# ============================================================
#
# ============================================================

DATA_FILE = "dual_pathway_machine_learning_data.xlsx"
OUTPUT_FILE = "table7_machine_learning_performance.xlsx"


# ============================================================
# 3. Load data
# ============================================================

df = pd.read_excel(DATA_FILE)

print("Original row count: ", len(df))
print("Original column count: ", len(df.columns))
print("Original column names:")
print(df.columns.tolist())


# ============================================================
#
# ============================================================

#
categorical_vars = [
    "age_group_reg",
    "sex_group_reg",
    "Major_categories_num",
    "Month",
    "main_num",
    "event_type"
]

#
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
    "medical_or_insufficient_scene"
]

#
task_vars = [
    "maintenance_installation_task",
    "cleaning_inspection_task",
    "material_handling_task",
    "cutting_processing_task",
    "demolition_task",
    "driving_operating_task"
]

#
#
predictors = categorical_vars + scene_vars + task_vars

#
required_cols = [
    "death_binary",
    "severe_binary"
] + predictors


# ============================================================
#
# ============================================================

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"The following columns are missing; check the Excel column names: {missing_cols}")

print("Required-column check passed.")


# ============================================================
#
# ============================================================

def clean_user_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SPSS user-missing codes and special Excel missing values to NaN.

    Note:
    severe_binary may be missing for fatal cases,
    because severe_binary is used only for the severe-injury pathway,
    and such cases must not be removed from the full sample.
    """

    d = data.copy()

    #
    d = d.replace(
        ["#NULL!", "NULL", "null", "", " ", "NaN", "nan"],
        np.nan
    )

    #
    for col in required_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    #
    d.loc[d["age_group_reg"].isin([0]), "age_group_reg"] = np.nan
    d.loc[d["sex_group_reg"].isin([3]), "sex_group_reg"] = np.nan
    d.loc[d["Major_categories_num"].isin([8, 15]), "Major_categories_num"] = np.nan
    d.loc[d["main_num"].isin([8]), "main_num"] = np.nan

    return d


df_clean = clean_user_missing(df)


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

        #
        use_cols = [target] + predictors
        d_model = d[use_cols].dropna().copy()

    elif path == "severe":
        target = "severe_binary"

        #
        d = d[d["death_binary"] == 0].copy()

        #
        use_cols = [target] + predictors
        d_model = d[use_cols].dropna().copy()

    else:
        raise ValueError("path must be either 'death' or 'severe'.")

    #
    d_model[target] = d_model[target].astype(int)

    #
    for col in categorical_vars:
        d_model[col] = d_model[col].astype(int).astype(str)

    #
    for col in scene_vars + task_vars:
        d_model[col] = d_model[col].astype(int)

    X = d_model[predictors]
    y = d_model[target]

    return X, y, d_model, target


X_death, y_death, death_model_data, death_target = build_path_data(
    df_clean,
    path="death"
)

X_severe, y_severe, severe_model_data, severe_target = build_path_data(
    df_clean,
    path="severe"
)

print("\nValid fatality-pathway sample size: ", len(death_model_data))
print(y_death.value_counts().sort_index())

print("\nValid severe-injury-pathway sample size: ", len(severe_model_data))
print(y_severe.value_counts().sort_index())


# ============================================================
#
# ============================================================

try:
    onehot = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    onehot = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse = False
    )

preprocess = ColumnTransformer(
    transformers=[
        ("cat", onehot, categorical_vars),
        ("num", "passthrough", scene_vars + task_vars)
    ]
)


# ============================================================
#
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=10,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=10,
        n_jobs=-1
    )
}


# ============================================================
#
# ============================================================

def evaluate_model(
    path_name: str,
    positive_event: str,
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Train and evaluate one model.

    The default prediction threshold is 0.5.
    Positive event:
        Fatality pathway = Fatality; 
        Severe-injury pathway = Severe injury.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    brier = brier_score_loss(y_test, y_prob)

    return {
        "Pathway": path_name,
        "Model": model_name,
        "Positive event": positive_event,
        "Training N": len(y_train),
        "Test N": len(y_test),
        "AUC": round(auc, 3),
        "Accuracy": round(accuracy, 3),
        "Sensitivity": round(sensitivity, 3),
        "Specificity": round(specificity, 3),
        "Precision": round(precision, 3),
        "F1-score": round(f1, 3),
        "Brier score": round(brier, 3),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
#
# ============================================================

results = []

for model_name, model in models.items():
    print(f"\nRunning fatality pathway: {model_name}")

    result = evaluate_model(
        path_name="Fatality pathway",
        positive_event="Fatality",
        model_name=model_name,
        model=model,
        X=X_death,
        y=y_death
    )

    results.append(result)


for model_name, model in models.items():
    print(f"\nRunning severe-injury pathway: {model_name}")

    result = evaluate_model(
        path_name="Severe-injury pathway",
        positive_event="Severe injury",
        model_name=model_name,
        model=model,
        X=X_severe,
        y=y_severe
    )

    results.append(result)


performance_table = pd.DataFrame(results)


# ============================================================
#
# ============================================================

sample_info = pd.DataFrame([
    {
        "Pathway": "Fatality pathway",
        "Outcome variable": "death_binary",
        "Sample scope": "Full sample after removing missing or invalid modeling values",
        "Valid N": len(death_model_data),
        "Class 0 N": int((y_death == 0).sum()),
        "Class 1 N": int((y_death == 1).sum()),
        "Positive event": "Fatality"
    },
    {
        "Pathway": "Severe-injury pathway",
        "Outcome variable": "severe_binary",
        "Sample scope": "Non-fatal sample (death_binary=0) after removing missing or invalid modeling values",
        "Valid N": len(severe_model_data),
        "Class 0 N": int((y_severe == 0).sum()),
        "Class 1 N": int((y_severe == 1).sum()),
        "Positive event": "Severe injury"
    }
])


# ============================================================
#
# ============================================================

def category_check(data: pd.DataFrame, path_name: str) -> pd.DataFrame:
    rows = []

    for col in categorical_vars:
        counts = data[col].value_counts().sort_index()

        for level, n in counts.items():
            rows.append({
                "Pathway": path_name,
                "Variable": col,
                "Category": level,
                "n": int(n)
            })

    return pd.DataFrame(rows)


death_cat_check = category_check(death_model_data, "Fatality pathway")
severe_cat_check = category_check(severe_model_data, "Severe-injury pathway")

category_check_table = pd.concat(
    [death_cat_check, severe_cat_check],
    ignore_index=True
)


# ============================================================
#
# ============================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    performance_table.to_excel(
        writer,
        sheet_name="Table7_Model_Performance",
        index=False
    )

    sample_info.to_excel(
        writer,
        sheet_name="Sample_Sizes",
        index=False
    )

    category_check_table.to_excel(
        writer,
        sheet_name="Category_Counts",
        index=False
    )


# ============================================================
#
# ============================================================

print("\nModel performance results:")
print(performance_table)

print(f"\nResults saved to: {OUTPUT_FILE}")