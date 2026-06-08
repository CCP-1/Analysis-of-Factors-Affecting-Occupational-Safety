"""
Table 6: Firth logistic regression for the severe-injury pathway.

Input:
    severe_injury_firth_regression_data.xlsx

The analysis is restricted to non-fatal cases and models severe_binary,
where 1 denotes severe injury and 0 denotes minor injury. The weekend variable
is not included.
"""
# ============================================================
#
# Outcome variable:severe_binary,1=Severe injury,0=Minor injury
#
#
# ============================================================

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from numpy.linalg import LinAlgError
import warnings

# ----------------------------
#
# ----------------------------
INPUT_FILE = "severe_injury_firth_regression_data.xlsx"
OUTPUT_FILE = "table6_severe_injury_firth_logistic.xlsx"

# ----------------------------
# 2. Load data
# ----------------------------
df = pd.read_excel(INPUT_FILE)

#
#
if "death_binary" in df.columns:
    df = df[df["death_binary"] == 0].copy()

# ----------------------------
#
# ----------------------------

y_col = "severe_binary"

model1_vars = [
    "age_group_reg",
    "sex_group_reg",
    "Major_categories_num"
]

model2_vars = model1_vars + [
    "Month",
    "main_num"
]

model3_vars = model2_vars + [
    "event_type"
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
    "medical_or_insufficient_scene"
]

model4_vars = model3_vars + scene_vars

task_vars = [
    "maintenance_installation_task",
    "cleaning_inspection_task",
    "material_handling_task",
    "cutting_processing_task",
    "demolition_task",
    "driving_operating_task"
]

model5_vars = model4_vars + task_vars

#
models = {
    "Model 1": model1_vars,
    "Model 2": model2_vars,
    "Model 3": model3_vars,
    "Model 4": model4_vars,
    "Model 5": model5_vars
}

# ----------------------------
#
# ----------------------------
#
#

categorical_vars = [
    "age_group_reg",
    "sex_group_reg",
    "Major_categories_num",
    "Month",
    "main_num",
    "event_type"
]

ref_levels = {
    "age_group_reg": "4",             # 60 or older
    "sex_group_reg": "2",             # Female
    "Major_categories_num": "14",     # Transportation and Material Moving Occupations
    "Month": "12",                   #
    "main_num": "12",                # Thunderstorm
    "event_type": "14"               # event_type=14
}

#
label_maps = {
    "age_group_reg": {
        "1": "Under 18",
        "2": "18-29",
        "3": "30-59",
        "4": "60 or older"
    },
    "sex_group_reg": {
        "1": "Male",
        "2": "Female"
    },
    "Month": {str(i): str(i) for i in range(1, 13)},
    "event_type": {str(i): str(i) for i in range(0, 15)},
    "main_num": {
        "1": "Clear",
        "2": "Clouds",
        "3": "Drizzle",
        "4": "Dust",
        "5": "Fog",
        "6": "Haze",
        "7": "Mist",
        "9": "Rain",
        "10": "Smoke",
        "11": "Snow",
        "12": "Thunderstorm"
    },
    "Major_categories_num": {
        "1": "Administrative Support Occupations",
        "2": "Construction and Extraction Occupations",
        "3": "Farming, Forestry, and Fishing Occupations",
        "4": "Handlers, Equipment Cleaners, Helpers, and Laborers",
        "5": "Installation, Maintenance, and Repair Occupations",
        "6": "Machine Operators, Assemblers, and Inspectors",
        "7": "Managerial and Professional Specialty Occupations",
        "9": "Precision Production, Craft, and Repair Occupations",
        "10": "Production Occupations",
        "11": "Sales Occupations",
        "12": "Service Occupations",
        "13": "Technicians and Related Support Occupations",
        "14": "Transportation and Material Moving Occupations"
    }
}

# ----------------------------
# 5. Data preprocessing
# ----------------------------

needed_cols = [y_col] + model5_vars
existing_needed_cols = [c for c in needed_cols if c in df.columns]

missing_cols = [c for c in needed_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"The following variables are missing; check the column names: {missing_cols}")

df_model = df[needed_cols].copy()

#
df_model = df_model.dropna().copy()

#
df_model[y_col] = df_model[y_col].astype(int)

#
def to_clean_str(x):
    if pd.isna(x):
        return np.nan
    try:
        fx = float(x)
        if fx.is_integer():
            return str(int(fx))
        return str(fx)
    except Exception:
        return str(x)

for col in categorical_vars:
    df_model[col] = df_model[col].apply(to_clean_str)

#
binary_vars = scene_vars + task_vars
for col in binary_vars:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0).astype(int)

#
print("Sample size: ", len(df_model))
print("severe_binary distribution:")
print(df_model[y_col].value_counts().sort_index())

# ----------------------------
#
# ----------------------------

def make_design_matrix(data, predictors):
    """
    Construct the design matrix:
    - categorical variables are dummy-coded using the specified reference category
    - scenario/activity variables enter directly as binary 0/1 variables
    """
    X_parts = []

    for col in predictors:
        if col in categorical_vars:
            s = data[col].astype(str)

            if col not in ref_levels:
                raise ValueError(f"{col} has no specified reference category")

            ref = ref_levels[col]
            levels = sorted(s.dropna().unique(), key=lambda v: float(v) if v.replace(".", "", 1).isdigit() else v)

            if ref not in levels:
                raise ValueError(f"{col}  reference category {ref} is absent from the current data; observed categories are {levels}")

            #
            ordered_levels = [ref] + [lv for lv in levels if lv != ref]

            cat = pd.Categorical(s, categories=ordered_levels)
            dummies = pd.get_dummies(cat, prefix=col, prefix_sep="=", drop_first=True, dtype=float)
            X_parts.append(dummies)

        else:
            X_parts.append(data[[col]].astype(float))

    X = pd.concat(X_parts, axis=1)

    #
    zero_var_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if zero_var_cols:
        print("Removed zero-variance columns: ", zero_var_cols)
        X = X.drop(columns=zero_var_cols)

    #
    X.insert(0, "Intercept", 1.0)

    return X

# ----------------------------
#
# ----------------------------

def penalized_loglik(X, y, beta):
    """
    Firth penalized likelihood:
    l*(beta)=l(beta)+0.5*log|I(beta)|
    """
    eta = np.clip(X @ beta, -35, 35)
    p = expit(eta)
    eps = 1e-12

    loglik = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    w = p * (1 - p)
    I = X.T @ (w[:, None] * X)

    sign, logdet = np.linalg.slogdet(I)
    if sign <= 0:
        logdet = -1e12

    return loglik + 0.5 * logdet


def fit_firth_logistic(X_df, y, max_iter=200, tol=1e-7):
    """
    Estimate the Firth logistic regression.
    Return beta, SE, OR, confidence intervals, and p-values.
    """
    X = X_df.values.astype(float)
    y = np.asarray(y).astype(float)

    n, k = X.shape
    beta = np.zeros(k)

    converged = False

    for iteration in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        p = expit(eta)
        w = np.clip(p * (1 - p), 1e-9, None)

        I = X.T @ (w[:, None] * X)

        try:
            I_inv = np.linalg.inv(I)
        except LinAlgError:
            I_inv = np.linalg.pinv(I)

        # hat values: h_i = w_i * x_i' I^{-1} x_i
        h = w * np.sum((X @ I_inv) * X, axis=1)

        # Firth adjusted score
        adjustment = (0.5 - p) * h
        U_star = X.T @ (y - p + adjustment)

        try:
            step = np.linalg.solve(I, U_star)
        except LinAlgError:
            step = np.linalg.pinv(I) @ U_star

        #
        old_ll = penalized_loglik(X, y, beta)
        step_factor = 1.0

        while step_factor > 1e-6:
            beta_new = beta + step_factor * step
            new_ll = penalized_loglik(X, y, beta_new)
            if new_ll >= old_ll:
                break
            step_factor /= 2

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break

        beta = beta_new

    #
    eta = np.clip(X @ beta, -35, 35)
    p = expit(eta)
    w = np.clip(p * (1 - p), 1e-9, None)
    I = X.T @ (w[:, None] * X)

    try:
        cov = np.linalg.inv(I)
    except LinAlgError:
        cov = np.linalg.pinv(I)

    se = np.sqrt(np.diag(cov))
    z = beta / se
    p_value = 2 * (1 - norm.cdf(np.abs(z)))

    result = pd.DataFrame({
        "term": X_df.columns,
        "coef": beta,
        "se": se,
        "z": z,
        "p_value": p_value,
        "OR": np.exp(beta),
        "CI_lower": np.exp(beta - 1.96 * se),
        "CI_upper": np.exp(beta + 1.96 * se),
        "converged": converged,
        "iterations": iteration + 1
    })

    return result


# ----------------------------
#
# ----------------------------

model_results = {}

for model_name, predictors in models.items():
    print(f"\nRunning {model_name} ...")
    X = make_design_matrix(df_model, predictors)
    y = df_model[y_col].values

    res = fit_firth_logistic(X, y)
    res = res[res["term"] != "Intercept"].copy()
    res["model"] = model_name

    model_results[model_name] = res

    print(f"{model_name} completed:")
    print("Convergence status: ", res["converged"].iloc[0])
    print("Number of iterations: ", res["iterations"].iloc[0])


# ----------------------------
#
# ----------------------------

def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def format_or_ci(row):
    return f"{row['OR']:.3f}{stars(row['p_value'])} ({row['CI_lower']:.3f}–{row['CI_upper']:.3f})"

def parse_term(term):
    """
    Split each model term into its variable name and category/value.
    """
    if "=" in term:
        var, level = term.split("=", 1)
        var = var.strip()
        level = level.strip()
        label = label_maps.get(var, {}).get(level, level)
        return var, label
    else:
        return term, "1"

def variable_layer(var):
    if var in ["age_group_reg", "sex_group_reg", "Major_categories_num"]:
        return "Model 1: Individual factors"
    elif var in ["Month", "main_num"]:
        return "Model 2: Environmental factors"
    elif var == "event_type":
        return "Model 3: Accident mechanism"
    elif var in scene_vars:
        return "Model 4: Scene factors"
    elif var in task_vars:
        return "Model 5: Task factors"
    else:
        return ""

#
all_terms = []
for m, res in model_results.items():
    all_terms.extend(res["term"].tolist())

all_terms = list(dict.fromkeys(all_terms))

#
var_order = (
    ["age_group_reg", "sex_group_reg", "Major_categories_num"] +
    ["Month", "main_num"] +
    ["event_type"] +
    scene_vars +
    task_vars
)

def term_sort_key(term):
    var, cat = parse_term(term)
    var_idx = var_order.index(var) if var in var_order else 999

    #
    raw_cat = cat
    reverse_maps = {v: k for k, v in label_maps.get(var, {}).items()}
    raw_value = reverse_maps.get(cat, cat)

    try:
        cat_idx = float(raw_value)
    except Exception:
        cat_idx = 999

    return (var_idx, cat_idx)

ordered_terms = sorted(all_terms, key=term_sort_key)

table_rows = []

for term in ordered_terms:
    var, cat = parse_term(term)
    row = {
        "Variable layer": variable_layer(var),
        "Variable": var,
        "Category/value": cat
    }

    for model_name in models.keys():
        res = model_results[model_name]
        match = res[res["term"] == term]
        if len(match) == 0:
            row[model_name] = "—"
        else:
            row[model_name] = format_or_ci(match.iloc[0])

    table_rows.append(row)

table6 = pd.DataFrame(table_rows)

# ----------------------------
#
# ----------------------------

#
sample_info = pd.DataFrame({
    "Item": [
        "Final modeling sample size",
        "Minor-injury cases (severe_binary=0)",
        "Severe-injury cases (severe_binary=1)",
        "Weekend included",
        "Outcome variable",
        "Target event"
    ],
    "Result": [
        len(df_model),
        int((df_model[y_col] == 0).sum()),
        int((df_model[y_col] == 1).sum()),
        "No",
        "severe_binary",
        "severe_binary=1 (severe injury)"
    ]
})

# Category_Counts
level_checks = []
for col in categorical_vars:
    vc = df_model[col].value_counts().sort_index()
    for level, n in vc.items():
        level_checks.append({
            "Variable": col,
            "Category": level,
            "Label": label_maps.get(col, {}).get(str(level), str(level)),
            "n": int(n),
            "Reference category": "Yes" if str(level) == ref_levels.get(col) else "No"
        })

level_check_df = pd.DataFrame(level_checks)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    sample_info.to_excel(writer, sheet_name="sample_info", index=False)
    level_check_df.to_excel(writer, sheet_name="levels_check", index=False)
    table6.to_excel(writer, sheet_name="table6_final", index=False)

    for model_name, res in model_results.items():
        out = res.copy()
        out["OR_95CI"] = out.apply(format_or_ci, axis=1)
        out.to_excel(writer, sheet_name=model_name.replace(" ", "_"), index=False)

print(f"\nResults saved to: {OUTPUT_FILE}")