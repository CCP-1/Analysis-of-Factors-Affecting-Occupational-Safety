# -*- coding: utf-8 -*-

"""
Table 9: Comparison of regression results and SHAP variable importance.

Input:
    regression_shap_comparison_input.xlsx

Category-level Model 5 regression estimates are summarized at the variable
level and compared with variable-level SHAP rankings. Individual category odds
ratios are not treated as equivalent to the aggregate SHAP contribution of a
variable.
"""
# ============================================================
#
# ============================================================

import re
import numpy as np
import pandas as pd


# ============================================================
#
# ============================================================

INPUT_FILE = "regression_shap_comparison_input.xlsx"
OUTPUT_FILE = "table9_regression_shap_comparison.xlsx"


DEATH_SHEET = "Fatality_Regression"
SEVERE_SHEET = "Severe_Regression"
SHAP_SHEET = "SHAP_Importance"


# ============================================================
#
# ============================================================

#
#
#
#
SHAP_CORE_RANK = 5
SHAP_IMPORTANT_RANK = 10
SHAP_MEDIUM_RANK = 15


# ============================================================
#
# ============================================================

variable_name_map = {
    "Major_categories": "Major_categories_num",
    "Major_categories_num": "Major_categories_num",

    "main": "main_num",
    "main_num": "main_num",

    "medical_or_insufficient_event": "medical_or_insufficient_scene",
    "medical_or_insufficient_scene": "medical_or_insufficient_scene",
}


def normalize_variable_name(var):
    """
    Normalize variable names across Tables 5, 6, and 8.
    """

    if pd.isna(var):
        return np.nan

    var = str(var).strip()

    return variable_name_map.get(var, var)


# ============================================================
#
# ============================================================

multi_category_vars = {
    "age_group_reg",
    "sex_group_reg",
    "Major_categories_num",
    "Month",
    "main_num",
    "event_type",
}


def get_variable_type(var):
    """
    Distinguish multicategory variables from binary variables.

    Multicategory variable:
        the regression table contains multiple categories;
        the SHAP table reports the aggregate contribution of the variable.

    Binary variable:
        these are usually scenario or work-activity variables;
        the regression term generally corresponds directly to the SHAP variable.
    """

    if var in multi_category_vars:
        return "Multicategory variable"

    return "Binary variable"


# ============================================================
#
# ============================================================

def parse_regression_cell(value):
    """
    Parse regression results in formats such as:
        1.268*** (1.153–1.394)
        0.918* (0.849–0.994)
        1.073 (0.668–1.724)
        —
        -

    Returns:
        OR
        Significance stars
        Statistically significant
        Direction

    Direction rules:
        asterisk(s) and OR > 1: risk
        asterisk(s) and OR < 1: protective
        no asterisk: not significant
    """

    if pd.isna(value):
        return {
            "OR": np.nan,
            "stars": "",
            "significant": False,
            "direction": "Not included or missing",
        }

    text = str(value).strip()

    if text in ["—", "-", "", "nan", "NaN", "None"]:
        return {
            "OR": np.nan,
            "stars": "",
            "significant": False,
            "direction": "Not included or missing",
        }

    match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(\*{1,3})?",
        text
    )

    if not match:
        return {
            "OR": np.nan,
            "stars": "",
            "significant": False,
            "direction": "Unparseable",
        }

    or_value = float(match.group(1))
    stars = match.group(2) if match.group(2) else ""

    significant = stars != ""

    if not significant:
        direction = "Not significant"
    elif or_value > 1:
        direction = "Risk"
    elif or_value < 1:
        direction = "Protective"
    else:
        direction = "No direction"

    return {
        "OR": or_value,
        "stars": stars,
        "significant": significant,
        "direction": direction,
    }


# ============================================================
#
# ============================================================

def load_regression_sheet(sheet_name, path_name):
    """
    Read Table 5 or Table 6 and extract Model 5 results.

    Note:
        Tables 5 and 6 contain category-level results;
        retain category-level details before aggregating to the variable level.
    """

    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    required = [
        "Variable layer",
        "Variable",
        "Category/value",
        "Model 5",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{sheet_name}  is missing the following columns: {missing}."
            f"Expected headers are: Variable layer, Variable, Category/value, and Model 5."
        )

    df = df.copy()

    df["Variable layer"] = df["Variable layer"].ffill()
    df["Variable"] = df["Variable"].ffill()

    df = df.dropna(
        subset=["Variable", "Category/value"],
        how="any"
    ).copy()

    df["Variable"] = df["Variable"].apply(normalize_variable_name)
    df["Variable type"] = df["Variable"].apply(get_variable_type)

    parsed = df["Model 5"].apply(parse_regression_cell)

    df["OR"] = parsed.apply(lambda x: x["OR"])
    df["Significance stars"] = parsed.apply(lambda x: x["stars"])
    df["Statistically significant"] = parsed.apply(lambda x: x["significant"])
    df["Category direction"] = parsed.apply(lambda x: x["direction"])

    df["Pathway"] = path_name

    keep_cols = [
        "Pathway",
        "Variable layer",
        "Variable",
        "Variable type",
        "Category/value",
        "Model 5",
        "OR",
        "Significance stars",
        "Statistically significant",
        "Category direction",
    ]

    return df[keep_cols]


# ============================================================
#
# ============================================================

def summarize_regression_by_variable(reg_df, path_name):
    """
    Aggregate category-level regression results to the variable level.

    Aggregation rules:
        1. no significant category:
            Regression significance = Not significant
            Regression direction = Not significant

        2. all significant categories have OR > 1:
            Regression direction = Risk

        3. all significant categories have OR < 1:
            Regression direction = Protective

        4. significant categories include both OR > 1 and OR < 1:
            Regression direction = Mixed

    This aggregation permits comparison with variable-level SHAP rankings.
    """

    rows = []

    for var, g in reg_df.groupby("Variable"):

        g = g.copy()
        var_type = get_variable_type(var)

        g_sig = g[g["Statistically significant"] == True].copy()

        n_terms = len(g)
        n_sig = len(g_sig)

        risk_count = int((g_sig["Category direction"] == "Risk").sum())
        protect_count = int((g_sig["Category direction"] == "Protective").sum())

        if n_sig == 0:
            reg_sig = "Not significant"
            reg_direction = "Not significant"
            direction_detail = "No significant categories"
            sig_summary = "No significant categories"

        else:
            reg_sig = "Significant"

            if risk_count > 0 and protect_count == 0:
                reg_direction = "Risk"
                direction_detail = "All significant categories show a risk direction"

            elif risk_count == 0 and protect_count > 0:
                reg_direction = "Protective"
                direction_detail = "All significant categories show a protective direction"

            elif risk_count > 0 and protect_count > 0:
                reg_direction = "Mixed"
                direction_detail = "Significant categories include both risk and protective directions"

            else:
                reg_direction = "Other"
                direction_detail = "Directions of significant categories cannot be classified"

            sig_items = []

            for _, row in g_sig.iterrows():
                sig_items.append(
                    f"{row['Category/value']}:OR={row['OR']:.3f}"
                    f"{row['Significance stars']},{row['Category direction']}"
                )

            #
            if len(sig_items) > 6:
                sig_summary = "; ".join(sig_items[:6]) + f"; plus {len(sig_items) - 6} additional significant terms"
            else:
                sig_summary = "; ".join(sig_items)

        rows.append({
            "Pathway": path_name,
            "Variable": var,
            "Variable type": var_type,
            "Regression terms": n_terms,
            "Significant terms": n_sig,
            "Significant risk-direction terms": risk_count,
            "Significant protective-direction terms": protect_count,
            "Regression significance": reg_sig,
            "Regression direction": reg_direction,
            "Direction description": direction_detail,
            "Significant-term summary": sig_summary,
        })

    return pd.DataFrame(rows)


# ============================================================
#
# ============================================================

def load_shap_sheet():
    """
    Read the variable-level SHAP table.

    The expected worksheet structure is:
        row 1: Rank | Fatality pathway | blank | Severe-injury pathway | blank
        row 2: blank | Original variable | MeanAbsSHAP | Original variable | MeanAbsSHAP
        data start on row 3

    The function automatically locates the row containing 'Original variable'.
    """

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHAP_SHEET,
        header=None
    )

    header_row = None

    for idx in range(len(raw)):
        row_values = [
            str(x).strip()
            for x in raw.iloc[idx].tolist()
        ]

        if "Original variable" in row_values:
            header_row = idx
            break

    if header_row is None:
        raise ValueError(
            "The 'Original variable' header was not found in the SHAP worksheet."
            "Check the format of the SHAP variable-importance worksheet."
        )

    data = raw.iloc[header_row + 1:].copy()

    data = data.iloc[:, 0:5]
    data.columns = [
        "Rank",
        "Fatality pathway variable",
        "Fatality MeanAbsSHAP",
        "Severe-injury pathway variable",
        "Severe-injury MeanAbsSHAP",
    ]

    data = data.dropna(how="all").copy()

    death_shap = data[
        [
            "Rank",
            "Fatality pathway variable",
            "Fatality MeanAbsSHAP",
        ]
    ].dropna(
        subset=["Fatality pathway variable"]
    ).copy()

    death_shap.columns = [
        "Fatality SHAP rank",
        "Variable",
        "Fatality MeanAbsSHAP",
    ]


    severe_shap = data[
        [
            "Rank",
            "Severe-injury pathway variable",
            "Severe-injury MeanAbsSHAP",
        ]
    ].dropna(
        subset=["Severe-injury pathway variable"]
    ).copy()

    severe_shap.columns = [
        "Severe-injury SHAP rank",
        "Variable",
        "Severe-injury MeanAbsSHAP",
    ]


    death_shap["Variable"] = death_shap["Variable"].apply(normalize_variable_name)
    severe_shap["Variable"] = severe_shap["Variable"].apply(normalize_variable_name)

    death_shap["Fatality SHAP rank"] = pd.to_numeric(
        death_shap["Fatality SHAP rank"],
        errors="coerce"
    )

    severe_shap["Severe-injury SHAP rank"] = pd.to_numeric(
        severe_shap["Severe-injury SHAP rank"],
        errors="coerce"
    )

    death_shap["Fatality MeanAbsSHAP"] = pd.to_numeric(
        death_shap["Fatality MeanAbsSHAP"],
        errors="coerce"
    )

    severe_shap["Severe-injury MeanAbsSHAP"] = pd.to_numeric(
        severe_shap["Severe-injury MeanAbsSHAP"],
        errors="coerce"
    )


    shap_summary = pd.merge(
        death_shap,
        severe_shap,
        on="Variable",
        how="outer"
    )

    shap_summary["Variable type"] = shap_summary["Variable"].apply(get_variable_type)

    return shap_summary


# ============================================================
#
# ============================================================

def shap_level(rank):
    """
    Assign an importance level from the SHAP rank.
    """

    if pd.isna(rank):
        return "Not listed in SHAP table"

    if rank <= SHAP_CORE_RANK:
        return "Core"

    if rank <= SHAP_IMPORTANT_RANK:
        return "Important"

    if rank <= SHAP_MEDIUM_RANK:
        return "Moderate"

    return "Low"


# ============================================================
#
# ============================================================

def classify_variable(row):
    """
    Classify each variable using variable-level regression summaries and SHAP rankings.

    Note:
        Classification is performed at the variable level, not the category level.
    """

    death_rank = row.get("Fatality SHAP rank", np.nan)
    severe_rank = row.get("Severe-injury SHAP rank", np.nan)

    death_reg_sig = row.get("Fatality regression significance", "Not significant")
    severe_reg_sig = row.get("Severe-injury regression significance", "Not significant")

    death_dir = row.get("Fatality regression direction", "Not significant")
    severe_dir = row.get("Severe-injury regression direction", "Not significant")

    death_top5 = pd.notna(death_rank) and death_rank <= SHAP_CORE_RANK
    severe_top5 = pd.notna(severe_rank) and severe_rank <= SHAP_CORE_RANK

    death_top10 = pd.notna(death_rank) and death_rank <= SHAP_IMPORTANT_RANK
    severe_top10 = pd.notna(severe_rank) and severe_rank <= SHAP_IMPORTANT_RANK

    death_top15 = pd.notna(death_rank) and death_rank <= SHAP_MEDIUM_RANK
    severe_top15 = pd.notna(severe_rank) and severe_rank <= SHAP_MEDIUM_RANK

    death_sig = death_reg_sig == "Significant"
    severe_sig = severe_reg_sig == "Significant"


    # ------------------------------------------------------------
    # 1. Dual-pathway core factor
    # ------------------------------------------------------------

    if death_top5 and severe_top5 and death_sig and severe_sig:
        return "Dual-pathway core factor"


    # ------------------------------------------------------------
    # 2. Stable high-risk factor
    # ------------------------------------------------------------

    if (
        death_sig and severe_sig
        and death_dir == "Risk"
        and severe_dir == "Risk"
        and death_top15
        and severe_top15
    ):
        return "Stable high-risk factor"


    # ------------------------------------------------------------
    # 3. Pathway-divergent factor
    # ------------------------------------------------------------

    if (
        death_sig and severe_sig
        and death_dir != severe_dir
        and (death_top15 or severe_top15)
    ):
        return "Pathway-divergent factor"

    if (
        (death_dir == "Mixed" or severe_dir == "Mixed")
        and (death_top15 or severe_top15)
    ):
        return "Pathway-divergent factor"

    if (
        (death_dir == "Protective" or severe_dir == "Protective")
        and (death_top5 or severe_top5)
    ):
        return "Pathway-divergent factor"


    # ------------------------------------------------------------
    # 4. pathway-specific factors
    # ------------------------------------------------------------

    if death_top10 and not severe_top15:
        return "Fatality-pathway-specific factor"

    if severe_top10 and not death_top15:
        return "Severe-injury-pathway-specific factor"


    # ------------------------------------------------------------
    # 5. Machine-learning-identified factor
    # ------------------------------------------------------------

    if (death_top15 and not death_sig) or (severe_top15 and not severe_sig):
        return "Machine-learning-identified factor"


    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    if (death_sig or severe_sig) and not (death_top15 or severe_top15):
        return "Regression-significant but low predictive contribution"


    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    return "Low-contribution or supplementary factor"


# ============================================================
#
# ============================================================

def generate_interpretation(row):
    """
    Generate a concise interpretation note for manuscript drafting.
    """

    var = row["Variable"]
    var_type = row["Variable type"]

    death_rank = row.get("Fatality SHAP rank", np.nan)
    severe_rank = row.get("Severe-injury SHAP rank", np.nan)

    death_dir = row.get("Fatality regression direction", "Not significant")
    severe_dir = row.get("Severe-injury regression direction", "Not significant")

    identify_type = row.get("Identification type", "")

    if var_type == "Multicategory variable":
        base = (
            f"{var}  is a multicategory variable; regression results describe category differences relative to the reference category, "
            f"whereas SHAP reflects the variable's aggregate predictive contribution."
        )
    else:
        base = (
            f"{var}  is binary, so its regression direction and SHAP importance can be compared directly at the variable level."
        )

    shap_part = (
        f"Fatality-pathway SHAP rank: {death_rank}; severe-injury-pathway SHAP rank: {severe_rank}."
    )

    reg_part = (
        f"Fatality-pathway regression direction: {death_dir}; severe-injury-pathway regression direction: {severe_dir}."
    )

    type_part = f"Overall classification: {identify_type}."

    return base + shap_part + reg_part + type_part


# ============================================================
#
# ============================================================

print("\n==================== Load regression tables ====================")

death_reg_detail = load_regression_sheet(
    sheet_name=DEATH_SHEET,
    path_name="Fatality pathway"
)

severe_reg_detail = load_regression_sheet(
    sheet_name=SEVERE_SHEET,
    path_name="Severe-injury pathway"
)


print("Fatality-pathway regression detail rows: ", len(death_reg_detail))
print("Severe-injury-pathway regression detail rows: ", len(severe_reg_detail))


print("\n==================== Aggregate regression results to variable level ====================")

death_reg_summary = summarize_regression_by_variable(
    death_reg_detail,
    path_name="Fatality pathway"
)

severe_reg_summary = summarize_regression_by_variable(
    severe_reg_detail,
    path_name="Severe-injury pathway"
)


death_reg_summary = death_reg_summary.rename(columns={
    "Regression terms": "Fatality regression terms",
    "Significant terms": "Fatality significant terms",
    "Significant risk-direction terms": "Fatality risk terms",
    "Significant protective-direction terms": "Fatality protective terms",
    "Regression significance": "Fatality regression significance",
    "Regression direction": "Fatality regression direction",
    "Direction description": "Fatality direction description",
    "Significant-term summary": "Fatality significant-term summary",
})


severe_reg_summary = severe_reg_summary.rename(columns={
    "Regression terms": "Severe-injury regression terms",
    "Significant terms": "Severe-injury significant terms",
    "Significant risk-direction terms": "Severe-injury risk terms",
    "Significant protective-direction terms": "Severe-injury protective terms",
    "Regression significance": "Severe-injury regression significance",
    "Regression direction": "Severe-injury regression direction",
    "Direction description": "Severe-injury direction description",
    "Significant-term summary": "Severe-injury significant-term summary",
})


death_reg_summary = death_reg_summary.drop(columns=["Pathway"])
severe_reg_summary = severe_reg_summary.drop(columns=["Pathway"])


print("\n==================== Load SHAP table ====================")

shap_summary = load_shap_sheet()


print("\n==================== Merge regression and SHAP results ====================")

table9 = pd.merge(
    shap_summary,
    death_reg_summary,
    on=["Variable", "Variable type"],
    how="outer"
)

table9 = pd.merge(
    table9,
    severe_reg_summary,
    on=["Variable", "Variable type"],
    how="outer"
)


# ============================================================
#
# ============================================================

table9["Fatality SHAP level"] = table9["Fatality SHAP rank"].apply(shap_level)
table9["Severe-injury SHAP level"] = table9["Severe-injury SHAP rank"].apply(shap_level)

table9["Identification type"] = table9.apply(classify_variable, axis=1)

table9["Interpretation note"] = table9.apply(generate_interpretation, axis=1)


# ============================================================
#
# ============================================================

type_order = {
    "Dual-pathway core factor": 1,
    "Stable high-risk factor": 2,
    "Pathway-divergent factor": 3,
    "Fatality-pathway-specific factor": 4,
    "Severe-injury-pathway-specific factor": 5,
    "Machine-learning-identified factor": 6,
    "Regression-significant but low predictive contribution": 7,
    "Low-contribution or supplementary factor": 8,
}

table9["Type order"] = table9["Identification type"].map(type_order).fillna(99)

table9["Best SHAP rank"] = table9[
    [
        "Fatality SHAP rank",
        "Severe-injury SHAP rank",
    ]
].min(axis=1)


table9 = table9.sort_values(
    by=[
        "Type order",
        "Best SHAP rank",
        "Variable",
    ],
    ascending=[
        True,
        True,
        True,
    ]
).reset_index(drop=True)


table9.insert(
    0,
    "No.",
    range(1, len(table9) + 1)
)


table9 = table9.drop(columns=["Type order"])


# ============================================================
#
# ============================================================

table9_main = table9[
    [
        "No.",
        "Variable",
        "Variable type",
        "Fatality regression significance",
        "Fatality regression direction",
        "Fatality SHAP rank",
        "Fatality SHAP level",
        "Severe-injury regression significance",
        "Severe-injury regression direction",
        "Severe-injury SHAP rank",
        "Severe-injury SHAP level",
        "Identification type",
    ]
].copy()


# ============================================================
#
# ============================================================

method_note = pd.DataFrame([
    {
        "Note item": "Comparison level",
        "Description": (
            "Tables 5 and 6 contain category-level regression results; Table 8 contains variable-level SHAP results."
            "Regression results are first aggregated to the variable level and then compared with variable-level SHAP rankings."
        )
    },
    {
        "Note item": "Multicategory-variable treatment",
        "Description": (
            "For multicategory variables such as event_type, Major_categories_num, Month, and main_num, "
            "the regression direction is determined from significant categories. If they include both OR > 1 and OR < 1, "
            "the variable-level direction is classified as mixed."
        )
    },
    {
        "Note item": "Binary-variable treatment",
        "Description": (
            "For scenario and work-activity variables, regression terms generally correspond directly to variables, "
            "allowing direct comparison with variable-level SHAP rankings."
        )
    },
    {
        "Note item": "Identification type",
        "Description": (
            "Identification type is determined from regression significance, regression direction, and SHAP ranks in both pathways, "
            "distinguishing dual-pathway core, stable high-risk, pathway-divergent, pathway-specific, and machine-learning-identified factors."
        )
    },
])


# ============================================================
#
# ============================================================

print("\n==================== Save results ====================")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    table9_main.to_excel(
        writer,
        sheet_name="Table9_Main_Text",
        index=False
    )

    table9.to_excel(
        writer,
        sheet_name="Table9_Variable_Comparison",
        index=False
    )

    death_reg_summary.to_excel(
        writer,
        sheet_name="Fatality_Reg_Summary",
        index=False
    )

    severe_reg_summary.to_excel(
        writer,
        sheet_name="Severe_Reg_Summary",
        index=False
    )

    shap_summary.to_excel(
        writer,
        sheet_name="SHAP_Variable_Summary",
        index=False
    )

    death_reg_detail.to_excel(
        writer,
        sheet_name="Fatality_Reg_Detail",
        index=False
    )

    severe_reg_detail.to_excel(
        writer,
        sheet_name="Severe_Reg_Detail",
        index=False
    )

    method_note.to_excel(
        writer,
        sheet_name="Method_Notes",
        index=False
    )


print("\nTable 9 regression-SHAP comparison completed.")
print(f"Results saved to: {OUTPUT_FILE}")

print("\nIdentification-type distribution:")
print(table9["Identification type"].value_counts())
