# -*- coding: utf-8 -*-

"""
Table 10: Scenario-activity co-occurrence and high-risk combination analysis.

Input:
    dual_pathway_machine_learning_data.xlsx

The script evaluates all 11 scenario variables by 6 work-activity variables,
calculates fatality and non-fatal severe-injury lift, and exports ranked
co-occurrence tables and a heatmap.
"""
# ============================================================
#
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
#
# ============================================================

DATA_FILE = "dual_pathway_machine_learning_data.xlsx"

OUTPUT_EXCEL = "table10_scenario_activity_cooccurrence.xlsx"
OUTPUT_HEATMAP = "figure_scenario_activity_cooccurrence_heatmap.png"


#
#
MIN_COOCCUR_N_FOR_RISK = 100


#
TOP_N_FOR_TEXT = 15


# ============================================================
#
# ============================================================

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


required_cols = [
    "death_binary",
    "severe_binary",
] + scene_vars + task_vars


# ============================================================
#
# ============================================================

print("\n==================== Load data ====================")

df = pd.read_excel(DATA_FILE)

print("Original sample size: ", len(df))
print("Original variable count: ", len(df.columns))


missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"The following columns are missing; check the Excel column names: {missing_cols}"
    )

print("Required-column check passed.")


# ============================================================
#
# ============================================================

print("\n==================== Data preprocessing ====================")

data = df[required_cols].copy()


#
data = data.replace(
    ["#NULL!", "NULL", "null", "", " ", "NaN", "nan"],
    np.nan
)


#
for col in required_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# ============================================================
#
# ============================================================

#
death_missing_n = int(data["death_binary"].isna().sum())

print("Missing death_binary values: ", death_missing_n)

data = data.dropna(subset=["death_binary"]).copy()
data["death_binary"] = data["death_binary"].astype(int)


#
#
#
nondeath_missing_severe = int(
    (
        (data["death_binary"] == 0)
        & (data["severe_binary"].isna())
    ).sum()
)

death_structural_missing_severe = int(
    (
        (data["death_binary"] == 1)
        & (data["severe_binary"].isna())
    ).sum()
)

print("Structural severe_binary missing values among fatal cases: ", death_structural_missing_severe)
print("Missing severe_binary values among non-fatal cases: ", nondeath_missing_severe)


data = data[
    ~(
        (data["death_binary"] == 0)
        & (data["severe_binary"].isna())
    )
].copy()


# ============================================================
#
# ============================================================

print("\n==================== Check scenario/activity missing values ====================")

binary_vars = scene_vars + task_vars

binary_missing = data[binary_vars].isna().sum()
binary_missing = binary_missing[binary_missing > 0]


if len(binary_missing) > 0:

    print("\nThe following scenario or activity variables contain missing values:")
    print(binary_missing)

    raise ValueError(
        "Scenario or work-activity variables contain missing values."
        "Confirm whether these missing values represent 0."
        "If blank cells mean 'not present', use fillna(0)."
        "If blank cells mean 'not coded', they must not be replaced with 0."
    )


#
for col in binary_vars:
    data[col] = data[col].astype(int)


#
invalid_binary_records = []

for col in binary_vars:

    valid_mask = data[col].isin([0, 1])

    if not valid_mask.all():

        invalid_values = sorted(
            data.loc[~valid_mask, col].dropna().unique().tolist()
        )

        invalid_binary_records.append({
            "Variable": col,
            "Invalid values": invalid_values,
            "Invalid N": int((~valid_mask).sum()),
        })


if invalid_binary_records:

    invalid_df = pd.DataFrame(invalid_binary_records)

    print("\nThe following scenario or activity variables contain non-binary values:")
    print(invalid_df)

    raise ValueError(
        "Scenario or work-activity variables contain values other than 0 or 1. Check the data."
    )


print("All scenario and activity variables are complete and binary.")


# ============================================================
# 8. Construct three-class injury outcome
# ============================================================

print("\n==================== Construct three-class injury outcome ====================")


def build_outcome(row):
    """
    death_binary = 1 -> Fatality; 
    death_binary = 0 and severe_binary = 1 -> severe injury;
    death_binary = 0 and severe_binary = 0 -> minor injury.
    """

    if row["death_binary"] == 1:
        return "Fatality"

    if int(row["severe_binary"]) == 1:
        return "Severe injury"

    if int(row["severe_binary"]) == 0:
        return "Minor injury"

    return np.nan


data["Injury outcome"] = data.apply(build_outcome, axis=1)

data = data.dropna(subset=["Injury outcome"]).copy()


print("Sample size available for co-occurrence analysis: ", len(data))
print(data["Injury outcome"].value_counts())


# ============================================================
#
# ============================================================

N_total = len(data)

n_death_total = int((data["Injury outcome"] == "Fatality").sum())
n_severe_total = int((data["Injury outcome"] == "Severe injury").sum())
n_light_total = int((data["Injury outcome"] == "Minor injury").sum())


death_rate_base = n_death_total / N_total
severe_rate_base_all = n_severe_total / N_total
light_rate_base_all = n_light_total / N_total


nondeath_total = n_severe_total + n_light_total

severe_rate_base_nondeath = (
    n_severe_total / nondeath_total
    if nondeath_total > 0
    else np.nan
)


baseline_table = pd.DataFrame([
    {
        "Metric": "Total sample size",
        "Value": N_total,
    },
    {
        "Metric": "Fatality cases",
        "Value": n_death_total,
    },
    {
        "Metric": "Severe-injury cases",
        "Value": n_severe_total,
    },
    {
        "Metric": "Minor-injury cases",
        "Value": n_light_total,
    },
    {
        "Metric": "Overall fatality rate",
        "Value": death_rate_base,
    },
    {
        "Metric": "Overall severe-injury rate_full sample",
        "Value": severe_rate_base_all,
    },
    {
        "Metric": "Overall minor-injury rate_full sample",
        "Value": light_rate_base_all,
    },
    {
        "Metric": "Severe-injury rate_nondeath",
        "Value": severe_rate_base_nondeath,
    },
    {
        "Metric": "Removed death_binary missing values",
        "Value": death_missing_n,
    },
    {
        "Metric": "Retained structural severe_binary missing values among fatal cases",
        "Value": death_structural_missing_severe,
    },
    {
        "Metric": "Removed severe_binary missing values among non-fatal cases",
        "Value": nondeath_missing_severe,
    },
])


print("\nOverall fatality rate: ", round(death_rate_base, 4))
print("Severe-injury rate among non-fatal cases: ", round(severe_rate_base_nondeath, 4))


# ============================================================
#
# ============================================================

print("\n==================== Calculate scenario-activity combinations ====================")

rows = []


for scene in scene_vars:

    for task in task_vars:

        mask = (
            (data[scene] == 1)
            & (data[task] == 1)
        )

        sub = data.loc[mask].copy()

        co_n = len(sub)

        death_n = int((sub["Injury outcome"] == "Fatality").sum()) if co_n > 0 else 0
        severe_n = int((sub["Injury outcome"] == "Severe injury").sum()) if co_n > 0 else 0
        light_n = int((sub["Injury outcome"] == "Minor injury").sum()) if co_n > 0 else 0


        if co_n > 0:

            death_rate = death_n / co_n
            severe_rate_all = severe_n / co_n
            light_rate_all = light_n / co_n

        else:

            death_rate = 0
            severe_rate_all = 0
            light_rate_all = 0


        nondeath_n = severe_n + light_n

        #
        #
        severe_rate_nondeath = (
            severe_n / nondeath_n
            if nondeath_n > 0
            else np.nan
        )


        death_lift = (
            death_rate / death_rate_base
            if death_rate_base > 0
            else np.nan
        )


        severe_lift_nondeath = (
            severe_rate_nondeath / severe_rate_base_nondeath
            if pd.notna(severe_rate_nondeath) and severe_rate_base_nondeath > 0
            else np.nan
        )


        rows.append({
            "Scenario variable": scene,
            "Work-activity variable": task,

            "Co-occurrence n": co_n,
            "Co-occurrence %": co_n / N_total * 100,

            "Fatality n": death_n,
            "Fatality %": death_rate * 100,
            "Fatality lift": death_lift,

            "Severe injury n": severe_n,
            "Severe injury %_full sample": severe_rate_all * 100,

            "Minor injury n": light_n,
            "Minor injury %_full sample": light_rate_all * 100,

            "Non-fatal n": nondeath_n,
            "Severe injury %_nonfatal": (
                severe_rate_nondeath * 100
                if pd.notna(severe_rate_nondeath)
                else np.nan
            ),
            "Severe-injury lift_nondeath": severe_lift_nondeath,
        })


combo_df = pd.DataFrame(rows)


# ============================================================
#
# ============================================================

combo_df = combo_df.sort_values(
    "Co-occurrence n",
    ascending=False
).reset_index(drop=True)


combo_df["Co-occurrence frequency rank"] = combo_df["Co-occurrence n"].rank(
    method="min",
    ascending=False,
).astype(int)


combo_df["Fatality rate rank"] = combo_df["Fatality %"].rank(
    method="min",
    ascending=False,
).astype(int)


#
#
combo_df["Non-fatal severe-injury rate rank"] = (
    combo_df["Severe injury %_nonfatal"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("Int64")
)


def classify_combo(row):
    """
    Generate combination-type labels from co-occurrence frequency, fatality lift, and severe-injury lift.
    """

    tags = []

    if row["Co-occurrence frequency rank"] <= TOP_N_FOR_TEXT:
        tags.append("High-frequency co-occurrence")

    if (
        row["Co-occurrence n"] >= MIN_COOCCUR_N_FOR_RISK
        and pd.notna(row["Fatality lift"])
        and row["Fatality lift"] >= 1.25
    ):
        tags.append("Fatality-risk elevating")

    if (
        row["Non-fatal n"] >= MIN_COOCCUR_N_FOR_RISK
        and pd.notna(row["Severe-injury lift_nondeath"])
        and row["Severe-injury lift_nondeath"] >= 1.10
    ):
        tags.append("Severe-injury-risk elevating")

    if not tags:
        return "Other combination"

    return "; ".join(tags)


combo_df["Combination type"] = combo_df.apply(classify_combo, axis=1)


# ============================================================
#
# ============================================================

text_table = combo_df[
    combo_df["Combination type"] != "Other combination"
].copy()


text_table = text_table.sort_values(
    by=[
        "Co-occurrence frequency rank",
        "Fatality lift",
        "Severe-injury lift_nondeath",
    ],
    ascending=[
        True,
        False,
        False,
    ],
    na_position="last",
).reset_index(drop=True)


text_table.insert(
    0,
    "Main-text order",
    range(1, len(text_table) + 1)
)


top_frequency = combo_df.sort_values(
    "Co-occurrence n",
    ascending=False,
).head(TOP_N_FOR_TEXT).copy()


top_death = combo_df[
    combo_df["Co-occurrence n"] >= MIN_COOCCUR_N_FOR_RISK
].sort_values(
    "Fatality lift",
    ascending=False,
).head(TOP_N_FOR_TEXT).copy()


# Top15_Severe_Lift:
#
top_severe = combo_df[
    (combo_df["Non-fatal n"] >= MIN_COOCCUR_N_FOR_RISK)
    & (combo_df["Severe-injury lift_nondeath"].notna())
].sort_values(
    "Severe-injury lift_nondeath",
    ascending=False,
).head(TOP_N_FOR_TEXT).copy()


#
#
undefined_severe_rate = combo_df[
    combo_df["Non-fatal n"] == 0
].copy()


co_matrix = combo_df.pivot(
    index="Scenario variable",
    columns="Work-activity variable",
    values="Co-occurrence n",
).loc[scene_vars, task_vars]


# ============================================================
# 13. Plot co-occurrence heatmap
# ============================================================

print("\n==================== Plot co-occurrence heatmap ====================")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]

plt.rcParams["axes.unicode_minus"] = False


fig, ax = plt.subplots(figsize=(12, 7))

im = ax.imshow(co_matrix.values)

ax.set_xticks(np.arange(len(task_vars)))
ax.set_yticks(np.arange(len(scene_vars)))

ax.set_xticklabels(task_vars, rotation=45, ha="right")
ax.set_yticklabels(scene_vars)

ax.set_xlabel("Work-activity variable")
ax.set_ylabel("Scenario variable")
ax.set_title("Scenario-activity co-occurrence frequency")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Co-occurrence frequency")

plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP, dpi=300, bbox_inches="tight")
plt.close()

print(f"Heatmap saved to: {OUTPUT_HEATMAP}")


# ============================================================
#
# ============================================================

print("\n==================== Save results ====================")

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

    text_table.to_excel(
        writer,
        sheet_name="Table10_Main_Text",
        index=False,
    )

    top_frequency.to_excel(
        writer,
        sheet_name="Top15_Cooccurrence",
        index=False,
    )

    top_death.to_excel(
        writer,
        sheet_name="Top15_Fatality_Lift",
        index=False,
    )

    top_severe.to_excel(
        writer,
        sheet_name="Top15_Severe_Lift",
        index=False,
    )

    undefined_severe_rate.to_excel(
        writer,
        sheet_name="Undefined_Severe_Rate",
        index=False,
    )

    combo_df.to_excel(
        writer,
        sheet_name="All_66_Combinations",
        index=False,
    )

    co_matrix.to_excel(
        writer,
        sheet_name="Cooccurrence_Matrix",
    )

    baseline_table.to_excel(
        writer,
        sheet_name="Overall baseline rates",
        index=False,
    )


print("\nTable 10 scenario-activity co-occurrence analysis completed.")
print(f"Excel results saved to: {OUTPUT_EXCEL}")
print(f"Heatmap saved to: {OUTPUT_HEATMAP}")


# ============================================================
#
# ============================================================

print("\nTop 5 high-frequency co-occurrences:")
print(
    top_frequency[
        [
            "Scenario variable",
            "Work-activity variable",
            "Co-occurrence n",
            "Fatality %",
            "Severe injury %_nonfatal",
            "Combination type",
        ]
    ].head(5)
)


print("\nTop 5 fatality-lift combinations:")
print(
    top_death[
        [
            "Scenario variable",
            "Work-activity variable",
            "Co-occurrence n",
            "Fatality %",
            "Fatality lift",
            "Combination type",
        ]
    ].head(5)
)


print("\nTop 5 severe-injury-lift combinations:")
print(
    top_severe[
        [
            "Scenario variable",
            "Work-activity variable",
            "Non-fatal n",
            "Severe injury %_nonfatal",
            "Severe-injury lift_nondeath",
            "Combination type",
        ]
    ].head(5)
)


if len(undefined_severe_rate) > 0:
    print(
        "\nSome combinations have zero non-fatal observations, so their severe-injury rates are undefined. "
        "They were exported separately to the Undefined_Severe_Rate worksheet."
    )
else:
    print("\nNo combinations have zero non-fatal observations.")
