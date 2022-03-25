import pandas as pd

cardio_raw = pd.read_csv(
    "./data/cardio_train.csv",
    sep=";",
    dtype={"gluc": "category", "cholesterol": "category"},
).drop(columns=["id", "alco", "active", "smoke", "gender"])

height_in_m = cardio_raw.query("height > 147 & height < 250")["height"].apply(
    lambda height: height * 0.01
)
height_weight_cleaned = cardio_raw.drop(columns=["height"]).join(height_in_m).dropna()

bmi_raw = cardio_raw.join(
    height_weight_cleaned.apply(
        lambda row: 1.3 * row.weight / pow(row.height, 2.5), axis=1
    ).rename("BMI")
).dropna()
bmi_cleaned = bmi_raw.drop(bmi_raw.query("BMI < 16 | BMI > 60").index)

cardio_raw_with_bmi = bmi_cleaned.join(
    pd.cut(
        bmi_cleaned["BMI"],
        [0, 18.4, 24.9, 29.9, 34.9, 39.9, 100],
        labels=[
            1,
            2,
            3,
            4,
            5,
            6,
        ],
    ).rename("BMI Category")
)

blood_pressure_cleaned = cardio_raw_with_bmi.drop(
    pd.concat(
        [
            cardio_raw_with_bmi.query("ap_lo < 50"),
            cardio_raw_with_bmi.query("ap_lo > 200"),
            cardio_raw_with_bmi.query("ap_hi < 60"),
            cardio_raw_with_bmi.query("ap_hi > 240"),
        ]
    ).index
)


def blood_pressure_category(row):
    if (row["ap_hi"] < 120) and (row["ap_lo"] < 80):
        return 1
    if (row["ap_hi"] <= 129) and (row["ap_lo"] < 80):
        return 2
    if (row["ap_hi"] <= 139) or (row["ap_lo"] <= 89):
        return 3
    if (row["ap_lo"] <= 179) or (row["ap_lo"] <= 119):
        return 4
    if (row["ap_hi"] >= 180) or (row["ap_lo"] >= 120):
        return 5


cardio_cleaned_with_new_categories = blood_pressure_cleaned.join(
    pd.DataFrame(
        blood_pressure_cleaned.apply(blood_pressure_category, axis=1)
        .rename("Blood Pressure Category")
        .astype("category")
    )
)

categorial_data_set = pd.get_dummies(
    cardio_cleaned_with_new_categories.drop(
        columns=["ap_hi", "ap_lo", "height", "weight", "BMI", "age"]
    ),
    drop_first=True,
)

continuous_dataset = cardio_cleaned_with_new_categories.drop(
    columns=["Blood Pressure Category", "BMI Category"]
)

other_features_raw = pd.read_csv(
    "./data/cardio_train.csv",
    sep=";",
    dtype={
        "cholesterol": "category",
        "gender": "category",
        "smoke": "category",
        "alco": "category",
        "active": "category",
    },
).drop(
    columns=[
        "id",
        "age",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cardio",
        "cholesterol",
        "gluc",
    ]
)
all_features_dataset = pd.get_dummies(
    cardio_cleaned_with_new_categories, drop_first=True
).join(pd.get_dummies(other_features_raw, drop_first=True))
