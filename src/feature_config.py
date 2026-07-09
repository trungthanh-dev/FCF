TARGET = "Consumer_Total_MomentaryFuel"
LEAKAGE_KEYWORDS = [
    "ShaftPower",
    "ShaftTorque",
    "RotationSpeed",
    "MomentaryFuel"
]

CATEGORICAL_KEYWORDS = [
    "FuelType"
]

CATEGORICAL_FEATURES = [
    "Consumer_Boiler_FuelType",
]
DROP_FEATURES = [
    "FuelType"
]
NON_FEATURE_COLUMNS = ["ship", "index"]

def detect_leakage_features(df):
    leakage = []
    for col in df.columns:
        if col == TARGET:
            continue
        if any(keyword in col for keyword in LEAKAGE_KEYWORDS):
            leakage.append(col)
    return leakage

def detect_drop_features(df):
    drop_features = []
    for col in df.columns:
        if any(keyword in col for keyword in DROP_FEATURES):
            drop_features.append(col)
    return drop_features

def detect_categorical_features(df):
    return[
        col for col in df.columns
        if any(keyword in col for keyword in CATEGORICAL_KEYWORDS)
    ]

def get_features(df):
    leakage = detect_leakage_features(df)
    drop_kw = detect_drop_features(df)
    return[
        col for col in df.columns
        if col not in leakage
        and col not in drop_kw
        and col not in NON_FEATURE_COLUMNS
        and col != TARGET
    ]

def split_feature_target(df):
    X = df[get_features(df)]
    y = df[TARGET]
    return X, y

def print_feature_summary(df):
    print(f"Total features: {len(df.columns)}")
    print(f"Leakage faetures: {len(detect_leakage_features(df))}")
    print(f"Selected features: {len(get_features(df))}")
    print("\nLeakage Features:")
    for col in detect_leakage_features(df):
        print(f"-{col}")