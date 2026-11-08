"""
Car Price Predictor - classroom demonstration.

The included CSV contains synthetic sample records. Predictions are illustrative
and should not be used for real-world vehicle valuation.
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "cars.csv"


def main():
    data = pd.read_csv(DATA_PATH)

    # Convert model year into vehicle age for a simple demonstration.
    current_year = datetime.now().year
    data["Car_Age"] = current_year - data["Year"]

    features = [
        "Brand",
        "Model",
        "Car_Age",
        "Kilometers_Driven",
        "Fuel_Type",
        "Transmission",
    ]
    target = "Price_INR"

    X = data[features]
    y = data[target]

    categorical_features = ["Brand", "Model", "Fuel_Type", "Transmission"]
    numeric_features = ["Car_Age", "Kilometers_Driven"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print("CAR PRICE PREDICTOR")
    print("-" * 40)
    print(f"Training records: {len(X_train)}")
    print(f"Testing records:  {len(X_test)}")
    print(f"Mean Absolute Error: INR {mae:,.2f}")

    example_car = pd.DataFrame(
        [
            {
                "Brand": "Maruti Suzuki",
                "Model": "Swift",
                "Car_Age": current_year - 2021,
                "Kilometers_Driven": 40000,
                "Fuel_Type": "Petrol",
                "Transmission": "Manual",
            }
        ]
    )

    estimated_price = model.predict(example_car)[0]
    print("\nExample car: 2021 Maruti Suzuki Swift")
    print(f"Estimated sample price: INR {estimated_price:,.0f}")
    print("\nNote: This is a learning demo using synthetic data.")


if __name__ == "__main__":
    main()
