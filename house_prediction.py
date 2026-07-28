import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

COLUMN_MAP = {
    "sqft": "sqft",
    "bedrooms" : "bedrooms",
    "bathrooms" : "bathrooms",
    "price" : "price",
}

DATA_PATH ="data/house_prices.csv"
OUTPUT_DIR="outputs" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    return df

def explore_data(df):
    print("\n== Dataset Overview ===")
    print(df.head())
    print("\n=== Summary Statistics ===")
    print(df.describe())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    # Pairwise scatter plots vs prices
    fig, axes = plt. subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["sqft", "bedrooms", "bathrooms"]):
        sns.scatterplot(
    x=df[col],
    y=df["price"],
    alpha=0.5,
    s=15,
    ax=ax
)
        ax.set_xlabel(col)
        ax.set_ylabel("price")
        ax.set_title(f"price vs {col}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_vs_price.png", dpi=150)
        plt.close()

        print(f"\nSaved EDA plots to {OUTPUT_DIR}/")

def train_model(df):
        X = df[["sqft", "bedrooms", "bathrooms"]]
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n=== Model Evaluation ===")
        for feature, coef in zip(X.columns, model.coef_):
            print(f" {feature:12s}: {coef:,.2f}")
        print(f" {'intercept':12}: {model.intercept_:,.2f}")

        print("\nModel Performance (test set) ===")
        print(f" R^2 Score: {r2:.4f}")
        print(f" RMSE: {rmse:,.2f}")
        print(f" MAE: {mae:,.2f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", label="Perfect Prediction")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Prices")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted.png", dpi=150)
        plt.close()

        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel("Predicted Price")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/residuals.png", dpi=150)
        plt.close()

        joblib.dump(model, f"{OUTPUT_DIR}/linear_regression_model.pkl")
        print(f"\nSaved trained model to {OUTPUT_DIR}/linear_regression_model.pkl")
        print(f"Saved evaluation plots to {OUTPUT_DIR}/")

        return model, (r2, rmse, mae)

def predict_price(model, sqft, bedrooms, bathrooms):
        X_new = pd.DataFrame(
            [[sqft, bedrooms, bathrooms]], columns=["sqft", "bedrooms", "bathrooms"]
        )
        return model.predict(X_new)[0]

if __name__ == "__main__":
        df = load_data()
        explore_data(df)
        model, metrics = train_model(df)

        print("\n=== Example Prediction ===")
        example = predict_price(model, sqft=2200, bedrooms=4, bathrooms=2)
        print(f" A 2200 sqft, 3-bed, 2-bath house is predicted to cost: ${example:,.2f}")