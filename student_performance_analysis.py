"""
Student Performance Analytics and AI Score Prediction
Data Analytics with AI mini-project.

Run:
    pip install -r requirements.txt
    python student_performance_analysis.py

The dataset is synthetic and created for educational demonstration.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path(__file__).with_name("student_performance.csv")
OUTPUT_DIR = Path(__file__).with_name("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    print("\\n=== Student Performance Analytics ===")
    print(f"Number of student records: {len(df)}")
    print("\\nSummary statistics:")
    print(df.describe(numeric_only=True).round(2).to_string())

    print("\\nCorrelation with final score:")
    print(df.select_dtypes("number").corr()["final_score"].sort_values(ascending=False).round(3).to_string())

    features = [
        "study_hours_per_week",
        "attendance_percent",
        "assignments_completed_percent",
        "previous_score",
    ]
    X = df[features]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    print("\\n=== AI Model Evaluation (test data) ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
    print(f"R-squared (R²): {r2:.3f}")

    results = X_test.copy()
    results["actual_final_score"] = y_test
    results["predicted_final_score"] = predictions.round(1)
    results.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)

    # Chart 1: study time vs final score
    plt.figure(figsize=(8, 5))
    plt.scatter(df["study_hours_per_week"], df["final_score"])
    plt.xlabel("Study hours per week")
    plt.ylabel("Final score")
    plt.title("Study Hours vs Final Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_hours_vs_score.png", dpi=150)
    plt.close()

    # Chart 2: feature correlation with final score
    correlations = df[features + ["final_score"]].corr()["final_score"].drop("final_score").sort_values()
    plt.figure(figsize=(8, 5))
    correlations.plot(kind="barh")
    plt.xlabel("Correlation with final score")
    plt.title("Factors Associated with Final Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlations.png", dpi=150)
    plt.close()

    # Example prediction for a hypothetical student
    sample = pd.DataFrame([{
        "study_hours_per_week": 10,
        "attendance_percent": 85,
        "assignments_completed_percent": 90,
        "previous_score": 75,
    }])
    predicted = float(model.predict(sample)[0])
    print("\\nExample hypothetical student prediction:")
    print(f"Predicted final score: {predicted:.1f}/100")
    print("\\nOutputs saved in:", OUTPUT_DIR)
    print("Note: predictions are estimates, not guarantees or real student assessments.")

if __name__ == "__main__":
    main()
