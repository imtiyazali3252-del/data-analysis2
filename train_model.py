"""
Multi-Feature Indian Salary Model Trainer
----------------------------------------
This script trains, evaluates, and compares multiple Machine Learning models
to predict salaries using a multi-feature Indian Salary dataset.

Algorithms Compared:
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The best performing model (based on R2 score) is saved to models/model.pkl
along with the preprocessor object saved to models/preprocessor.pkl.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


class SalaryModelTrainer:
    """
    Handles preprocessing, multi-model training, comparison, evaluation,
    visualization, and serialization of the best model and preprocessor.
    """

    def __init__(self, data_path: str, models_dir: str, screenshots_dir: str):
        """
        Initializes path settings and defines feature columns.
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.screenshots_dir = screenshots_dir

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Feature Column Definitions
        self.num_cols = ["Age", "YearsExperience"]
        self.cat_cols = [
            "EducationLevel",
            "JobTitle",
            "City",
            "CompanySize",
            "Industry",
            "WorkType",
        ]

        # Define preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
            ]
        )

        # Model storage
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        # Evaluation metrics storage
        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def load_data(self) -> None:
        """Loads the dataset from the CSV file."""
        print(f"[1/7] Loading dataset from '{self.data_path}'...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded. Shape: {self.df.shape}")
        print("First 5 rows of data:")
        print(self.df.head())

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """Splits the multi-feature dataset into training and testing sets."""
        print("[2/7] Splitting data into train and test sets...")
        if self.df is None:
            raise ValueError("Dataset is not loaded. Call load_data() first.")

        # X contains all 8 input features
        X = self.df[self.num_cols + self.cat_cols]
        # y contains the target salary
        y = self.df["Salary"].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train size: {self.X_train.shape[0]} samples, Test size: {self.X_test.shape[0]} samples")

    def preprocess_and_train(self) -> None:
        """Preprocesses data using ColumnTransformer and trains all candidate models."""
        print("[3/7] Preprocessing features and training candidate models...")
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data is not split. Call split_data() first.")

        # Fit preprocessor on training data and transform it
        X_train_processed = self.preprocessor.fit_transform(self.X_train)

        # Train each algorithm
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train_processed, self.y_train)
            
        print("All models trained successfully.")

    def evaluate_models(self) -> None:
        """Evaluates all trained models on the test set and selects the best one."""
        print("[4/7] Evaluating and comparing models on test set...")
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data is not split. Call split_data() first.")

        # Transform test data using fitted preprocessor
        X_test_processed = self.preprocessor.transform(self.X_test)

        best_r2 = -float("inf")

        print("\nModel Comparison Table:")
        print("=" * 90)
        print(f"{'Model Name':<25} | {'MAE':<15} | {'MSE':<18} | {'RMSE':<15} | {'R2 Score':<10}")
        print("=" * 90)

        for name, model in self.models.items():
            # Predict
            y_pred = model.predict(X_test_processed)

            # Metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            self.results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "predictions": y_pred}
            print(f"{name:<25} | {mae:<15.2f} | {mse:<18.2f} | {rmse:<15.2f} | {r2:<10.4f}")

            # Keep track of the best model based on R2 Score
            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name
                self.best_model = model

        print("=" * 90)
        print(f"\n>>> Best Model Selected: '{self.best_model_name}' with R2 Score: {best_r2:.4f}\n")

    def save_artifacts(self) -> None:
        """Saves the selected best model and preprocessor to disk."""
        print(f"[5/7] Saving best model and preprocessor to '{self.models_dir}'...")
        os.makedirs(self.models_dir, exist_ok=True)

        model_path = os.path.join(self.models_dir, "model.pkl")
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.pkl")

        # Save model and column preprocessor
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)

        print(f"Best Model ('{self.best_model_name}') saved at: {model_path}")
        print(f"Preprocessor saved at: {preprocessor_path}")

    def visualize_training(self) -> None:
        """Plots Actual vs. Predicted values for the best model to check prediction accuracy."""
        print("[6/7] Visualizing predictions vs actual values...")
        if self.best_model_name is None:
            raise ValueError("No model has been evaluated. Call evaluate_models() first.")

        os.makedirs(self.screenshots_dir, exist_ok=True)
        plot_path = os.path.join(self.screenshots_dir, "training_visualization.png")

        y_pred = self.results[self.best_model_name]["predictions"]

        # Plot actual vs predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred, color="teal", alpha=0.6, edgecolors="k", label="Test Predictions")

        # Plot 45-degree identity line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Fit Line")

        plt.xlabel("Actual Salary (₹)")
        plt.ylabel("Predicted Salary (₹)")
        plt.title(f"Actual vs Predicted Salaries ({self.best_model_name})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Prediction visualization saved at: {plot_path}")

    def run_pipeline(self) -> None:
        """Runs the entire model training pipeline."""
        print("=" * 80)
        print("Starting Multi-Feature Indian Salary Prediction Pipeline")
        print("=" * 80)
        self.load_data()
        self.split_data()
        self.preprocess_and_train()
        self.evaluate_models()
        self.save_artifacts()
        self.visualize_training()
        print("[7/7] Multi-feature pipeline completed successfully!")
        print("=" * 80)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Salary_Data.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

    trainer = SalaryModelTrainer(
        data_path=DATASET_PATH,
        models_dir=MODELS_DIR,
        screenshots_dir=SCREENSHOTS_DIR
    )
    trainer.run_pipeline()
