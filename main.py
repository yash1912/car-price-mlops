import mlflow
import utils
import pandas as pd
import logging
import sys
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
import argparse
import config
import webbrowser

# Argument parser for test mode
parser = argparse.ArgumentParser()
parser.add_argument('--in_test_mode', action='store_true', help="Run in test mode")
args = parser.parse_args()

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler('app.log'))

# Load dataset
df = pd.read_csv(config.DATASET_PATH)

# Set MLflow experiment
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

# Preprocess data
df = utils.preprocess_data(df)

# Push data to lakeFS
utils.push_data_to_lakefs(df=df, data_name="cars_data", repo_name=config.LAKEFS_REPO_NAME, version="v1")

# Split data into features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = utils.get_data_splits(X, y)

# Start MLflow run
with mlflow.start_run():
    # Train or load model
    if not args.in_test_mode:
        tuned_rf_model, best_params = utils.tune_random_forest_model(X_train, y_train)
        utils.save_model(tuned_rf_model, config.MODEL_PATH)
        mlflow.sklearn.log_model(tuned_rf_model, "random_forest_model")
        mlflow.log_params(best_params)
    else:
        tuned_rf_model = utils.load_model(config.MODEL_PATH)

    # Evaluate model
    rmse, mse, mae = utils.evaluate_model(tuned_rf_model, X_test, y_test)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)

    # Modify test data and re-evaluate
    X_test_modified = utils.modify_test_data(X_test)
    X_test_modified.to_csv("data/X_test_modified.csv", index=False)
    rmse_modified, mse_modified, mae_modified = utils.evaluate_model(
        tuned_rf_model,
        X_test_modified.drop(columns=["Selling_Price", "price_preds"], errors="ignore"),
        y_test.loc[X_test_modified.index]
    )
    mlflow.log_metric("rmse_modified", rmse_modified)
    mlflow.log_metric("mse_modified", mse_modified)
    mlflow.log_metric("mae_modified", mae_modified)

    # Prepare data for Evidently reports
    X_train_clean = X_train.drop(columns=["Selling_Price", "price_preds"], errors="ignore")
    X_test_clean = X_test.drop(columns=["Selling_Price", "price_preds"], errors="ignore")
    X_test_modified_clean = X_test_modified.drop(columns=["Selling_Price", "price_preds"], errors="ignore")

    # Add target and prediction columns for regression analysis
    X_train_regression = X_train_clean.copy()
    X_test_regression = X_test_clean.copy()
    X_train_regression[config.TARGET_COLUMN] = y_train
    X_test_regression[config.TARGET_COLUMN] = y_test
    X_train_regression[config.PREDICTION_COLUMN] = tuned_rf_model.predict(X_train_clean)
    X_test_regression[config.PREDICTION_COLUMN] = tuned_rf_model.predict(X_test_clean)

    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = config.TARGET_COLUMN
    column_mapping.prediction = config.PREDICTION_COLUMN

    # Generate data drift report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=X_train_clean, current_data=X_test_clean, column_mapping=column_mapping)
    data_drift_report.save_html(config.MONITORING_REPORT_PATH)
    logger.info(f"Data drift report saved to {config.MONITORING_REPORT_PATH}")
    webbrowser.open_new_tab("file:///" + config.MONITORING_REPORT_PATH)

    # Generate regression report
    regression_report = Report(metrics=[RegressionPreset()])
    regression_report.run(reference_data=X_train_regression, current_data=X_test_regression, column_mapping=column_mapping)
    X_train_regression.to_csv("data/X_train_regression.csv", index=False)
    X_test_regression.to_csv("data/X_test_regression.csv", index=False)
    regression_report.save_html(config.REGRESSION_REPORT_PATH)
    logger.info(f"Regression report saved to {config.REGRESSION_REPORT_PATH}")
    webbrowser.open_new_tab("file:///" + config.REGRESSION_REPORT_PATH)

    # Modified Test Data Drift Report
    modified_data_drift_report = Report(metrics=[DataDriftPreset()])
    modified_data_drift_report.run(reference_data=X_train_clean, current_data=X_test_modified_clean, column_mapping=column_mapping)
    X_test_modified_clean.to_csv("data/X_test_modified_clean.csv", index=False)
    modified_data_drift_report.save_html("monitoring_report_modified_drift.html")
    logger.info("Modified Data Drift Report saved as monitoring_report_modified_drift.html")
    webbrowser.open_new_tab("file:///" + config.MODIFIED_MONITORING_REPORT_PATH)

    # Modified Test Data Regression Report
    X_test_modified_regression = X_test_modified_clean.copy()
    X_test_modified_regression[config.TARGET_COLUMN] = y_test.loc[X_test_modified_clean.index]
    X_test_modified_regression[config.PREDICTION_COLUMN] = tuned_rf_model.predict(X_test_modified_clean)
    modified_regression_report = Report(metrics=[RegressionPreset()])
    modified_regression_report.run(
        reference_data=X_train_regression,
        current_data=X_test_modified_regression,
        column_mapping=column_mapping
    )
    X_test_modified_regression.to_csv("data/X_test_modified_regression.csv", index=False)
    modified_regression_report.save_html("monitoring_report_modified_regression.html")
    logger.info("Modified Regression Report saved as monitoring_report_modified_regression.html")
    webbrowser.open_new_tab("file:///" + config.MODIFIED_REGRESSION_REPORT_PATH)
