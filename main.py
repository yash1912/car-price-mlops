import mlflow
import utils
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset, RegressionPerformancePreset
import logging, sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler([logging.StreamHandler(sys.stdout), logging.FileHandler('app.log')])


df = pd.read_csv("data/cars data.csv") 

mlflow.set_experiment("Car Price Prediction")

with mlflow.start_run():
    # Preprocess the data
    df = utils.preprocess_data(df)
    
    # Push the processed data to LakeFS
    utils.push_data_to_lakefs(df=df, data_name="cars_data", repo_name="mlops-final", version="v1")
    
    # Separate features and target variable
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = utils.get_data_splits(X, y)
    
    # Tune the Random Forest model and get the best parameters
    tuned_rf_model, best_params = utils.tune_random_forest_model(X_train, y_train)
    
    # Evaluate the model on the test set
    rmse, mse, mae = utils.evaluate_model(tuned_rf_model, X_test, y_test)
    
    # Log the tuned model to MLflow
    mlflow.sklearn.log_model(tuned_rf_model, "random_forest_model")
    
    # Log evaluation metrics to MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    
    # Log the best parameters found during tuning
    mlflow.log_params(best_params)

report = Report(
    metrics=[DataDriftPreset(), RegressionPerformancePreset()]
)
report.run(reference_data=X_train, current_data=X_test)
report.save_html("monitoring_report.html")
logging("Monitoring report saved as monitoring_report.html")