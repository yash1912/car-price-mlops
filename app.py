from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import utils
import config
import os
import pickle
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset 
from evidently import ColumnMapping
from typing import Optional

app = FastAPI()
X_train_regression = pd.read_csv("data/X_train_regression.csv")
X_train_clean = pd.read_csv("data/X_train_clean.csv")
X_test_regression = pd.read_csv("data/X_test_regression.csv")
# Load the pre-trained model
model = pickle.load(open(config.MODEL_PATH, 'rb'))

# Health check route
@app.get("/")
def read_root():
    return {"message": "API is up and running!"}

# Predict route for a single example
@app.post("/predict_single")
def predict_single(
    Year: int = Form(...),
    Present_Price: float = Form(...),
    Kms_Driven: int = Form(...),
    Owner: int = Form(...),
    Fuel_Type: str = Form(...),
    Seller_Type: str = Form(...),
    Transmission: str = Form(...)
):
    # Preprocess inputs
    Fuel_Type_Diesel = 0
    Fuel_Type_Petrol = 0
    if Fuel_Type == 'Petrol':
        Fuel_Type_Petrol = 1
    elif Fuel_Type == 'Diesel':
        Fuel_Type_Diesel = 1

    Year = 2020 - Year
    Seller_Type_Individual = 1 if Seller_Type == 'Individual' else 0
    Transmission_Mannual = 1 if Transmission == 'Manual' else 0

    # Prepare data for prediction
    input_data = [[
        Present_Price, Kms_Driven, Owner, Year,
        Fuel_Type_Diesel, Fuel_Type_Petrol,
        Seller_Type_Individual, Transmission_Mannual
    ]]
    
    # Predict
    prediction = model.predict(input_data)
    output = round(prediction[0], 2)

    return {"predicted_selling_price": output}

# Batch prediction and generate Evidently reports
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # Load the uploaded file
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    try:
        # Read the file into a DataFrame
        X_test = pd.read_csv(file_location)

        # Preprocess the data
        # def preprocess_batch_data(row):
        #     Fuel_Type_Diesel = 0
        #     Fuel_Type_Petrol = 0
        #     if row['Fuel_Type'] == 'Petrol':
        #         Fuel_Type_Petrol = 1
        #     elif row['Fuel_Type'] == 'Diesel':
        #         Fuel_Type_Diesel = 1

        #     Year = 2020 - row['Year']
        #     Seller_Type_Individual = 1 if row['Seller_Type'] == 'Individual' else 0
        #     Transmission_Mannual = 1 if row['Transmission'] == 'Manual' else 0

        #     return [
        #         row['Present_Price'], row['Kms_Driven'], row['Owner'], Year,
        #         Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual
        #     ]

        # # Apply preprocessing to each row
        # preprocessed_data = X_test.apply(preprocess_batch_data, axis=1, result_type='expand')
        # preprocessed_data.columns = [
        #     'Present_Price', 'Kms_Driven', 'Owner', 'Year',
        #     'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
        #     'Seller_Type_Individual', 'Transmission_Mannual'
        # ]
        preprocessed_data = X_test.copy()

        # Generate predictions
        preprocessed_data[config.PREDICTION_COLUMN] = model.predict(preprocessed_data)
        # Save the predictions as a new CSV
        predictions_file = config.PREDICTIONS_PATH
        preprocessed_data.to_csv(predictions_file, index=False)
        preprocessed_data[config.TARGET_COLUMN] = X_test_regression[config.TARGET_COLUMN]

        # Column mapping for Evidently
        column_mapping = ColumnMapping()
        column_mapping.target = config.TARGET_COLUMN
        column_mapping.prediction = config.PREDICTION_COLUMN

        # Generate Data Drift Report
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(reference_data=X_train_clean, current_data=preprocessed_data.drop(columns=[config.PREDICTION_COLUMN, config.TARGET_COLUMN], axis=1, errors='ignore'), column_mapping=column_mapping)
        data_drift_path = config.PREDICTED_DATA_DRIFT_REPORT_PATH
        data_drift_report.save_html(data_drift_path)

        # Generate Regression Report
        regression_report = Report(metrics=[RegressionPreset()])
        regression_report.run(reference_data=X_train_regression, current_data=preprocessed_data, column_mapping=column_mapping)
        regression_path = config.PREDICTED_REGRESSION_REPORT_PATH
        regression_report.save_html(regression_path)

        label_drift_report = Report(metrics=[TargetDriftPreset()])
        column_mapping = ColumnMapping()
        column_mapping.prediction = config.PREDICTION_COLUMN
        label_drift_report.run(reference_data=X_test_regression, current_data=preprocessed_data, column_mapping=column_mapping)
        label_drift_path = config.PREDICTED_LABEL_DRIFT_REPORT_PATH
        label_drift_report.save_html(label_drift_path)

        # return {
        #     "status": "success",
        #     "message": "Predictions and reports generated successfully.",
        #     "predictions_file": predictions_file,
        #     "data_drift_report": data_drift_path,
        #     "regression_report": regression_path
        # }
        return HTMLResponse(content=f"""
        <html>
            <head><title>Reports</title></head>
            <body>
                <h1>Generated Reports</h1>
                <a href="/view_report?report_type=data_drift" target="_blank">View Data Drift Report</a><br>
                <a href="/view_report?report_type=regression" target="_blank">View Regression Report</a>
                <a href="/view_report?report_type=label_drift" target="_blank">View Prediction Report</a>
            </body>
        </html>
        """)
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_location):
            os.remove(file_location)
# Endpoint to serve the generated HTML report
@app.get("/view_report")
def view_report(report_type: str):
    """
    View the generated report based on the report type.

    Parameters:
    - report_type: Type of report ("data_drift" or "regression").

    Returns:
    - HTML content of the selected report.
    """
    if report_type == "data_drift":
        report_path = config.PREDICTED_DATA_DRIFT_REPORT_PATH
    elif report_type == "regression":
        report_path = config.PREDICTED_REGRESSION_REPORT_PATH
    elif report_type == "label_drift":
        report_path = config.PREDICTED_LABEL_DRIFT_REPORT_PATH
    else:
        return JSONResponse(content={"error": "Invalid report type."}, status_code=400)

    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return JSONResponse(content={"error": "Report file not found."}, status_code=404)
# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
