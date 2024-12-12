# 🚗 **Resale Price Prediction of Used Cars**

## **Overview**
This project aims to build a machine learning pipeline to predict the resale price of used cars based on features such as manufacturing year, present market price, kilometers driven, and ownership history. The solution was developed using AutoML to refine the selection of algorithms and optimize model performance.

---

## **Problem Statement**
> *How can we accurately predict the resale price of a used car based on its features such as manufacturing year, present market price, kilometers driven, and ownership history?*

---

## **Goal**
- Build a machine learning pipeline leveraging AutoML to predict the **Selling Price** of used cars.
- Deploy the model with monitoring to ensure reliability and evaluate the impact of feature modifications on predictions.

---

## **Features**
The dataset includes the following key features:
- **Manufacturing Year**: Year the car was manufactured.
- **Present Market Price**: Current market value of the car.
- **Kilometers Driven**: Total distance the car has been driven.
- **Ownership History**: Details of previous ownership.
- **Additional Features**: Fuel type, transmission, number of owners, etc.

---

## **Technologies Used**
- **AutoML**: Automated selection and tuning of machine learning models.
- **Python**: For data preprocessing and additional analysis.
- **H2O.ai**: For model training and evaluation.
- **Visualization**: Power BI and Matplotlib for result presentation.

---

## **Results**
Top-performing models from AutoML:
1. **Model ID**: `DRF_grid_1_AutoML_1_20241212_54440_model_215`
   - **RMSE**: 1.28412
   - **MAE**: 0.662443
   - **Algorithm**: Distributed Random Forest (DRF).
2. **Model ID**: `XGBoost_grid_1_AutoML_1_20241212_54440_model_484`
   - **RMSE**: 1.37046
   - **MAE**: 0.685618
   - **Algorithm**: XGBoost.
