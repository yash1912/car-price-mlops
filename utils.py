import pandas as pd
from lakefs_spec import LakeFSFileSystem
from lakefs.client import LakeFSClient
import os
from dotenv import load_dotenv
import lakefs
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import pickle
import numpy as np
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Change this line:
# logger.addHandler([logging.StreamHandler(sys.stdout), logging.FileHandler('app.log')]) 
logger.addHandler(logging.StreamHandler(sys.stdout)) 
logger.addHandler(logging.FileHandler('app.log'))

def push_data_to_lakefs(df: pd.DataFrame, data_name: str, repo_name: str = "datarepo", version: str = "v1") -> None:
    """
    Pushes a DataFrame to a lakeFS repository.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be pushed to lakeFS.
    data_name : str
        The name to be used for the data file in lakeFS.
    repo_name : str, optional
        The name of the lakeFS repository, by default "datarepo".
    version : str, optional
        The version tag for the data, by default "v1".
    """
    # Retrieve lakeFS credentials from environment variables
    HOST_NAME, USERNAME, PASSWORD = os.getenv('HOST_NAME'), os.getenv('USERNAME'), os.getenv('PASSWORD')

    # Initialize the lakeFS file system with credentials
    fs = LakeFSFileSystem(host=HOST_NAME, username=USERNAME, password=PASSWORD)

    # Create a new repository in lakeFS
    try:
        repo = lakefs.Repository(repo_name, client=fs.client).create(storage_namespace=f"local://{repo_name}")
    except:
        repo = lakefs.Repository(repo_name, client=fs.client)
        logger.info(f"Repository {repo_name} already exists")

    logger.info(f"Created lakeFS repository: {repo}")

    # Start a transaction to add data to the main branch of the repository
    with fs.transaction(repo_name, 'main') as tx:
        filename = os.path.basename(data_name)
        # Save the DataFrame to a CSV file in the lakeFS repository
        df.to_csv(f'lakefs://{repo_name}/{tx.branch.id}/{data_name}.csv', index=False, storage_options=fs.storage_options)

        # Commit the transaction with a message
        tx.commit(message=f"Added data {version}")

    logger.info("Pushed data to lakeFS")

def preprocess_data(df: pd.DataFrame, in_test_mode: bool = False) -> pd.DataFrame:
    """
    Preprocess the data by dropping null values, 
    selecting the columns to use for the model, 
    calculating the number of years since the car was bought, 
    and one-hot encoding categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """

    # Drop rows with missing values
    if df.isnull().any().any():
        logger.info("Dropping null values")
        df = df.dropna()

    # Select the columns to use
    columns_to_use = ['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']
    df = df[columns_to_use]

    # Calculate the number of years since the car was bought
    df['num_years'] = 2020 - df['Year']
    df.drop(['Year'], axis=1, inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

def get_data_splits(X, y = None, test_size=0.2, random_state=42): #added random state for reproducibility
    if y is None:
        df_train, df_test = train_test_split(X, test_size=test_size, random_state=random_state, shuffle=True)
        return df_train, df_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
        return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, **kwargs):
    """
    Train a RandomForestRegressor model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series
        Target values for training.
    **kwargs : dict
        Additional arguments for RandomForestRegressor.

    Returns
    -------
    model : RandomForestRegressor
        Trained RandomForestRegressor model.
    """
    # Initialize the RandomForestRegressor with given hyperparameters
    model = RandomForestRegressor(**kwargs)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Return the trained model
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a trained model on the test data.

    Parameters
    ----------
    model : sklearn.Model
        Trained model to evaluate.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True values for test data.

    Returns
    -------
    tuple
        A tuple containing RMSE, MSE, and MAE of the model on the test data.
    """

    # Predict the target values using the test features
    y_pred = model.predict(X_test)
    # Calculate Root Mean Squared Error
    rmse = root_mean_squared_error(y_test, y_pred)
    # Calculate Mean Squared Error
    mse = np.square(root_mean_squared_error(y_test, y_pred))
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    return rmse, mse, mae

def tune_random_forest_model(X_train, y_train):
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2'],  # Corrected max_features
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'friedman_mse'],
        'random_state': [42]
                        }
    model = RandomForestRegressor()
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, scoring='neg_mean_squared_error', n_iter=10, cv=5, random_state=42, n_jobs=8)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def save_model(model, model_path: str = 'model.pkl'):
    pickle.dump(model, open(model_path, 'wb'))

def load_model(model_path: str = 'model.pkl'):
    return pickle.load(open(model_path, 'rb'))

def modify_test_data(X_test, num_changes=2):

    ##NOTE: modify this
    """Modifies specified number of rows in test data by swapping values between 'Kms_Driven' and 'Present_Price'."""
    X_test_modified = X_test.copy() # Important: Create a copy to avoid modifying the original DataFrame

    X_test_modified['Kms_Driven'] = X_test_modified['Present_Price']
    X_test_modified['Present_Price'] = X_test['Kms_Driven'].copy()
    
    return X_test_modified