# Import Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

# Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Submission MSML - Car Price Regression")

# Load Dataset
df = pd.read_csv('used_car_price_dataset_extended_preprocessing.csv')

# Split Dataset
X = df.drop('price_usd', axis=1)
y = df['price_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

input_example = X_train[0:5]

with mlflow.start_run(run_name="GBR_standard"):
    # Parameter yang dicoba & log
    loss='squared_error'
    learning_rate=0.1
    max_depth=10
    random_state=42
    mlflow.autolog()

    # Train model
    model = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, 
                                      max_depth=max_depth, random_state=random_state)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    
    # Log hasil test model
    score_gbr = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    mae_gbr = mean_absolute_error(y_test, y_pred)
    mse_gbr = mean_squared_error(y_test, y_pred)
    r2_gbr = r2_score(y_test, y_pred)
    
    mlflow.log_metric("test_MAE_manual", mae_gbr)
    mlflow.log_metric("test_MSE_manual", mse_gbr)
    mlflow.log_metric("test_R²_manual", r2_gbr)
    mlflow.log_metric("test_RMSE_manual", np.sqrt(mse_gbr))
    mlflow.log_metric("test_score_manual", score_gbr)