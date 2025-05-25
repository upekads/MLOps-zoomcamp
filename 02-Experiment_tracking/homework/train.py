import os
import pickle
import click
import mlflow
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///C:/Users/gbm7594/Desktop/N/ML_Courses/MLOps_ZoomCamp/course/MLOps-zoomcamp/02-Experiment_tracking/mlflow.db")
mlflow.set_experiment("homework2_experiment")




def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():
        mlflow.set_tag("developer","upeka")
        mlflow.sklearn.autolog()
        #track dataset
        #mlflow.log_param("train-data-path","./data/green_tripdata_2021-01.parquet")
        #mlflow.log_param("vali-data-path","./data/green_tripdata_2021-02.parquet")
    

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        
        
        #mlflow.log_param("alpha", alpha)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        
        #mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':

    run_train()
