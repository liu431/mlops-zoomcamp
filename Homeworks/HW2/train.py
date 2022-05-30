import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    print(os.getcwd())

    mlflow.set_tracking_uri("sqlite:///mlflow-HW2.db")
    mlflow.set_experiment("nyc-taxi-experiment-HW2")
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        with open('artifacts/rfg.bin', 'wb') as f_out:
            pickle.dump(rf, f_out)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        print(rmse)
        mlflow.log_artifact(local_path="artifacts/rfg.bin", artifact_path="models_pickle")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    run(args.data_path)
