import os

import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from taxifare.interface.main import evaluate, preprocess, train
from taxifare.ml_logic.registry import mlflow_transition_model
from taxifare.params import *

@task
def preprocess_new_data(min_date: str, max_date: str):
    pass  # YOUR CODE HERE

@task
def evaluate_production_model(min_date: str, max_date: str):
    pass  # YOUR CODE HERE

@task
def re_train(min_date: str, max_date: str, split_ratio: str):
    pass  # YOUR CODE HERE

@task
def transition_model(current_stage: str, new_stage: str):
    pass  # YOUR CODE HERE


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """

    min_date = EVALUATION_START_DATE
    max_date = str(datetime.strptime(min_date, "%Y-%m-%d") + relativedelta(months=1)).split()[0]

    pass  # YOUR CODE HERE

if __name__ == "__main__":
    train_flow()
