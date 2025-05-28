import os

import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from taxifare.interface.main import evaluate, preprocess, train
from taxifare.ml_logic.registry import mlflow_transition_model
from taxifare.params import *
import requests

@task
def preprocess_new_data(min_date: str, max_date: str):
    preprocess(min_date=min_date, max_date=max_date)

@task
def evaluate_production_model(min_date: str, max_date: str):
    return evaluate(min_date=min_date, max_date=max_date, stage="Production")

@task
def re_train(min_date: str, max_date: str, split_ratio: str):
    return train(min_date=min_date, max_date=max_date, split_ratio=split_ratio)

@task
def transition_model(current_stage: str, new_stage: str):
    return mlflow_transition_model(current_stage=current_stage, new_stage=new_stage)


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

    # Step 1: Preprocess new data
    _ = preprocess_new_data.submit(min_date, max_date)

    # Step 2: Evaluate current production model
    old_mae_future = evaluate_production_model.submit(min_date, max_date)

    # Step 3: Train a new model and evaluate it
    new_mae_future = re_train.submit(min_date, max_date, split_ratio=0.8)

    # Step 4: Wait for both MAEs and compare
    old_mae = old_mae_future.result()
    new_mae = new_mae_future.result()

    print(f"📉 Old MAE: {old_mae} | 📈 New MAE: {new_mae}")

    if new_mae < old_mae:
        print("✅ New model is better. Promoting to Production...")
        transition_model.submit(current_stage="None", new_stage="Production")
    elif old_mae > 10 and new_mae > 10:
        print("🚨 Warning: Both models are underperforming!")
        # You could trigger a notification task here
    else:
        print("⚖️ New model not better. Keeping current production model.")

@task
def notify(old_mae, new_mae):
    """
    Notify about the performance
    """
    base_url = 'https://chat.api.lewagon.com'
    channel = 'YOUR_BATCH_NUMBER' # Change to your batch number
    url = f"{base_url}/{channel}/messages"
    author = 'YOUR_GITHUB_NICKNAME' # Change this to your github nickname
    if new_mae < old_mae and new_mae < 2.5:
        content = f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"
    elif old_mae < 2.5:
        content = f"✅ Old model still good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    else:
        content = f"🚨 No model good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    data = dict(author=author, content=content)
    response = requests.post(url, data=data)
    response.raise_for_status()

if __name__ == "__main__":
    train_flow()
