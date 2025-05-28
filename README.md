
**🥁 Discover Model Lifecycle Automation and Orchestration 🎻**

In the previous unit, you implemented a full model lifecycle in the cloud:
1. Sourcing data from a data warehouse (Google BigQuery) and storing model weights on a bucket (GCS)
2. Launching a training task on a virtual machine (VM), including evaluating the model performance and making predictions

The _WagonCab_ team is really happy with your work and assigns you to a new mission: **ensure the validity of the trained model over time.**

As you might imagine, the fare amount of a taxi ride tends to change over time with the economy, and the model could be accurate right now but obsolete in the future.

---

🤯 After a quick brainstorming session with your team, you come up with a plan:
1. Implement a process to monitor the performance of the `Production` model over time
2. Implement an automated workflow to:
    - Fetch fresh data
    - Preprocess the fresh data
    - Evaluate the performance of the `Production` model on fresh data
    - Train a `Staging` model on the fresh data, _in parallel to the task above_
    - Compare `Production` vs `Staging` performance
    - Set a threshold for a model being good enough for production
    - If `Staging` better than both `Production` and the threshold, put it into production automatically
    - Otherwise where `Production` is better and still above the threshold leave it in production.
    - If neither meet the threshold *notify a human who will decide* whether or not to deploy the `Staging` model to `Production` and what others fixes are needed!
3. Deploy this workflow and wait for fresh data to come

<img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/wagoncab-workflow.png" alt="wagoncab_workflow" height=500>


# 1️⃣ Setup

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

## Install Requirements

**💻 Install version `0.0.10` of the `taxifare` package**

```bash
make reinstall_package
```

Notice we've added 3 new packages: `mlflow`, `prefect` and `psycopg2-binary`

**✅ Check your `taxifare` package version**

```bash
pip list | grep taxifare
# taxifare                  0.0.10
```

**💻 _copy_ the `.env.sample` file, _fill_ `.env`, _allow_ `direnv`**

We want to see some proper learning curve today: Let's set

```bash
DATA_SIZE='200k'
```

We'll move to `all` at the very end!

🏁 You are up and ready!

</details>


# 2️⃣ Performance Monitoring with MLflow

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>


🤯 You may remember that handling model versioning with local storage or GCS was quite shaky! We had to store weights as `model/{current_timestamp}.h5`, then and sort by most_recent etc...

🤗 Welcome **MLFlow**! It will:
- **store** both trained models weights and the results of our experiments (metrics, params) in the cloud!
- allow us to **tag** our models
- allow us to visually **monitor** the evolution of the performance of our models, experiment after experiment!

🔎 We have only slightly updated your taxifare package compared with unit 02:
- `interface/main.py`: `train()` and `evaluate()` are now decorated with `@mlflow_run`
- `ml_logic/registry.py`: defines `mlflow_run()` to automatically log TF training params!
- `interface/workflow.py`: (Keep for later) Entry point to run the _"re-train-if-performance-decreases"_ worflow)

## 2.1) Configure your Project for MLflow

#### MLflow Server

> The **WagonCab** tech team put in production an **MLflow** server located at [https://mlflow.lewagon.ai](https://mlflow.lewagon.ai), you will use in to track your experiments and store your trained models.

#### Environment Variables

**📝 Look at your `.env` file and discover 4 new variables to edit**:

- `MODEL_TARGET` (`local`, `gcs`, or now `mlflow`) which defines how the `taxifare` package should save the _outputs of the training_
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT`, which is the name of the experiment, should contain `taxifare_experiment_<user.github_nickname>`
- `MLFLOW_MODEL_NAME`, which is the name of your model, should contain `taxifare_<user.github_nickname>`


**🧪 Run the tests with `make test_mlflow_config`**

## 2.2) Update `taxifare` package to push your training results to MLflow

Now that your MLflow config is set up, you need to update your package so that the trained **model**, its **params** and its **performance metrics** are pushed to MLflow every time you run an new experiment, i.e. a new training.

#### a): Understand the setup

**❓ Which module of your `taxifare` package is responsible for saving the training outputs?**

<details>
  <summary markdown='span'>Answer</summary>

It is the role of the `taxifare.ml_logic.registry` module to save the trained model, its parameters, and its performance metrics, all thanks to the `save_model()`, `save_results()`, and `mlflow_run()` functions.

- `save_model` to save the models!
- `save_results` to save parameters and metrics
- `mlflow_run` is a decorator to start the runs and start the tf autologging
</details>

#### b): Do the first train run!

First, check if you already have a processed dataset available with the correct DATA_SIZE.

```bash
make show_sources_all
```

If not,
```bash
make run_preprocess
```

Now, lets do a first run of training to see what our decorator `@mlflow_run` creates for us thanks to  `mlflow.tensorflow.autolog()`

```bash
make run_train
```

☝️ This time, you should see the print "✅ mlflow_run autolog done"

**❓ Checkout what is logged on your experiment on https://mlflow.lewagon.ai/**
- Try to plot the your learning curve of `mae` and `val_mae` as function of epochs directly on the website UI !

#### c): Save the additional params manually on mlflow!

Beyond tensorflow specific training metrics, what else do you think we would want to log as well ?

<details>
<summary markdown='span'>💡 Solution</summary>

We can give more context:
  - Was this a train() run or evaluate()?
  - Data: How much data was used for this training run!
  - etc...

</details>


**❓ Edit `registry::save_results` so that when the model target is mlflow also save our additional params and metrics to mlflow.**

💡 Try Cmd-Shift-R for global symbol search - thank me later =)

<details>
<summary markdown='span'>🎁 Solution</summary>

For params
```python
if MODEL_TARGET == "mlflow":
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        mlflow.log_metrics(metrics)
    print("✅ Results saved on mlflow")
```

</details>


#### d): Save the model weights through MLflow, instead of manually on GCS


Let's have a look at `taxifare.ml_logic.registry::save_model`

- 🤯 Handling model versioning manually with local storage or GCS was quite shaky! We have to store weights as `model/{current_timestamp}.h5`, then and sort by most_recent etc...

- Let's use mlflow `mlflow.tensorflow.log_model` method to store model for us instead! MLflow will use its own AWS S3 bucket (equivalent to GCS) !

**💻 Complete the first step of the `save_model` function**

```python
# registry.py
def save_model():
    # [...]

    if MODEL_TARGET == "mlflow":
        # YOUR CODE HERE

```

<details>
<summary markdown='span'>🎁 Solution</summary>

```python
mlflow.tensorflow.log_model(model=model,
                        artifact_path="model",
                        registered_model_name=MLFLOW_MODEL_NAME
                        )
print("✅ Model saved to mlflow")
```



</details>

#### e): Automatic staging

Once a new model is trained, it should be moved into staging, and then compared with the model in production, if there is an improvement it should be moved into production instead!

❓ Add your code at the section in `interface.main` using `registry.mlflow_transition_model`:

```python
    def train():
    # [...]
        # The latest model should be moved to staging
        pass  # YOUR CODE HERE
```


Make a final training so as to save model to ML flow in "Staging" stage
🤔 Why staging? We never want to put in production a model without checking it's metric first!


```bash
make run_train
```
It should print something like this

- ✅ Model saved to mlflow
- ✅ Model <model_name> version 1 transitioned from None to Staging

Take a look at your model now on [https://mlflow.lewagon.ai](https://mlflow.lewagon.ai)

<details>
  <summary markdown='span'> 💡 You should get something like this </summary>

  <img style="width: 100%;" src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/mlflow_push_model.png' alt='mlflow_push_model'/>

</details>

## 2.3) Make a Prediction from your Model Saved in MLflow

"What's the point of storing my model on MLflow", you say? Well, for starters, MLflow allows you to very easily handle the lifecycle stage of the model (_None_, _Staging_ or _Production_) to synchronize the information across the team. And more importantly, it allows any application to load a trained model at any given stage to make a prediction.

First, notice that `make run_pred` requires a model in Production by default (not in Staging)

👉 Let's manually change your model from "Staging" to "Production" in mlflow graphical UI!

<img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/model_staging.png'>

**💻 Then, complete the `load_model` function in the `taxifare.ml_logic.registry` module**

- And try to run a prediction using `make run_pred`
- 💡 Hint:  Have a look at the [MLflow Python API for Tensorflow](https://mlflow.org/docs/2.1.1/python_api/mlflow.tensorflow.html) and find a function to retrieve your trained model.

<details>
  <summary markdown='span'>🎁 Solution</summary>

```python
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

try:
    model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
    model_uri = model_versions[0].source
    assert model_uri is not None
except:
    print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")
    return None

model = mlflow.tensorflow.load_model(model_uri=model_uri)

print("✅ model loaded from mlflow")
```
</details>

**💻 Check that you can also evaluate your production model by calling `make run_evaluate`**

✅ When you are all set, track your progress on Kitt with `make test_kitt`
🏁 Congrats! Your `taxifare` package is now persisting every aspect of your experiments on **MLflow**, and you have a _production-ready_ model!

</details>

# 3️⃣ Automate the workflow with Prefect

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>


Currently our retraining process relies on us running and comparing results manually. Lets build a prefect workflow to automate this process!

## 3.1) Prefect setup

- Checkout the `.env` make sure **PREFECT_FLOW_NAME** is filled.
- **Go to https://www.prefect.io/**, log in and then create a workspace!
- **Authenticate via the cli**:
```bash
prefect cloud login
```

📝 Edit your `.env` project configuration file:**
- `PREFECT_FLOW_NAME` should follow the `taxifare_lifecycle_<user.github_nickname>` convention
- `PREFECT_LOG_LEVEL` should say `WARNING`(more info [here](https://docs.prefect.io/core/concepts/logging.html)).

**🧪 Run the tests with `make test_prefect_config`**

Now by running `make run_workflow` on your prefect cloud dashboard you should see an empty flow run appear on your cloud dashboard.

## 3.2) Build your flow!

🎯 Now you need to work on completing `train_flow()` that you will find in `workflow.py`.

```python
@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the prefect workflow for the `taxifare` package. It should:
    - preprocess 1 month of new data, starting from EVALUATION_START_DATE
    - compute `old_mae` by evaluating current production model in this new month period
    - compute `new_mae` by re-training then evaluating current production model on this new month period
    - if new better than old, replace current production model by new one
    - if neither models are good enough, send a notification!
    """
```

#### a) Lets start by just the first two tasks to get `old_mae`

💡 Keep your code DRY: Our tasks simply call our various `main.py` entrypoints with argument of our choice! We could even get rid of them entirely and simply decorate our main entrypoints with @tasks. How elegant is that!

💡 Quick TLDR on how prefect works:

```python
# Define your tasks
@task
def task1():
  pass

@task
def task2():
  pass

# Define your workflow
@flow
def myworkflow():
    # Define the orchestration graph ("DAG")
    task1_future = task1.submit()
    task2_future = task2.submit(..., wait_for=[task1_future]) # <-- task2 starts only after task1

    # Compute your results as actual python object
    task1_result = task1_future.result()
    task2_result = task2_future.result()

    # Do something with the results (e.g. compare them)
    assert task1_result < task2_result

# Actually launch your workflow
myworkflow()
```

**🧪 Check your code with `make run_workflow`**

You should see two tasks run one after the other like below 👇

<img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/prefect-preprocess-evaluate.png" width=700>

#### b) Then try to add the last 2 tasks: `new_mae` computation and comparison for deployment to Prod !

💡 In the flow task `re_train` make sure to set split size to 0.2: as only using 0.02 won't be enough when we are getting new data for just one month.

**🧪 `make run_workflow` again: you should see a workflow like this in your prefect dashboard**

<img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/retrain-jan.png" width=700>

#### c) What if neither model are good enough ?

We have a scenario where **neither model is good enough** - in that case, we want to send messages to our team and say what has happened with a model depending on the retraining!

**❓ Implement the `notify` task**

<details>
  <summary markdown='span'>👇 Code to copy-paste</summary>


```python
# flow.py
import requests

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
```

</details>

✅ When you are all set, track your results on Kitt with `make test_kitt`

</details>


# 4️⃣ Play the full cycle: from Jan to Jun 2015

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

## 4.1) Let's get real with `all` data! 💪

First, train a full model **up to Jan 2015** on `all` data

```bash
DATA_SIZE='all'
MLFLOW_MODEL_NAME=taxifare_<user.github_nickname>_all
```

```bash
direnv reload
```

ONLY IF you haven't done it yet with `all` data in the past!
```bash
# make run_preprocess
```

Then
```bash
make run_train
```

✅ And manually put this first model manually to production.

**📆 We are now end January**

```bash
EVALUATION_START_DATE="2015-01-01"
```

Compare your current model with a newly trained one

```bash
make run_workflow
```

🎉 Our new model retrained on the data in Jan should performs slightly better so we have rolled it into production!

✅ Check your notification on https://chat.api.lewagon.com/<user.batch_slug>

**📆 We are now end February**

```bash
EVALUATION_START_DATE="2015-02-01"
direnv reload
make run_workflow
```

**📆 We are now end March**
```bash
EVALUATION_START_DATE="2015-03-01"
direnv reload
make run_workflow
```

**📆 We are now end April**
...


🏁🏁🏁🏁 Congrats on plugging the `taxifare` package into a fully automated workflow lifecycle!

</details>


# 5️⃣ Optionals

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

### Model Fine-Tuning

1. Before deciding which model version to put in production, try a couple of hyperparameters during the training phase, by wisely testing (grid-searching?) various values for `batch_size`,  `learning_rate` and `patience`.
2. In addition, after fine-tuning and deciding on a model, try to re-train using the whole new dataset of each month, and not just the "train_new".

### Prefect orion server

1. Try to replace prefect cloud with a locally run [prefect local UI](https://docs.prefect.io/ui/overview/#using-the-prefect-ui)
2. Add a work queue
3. Put this onto a vm to with a schedule to have a truly automated model lifecycle!

</details>
