import os
import logging
import pandas as pd
import azureml.core
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Dataset
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.runconfig import (
    CondaDependencies,
    RunConfiguration,
)
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import (
    Pipeline,
    PipelineParameter,
    PipelineData,
    TrainingOutput,
    Schedule,
    ScheduleRecurrence,
)
from azureml.pipeline.steps import PythonScriptStep, AutoMLStep
from azureml.train.automl import AutoMLConfig


print("This notebook was created using version 1.6.0 of the Azure ML SDK")
print(
    "You are currently using version",
    azureml.core.VERSION,
    "of the Azure ML SDK",
)

ws = Workspace(
    subscription_id="45b59352-da58-4e3a-beab-1a4518951e4e",
    resource_group="kk6gpv-rg",
    workspace_name="kk6gpv-aml",
    auth=ServicePrincipalAuthentication(
        tenant_id=os.environ["tenant_id"],
        service_principal_id=os.environ["sp_id"],
        service_principal_password=os.environ["sp_password"],
    ),
)
dstor = ws.get_default_datastore()


# cancel all pipeline schedules
print("Scheduled pipelines before:")
scheds = Schedule.list(ws)
print(scheds)
for sched in scheds:
    sched.disable()
    print(sched.id)

print("Scheduled pipelines after:")
scheds = Schedule.list(ws)
print(scheds)

# Choose a name for the run history container in the workspace.
experiment_name = "retrain-noaaweather"
experiment = Experiment(ws, experiment_name)

output = {}
output["Subscription ID"] = ws.subscription_id
output["Workspace"] = ws.name
output["Resource Group"] = ws.resource_group
output["Location"] = ws.location
output["Run History Name"] = experiment_name
# pd.set_option("display.max_colwidth", -1)
outputDf = pd.DataFrame(data=output, index=[""])
outputDf.T

# Choose a name for your CPU cluster
amlcompute_cluster_name = "cont-cluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2", max_nodes=4
    )
    compute_target = ComputeTarget.create(
        ws, amlcompute_cluster_name, compute_config
    )

compute_target.wait_for_completion(show_output=True)

# create a new RunConfig object
conda_run_config = RunConfiguration(framework="python")

# Set compute target to AmlCompute
conda_run_config.target = compute_target

conda_run_config.environment.docker.enabled = True
conda_run_config.environment.docker.base_image = (
    azureml.core.runconfig.DEFAULT_CPU_IMAGE
)

cd = CondaDependencies.create(
    pip_packages=[
        "azureml-defaults",
        "azureml-sdk[automl]",
        "applicationinsights",
        "azureml-opendatasets",
    ],
    conda_packages=["numpy==1.16.2"],
    pin_sdk_version=False,
)
# cd.add_pip_package('azureml-explain-model')
conda_run_config.environment.python.conda_dependencies = cd

print("run config is ready")

# The name and target column of the Dataset to create
dataset = "NOAA-Weather-DS4"
target_column_name = "temperature"

ds_name = PipelineParameter(name="ds_name", default_value=dataset)
upload_data_step = PythonScriptStep(
    script_name="upload_weather_data.py",
    allow_reuse=False,
    name="upload_weather_data",
    arguments=["--ds_name", ds_name],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

data_pipeline = Pipeline(
    description="pipeline_with_uploaddata",
    workspace=ws,
    steps=[upload_data_step],
)

# data_pipeline_run = experiment.submit(
#     data_pipeline, pipeline_parameters={"ds_name": dataset}
# )
# data_pipeline_run.wait_for_completion(show_output=False)

pipeline_name = "DataIngestion-Pipeline-NOAAWeather"
published_pipeline = data_pipeline.publish(
    name=pipeline_name, description="Pipeline that updates NOAAWeather Dataset"
)
published_pipeline
schedule = Schedule.create(
    workspace=ws,
    name="RetrainingSchedule-DataIngestion",
    pipeline_parameters={"ds_name": dataset},
    pipeline_id=published_pipeline.id,
    experiment_name=experiment_name,
    datastore=dstor,
    wait_for_provisioning=True,
    recurrence=ScheduleRecurrence(frequency="Hour", interval=24,),
)

# The model name with which to register the trained model in the workspace.
model_name = PipelineParameter("model_name", default_value="noaaweatherds")
data_prep_step = PythonScriptStep(
    script_name="check_data.py",
    allow_reuse=False,
    name="check_data",
    arguments=["--ds_name", ds_name, "--model_name", model_name],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

automl_settings = {
    "iteration_timeout_minutes": 10,
    "experiment_timeout_hours": 0.25,
    "n_cross_validations": 3,
    "primary_metric": "r2_score",
    "max_concurrent_iterations": 3,
    "max_cores_per_iteration": -1,
    "verbosity": logging.INFO,
    "enable_early_stopping": True,
}
train_ds = Dataset.get_by_name(ws, dataset)
train_ds = train_ds.drop_columns(["partition_date"])
train_ds = train_ds.to_pandas_dataframe()
train_ds = train_ds.drop_duplicates(subset=["usaf", "datetime"])
local_path = "data/prepared.csv"
train_ds.to_csv(local_path)
datastore = ws.get_default_datastore()
train_ds = Dataset.Tabular.from_delimited_files(datastore.path(local_path))
automl_config = AutoMLConfig(
    task="forecasting",
    time_column_name="datetime",
    grain_column_names=["usaf"],
    featurization="auto",
    debug_log="automl_errors.log",
    path=".",
    compute_target=compute_target,
    training_data=train_ds,
    label_column_name=target_column_name,
    **automl_settings
)

metrics_output_name = "metrics_output"
metrics_data = PipelineData(
    name="metrics_data",
    datastore=dstor,
    pipeline_output_name=metrics_output_name,
    training_output=TrainingOutput(type="Metrics"),
)
best_model_output_name = "best_model_output"
model_data = PipelineData(
    name="model_data",
    datastore=dstor,
    pipeline_output_name=best_model_output_name,
    training_output=TrainingOutput(type="Model"),
)

automl_step = AutoMLStep(
    name="automl_module",
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=False,
)

register_model_step = PythonScriptStep(
    script_name="register_model.py",
    name="register_model",
    allow_reuse=False,
    arguments=[
        "--model_name",
        model_name,
        "--model_path",
        model_data,
        "--ds_name",
        ds_name,
    ],
    inputs=[model_data],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

training_pipeline = Pipeline(
    description="training_pipeline",
    workspace=ws,
    steps=[data_prep_step, automl_step, register_model_step],
)

# training_pipeline_run = experiment.submit(
#     training_pipeline,
#     pipeline_parameters={"ds_name": dataset, "model_name": "noaaweatherds"},
# )
# training_pipeline_run.wait_for_completion(show_output=False)

pipeline_name = "Retraining-Pipeline-NOAAWeather"
published_pipeline = training_pipeline.publish(
    name=pipeline_name, description="Pipeline that retrains AutoML model"
)
published_pipeline
schedule = Schedule.create(
    workspace=ws,
    name="RetrainingSchedule",
    pipeline_parameters={"ds_name": dataset, "model_name": "noaaweatherds"},
    pipeline_id=published_pipeline.id,
    experiment_name=experiment_name,
    datastore=dstor,
    wait_for_provisioning=True,
    polling_interval=5,
)

print("Scheduled pipelines final:")
scheds = Schedule.list(ws)
print(scheds)
