#!/usr/bin/env python
# coding: utf-8
# # Step 5 — SageMaker Pipeline (Full Orchestration)
# Chains preprocessing + KMeans training + MBA rules into one repeatable pipeline.

import sagemaker, boto3, time
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput

role          = sagemaker.get_execution_role()
pipeline_sess = PipelineSession()
BUCKET        = 'sagemaker-product-ap-south-1'

# ── Step A: Preprocessing ─────────────────────────────────────────────────────
processor = SKLearnProcessor(
    framework_version='1.0-1',
    role=role,
    instance_type='ml.t3.medium',
    instance_count=1,
    sagemaker_session=pipeline_sess
)

preprocessing_step = ProcessingStep(
    name='FeatureEngineering',
    processor=processor,
    code='./scripts/preprocessing.py',
    inputs=[
        ProcessingInput(
            source=f's3://{BUCKET}/data/raw/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='processed',
            source='/opt/ml/processing/output',
            destination=f's3://{BUCKET}/data/processed/'
        )
    ]
)

# ── Step B: KMeans Training ───────────────────────────────────────────────────
estimator = SKLearn(
    entry_point='train_kmeans.py',
    source_dir='./scripts/',
    framework_version='1.0-1',
    role=role,
    instance_type='ml.m5.large',
    instance_count=1,
    hyperparameters={
        'n_clusters':   4,
        'random_state': 42,
        'max_iter':     300
    },
    sagemaker_session=pipeline_sess
)

training_step = TrainingStep(
    name='KMeansTraining',
    estimator=estimator,
    inputs={
        'train': preprocessing_step.properties
                 .ProcessingOutputConfig.Outputs['processed'].S3Output.S3Uri
    },
    depends_on=[preprocessing_step]
)

# ── Step C: MBA Rules Generation ──────────────────────────────────────────────
# Runs after KMeans so user_clusters.csv is already in S3
mba_processor = SKLearnProcessor(
    framework_version='1.0-1',
    role=role,
    instance_type='ml.t3.medium',
    instance_count=1,
    sagemaker_session=pipeline_sess,
    base_job_name='hybrid-rec-mba'
)

mba_step = ProcessingStep(
    name='MBArulesGeneration',
    processor=mba_processor,
    code='./scripts/03_mba_rules.py',
    inputs=[
        ProcessingInput(
            source=f's3://{BUCKET}/data/raw/',
            destination='/opt/ml/processing/input/raw'
        ),
        ProcessingInput(
            source=f's3://{BUCKET}/models/kmeans_artifacts/',
            destination='/opt/ml/processing/input/models'
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name='mba_output',
            source='/opt/ml/processing/output',
            destination=f's3://{BUCKET}/models/'
        )
    ],
    depends_on=[training_step]
)

# ── Assemble and run pipeline ─────────────────────────────────────────────────
pipeline = Pipeline(
    name='HybridRecPipeline',
    steps=[preprocessing_step, training_step, mba_step],
    sagemaker_session=pipeline_sess
)

pipeline.upsert(role_arn=role)
print('Pipeline registered.')

execution = pipeline.start()
print(f'Execution started: {execution.arn}')

sm = boto3.client('sagemaker', region_name='ap-south-1')
print('Waiting for pipeline to complete...')

while True:
    response = sm.describe_pipeline_execution(PipelineExecutionArn=execution.arn)
    status   = response['PipelineExecutionStatus']
    print(f'  Status: {status}')
    if status in ('Succeeded', 'Failed', 'Stopped'):
        break
    time.sleep(30)

print('\n── Step Results ──')
steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution.arn)
for step in steps['PipelineExecutionSteps']:
    print(f"  {step['StepName']:25s} → {step['StepStatus']}")
    if step['StepStatus'] == 'Failed':
        print(f"    Reason: {step.get('FailureReason', 'No reason provided')}")

if status == 'Succeeded':
    print('\nPipeline complete!')
    print('Next: python 06a_deploy_endpoint.py && python 06_demo_simulation.py --user_id U0001')
else:
    raise RuntimeError(f'Pipeline failed: {status}')