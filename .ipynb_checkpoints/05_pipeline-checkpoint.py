#!/usr/bin/env python
# coding: utf-8

# # Step 5 — SageMaker Pipeline (Full Orchestration)
# Chains preprocessing + KMeans training into one repeatable pipeline.

# In[ ]:
#!/usr/bin/env python
# coding: utf-8
# # Step 5 — SageMaker Pipeline (Full Orchestration)
# Chains preprocessing + KMeans training into one repeatable pipeline.
#!/usr/bin/env python
# coding: utf-8
# # Step 5 — SageMaker Pipeline (Full Orchestration)
# Chains preprocessing + KMeans training into one repeatable pipeline.

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
# ml.t3.medium IS supported for processing jobs in ap-south-1
# (confirmed working in step 02 — only training jobs reject t3)
processor = SKLearnProcessor(
    framework_version='1.0-1',
    role=role,
    instance_type='ml.t3.medium',      # <-- t3 is fine for processing jobs
    instance_count=1,
    sagemaker_session=pipeline_sess
)

preprocessing_step = ProcessingStep(
    name='FeatureEngineering',
    processor=processor,
    code='../scripts/preprocessing.py',
    inputs=[
        ProcessingInput(
            source=f's3://{BUCKET}/data/raw/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='processed',   # explicit name so training step can reference it
            source='/opt/ml/processing/output',
            destination=f's3://{BUCKET}/data/processed/'
        )
    ]
)

# ── Step B: KMeans Training ───────────────────────────────────────────────────
# ml.m5.large required for training jobs in ap-south-1 (t3 not supported for training)
estimator = SKLearn(
    entry_point='train_kmeans.py',
    source_dir='../scripts/',
    framework_version='1.0-1',
    role=role,
    instance_type='ml.m5.large',       # <-- m5 required for training jobs
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

# ── Assemble and run pipeline ─────────────────────────────────────────────────
pipeline = Pipeline(
    name='HybridRecPipeline',
    steps=[preprocessing_step, training_step],
    sagemaker_session=pipeline_sess
)

pipeline.upsert(role_arn=role)
print('Pipeline registered.')

execution = pipeline.start()
print(f'Execution started: {execution.arn}')

# Poll with per-step status so failures are visible immediately
sm = boto3.client('sagemaker', region_name='ap-south-1')
print('Waiting for pipeline to complete...')

while True:
    response = sm.describe_pipeline_execution(PipelineExecutionArn=execution.arn)
    status   = response['PipelineExecutionStatus']
    print(f'  Status: {status}')
    if status in ('Succeeded', 'Failed', 'Stopped'):
        break
    time.sleep(30)

# Print per-step results
print('\n── Step Results ──')
steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution.arn)
for step in steps['PipelineExecutionSteps']:
    print(f"  {step['StepName']:25s} → {step['StepStatus']}")
    if step['StepStatus'] == 'Failed':
        print(f"    Reason: {step.get('FailureReason', 'No reason provided')}")

if status == 'Succeeded':
    print('\nPipeline complete!')
else:
    raise RuntimeError(f'Pipeline failed with status: {status}. See step reasons above.')