#!/usr/bin/env python
# coding: utf-8
# # Step 2 — Feature Engineering (SageMaker Processing Job)
# Runs `scripts/preprocessing.py` on AWS managed infrastructure.

import sagemaker, boto3
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

sess   = sagemaker.Session()
role   = sagemaker.get_execution_role()
BUCKET = 'sagemaker-product-ap-south-1'
print('Role:', role)

sklearn_processor = SKLearnProcessor(
    framework_version='1.0-1',
    role=role,
    instance_type='ml.t3.medium',
    instance_count=1,
    base_job_name='hybrid-rec-preprocessing'
)

sklearn_processor.run(
    code='./scripts/preprocessing.py',
    inputs=[
        ProcessingInput(
            source=f's3://{BUCKET}/data/raw/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=f's3://{BUCKET}/data/processed/'
        )
    ],
    wait=True,
    logs=True
)

print('Processing complete! Output at:', f's3://{BUCKET}/data/processed/')