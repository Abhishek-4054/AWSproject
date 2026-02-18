#!/usr/bin/env python
# coding: utf-8
# # Step 4 — KMeans User Segmentation (SageMaker Training Job)
# Trains KMeans on user behavior features. Evaluates with Silhouette Score and Davies-Bouldin Score.

import sagemaker, boto3, tarfile, json, os
import pandas as pd
import matplotlib.pyplot as plt

sess   = sagemaker.Session()
role   = sagemaker.get_execution_role()
BUCKET = 'sagemaker-product-ap-south-1'

# ── Training Job ──────────────────────────────────────────────────────────────
# ml.t3.medium is NOT available for training jobs in ap-south-1 → use ml.m5.large
from sagemaker.sklearn.estimator import SKLearn

kmeans_estimator = SKLearn(
    entry_point='train_kmeans.py',
    source_dir='../scripts/',
    framework_version='1.0-1',
    instance_type='ml.m5.large',
    instance_count=1,
    role=role,
    hyperparameters={
        'n_clusters':   4,
        'random_state': 42,
        'max_iter':     300
    },
    base_job_name='hybrid-rec-kmeans'
)

kmeans_estimator.fit(
    {'train': f's3://{BUCKET}/data/processed/'},
    wait=True
)

print('Training complete!')
print('Model artifacts:', kmeans_estimator.model_data)

# ── Download & extract model artifacts ───────────────────────────────────────
s3 = boto3.client('s3')
model_uri = kmeans_estimator.model_data
parts     = model_uri.replace('s3://', '').split('/', 1)

os.makedirs('../models/kmeans_artifacts', exist_ok=True)
s3.download_file(parts[0], parts[1], 'model.tar.gz')

with tarfile.open('model.tar.gz', 'r:gz') as tar:
    tar.extractall('../models/kmeans_artifacts/', filter='data')

# ── Display evaluation metrics ────────────────────────────────────────────────
with open('../models/kmeans_artifacts/metrics.json') as f:
    metrics = json.load(f)

print('\n=== Clustering Evaluation Metrics ===')
print(f"Silhouette Score     : {metrics['silhouette_score']}  (higher is better, max=1.0)")
print(f"Davies-Bouldin Score : {metrics['davies_bouldin_score']}  (lower is better)")
print(f"Inertia              : {metrics['inertia']}")
print(f"\nPersona Map: {metrics['persona_map']}")

# ── Plot cluster profiles ─────────────────────────────────────────────────────
# cluster_profiles has clusters as rows, features as columns.
# Transpose → columns become cluster IDs so profiles[cluster][feature] works.
profiles = pd.DataFrame(metrics['cluster_profiles']).T

features = [
    'total_spend', 'purchase_frequency', 'avg_basket_value',
    'unique_products', 'category_diversity', 'price_sensitivity'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
colors = ['#2E86C1', '#1E8449', '#D35400', '#8E44AD']

for ax, feat in zip(axes.flatten(), features):
    clusters = list(profiles.columns)
    vals     = [float(profiles[c][feat]) for c in clusters]
    bars     = ax.bar(clusters, vals, color=colors[:len(clusters)])
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('Cluster')
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2., bar.get_height(),
            f'{v:.2f}', ha='center', va='bottom', fontsize=8
        )

plt.suptitle('KMeans Cluster Profiles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.show()
print('Cluster profile plot saved to ../models/cluster_profiles.png')