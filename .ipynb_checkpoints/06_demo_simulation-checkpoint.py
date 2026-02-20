#!/usr/bin/env python
# coding: utf-8
# # Step 6 — Hybrid Recommender & Demo Simulation
# Loads all artifacts directly from S3 via boto3, runs the HybridRecommender.

"""
scripts/demo_recommend.py
S3 se artifacts load karta hai (boto3) + SageMaker endpoint se cluster fetch karta hai
+ MBA rules se products recommend karta hai.

Usage:
    python 06_demo_simulation.py --user_id U0007
    python 06_demo_simulation.py --user_id U0007 --top_n 10
"""

import boto3
import json
import io
import argparse
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
ENDPOINT_NAME = "hybrid-rec-endpoint"
REGION        = "ap-south-1"
BUCKET        = "sagemaker-product-ap-south-1"

S3_USER_CLUSTERS = "models/kmeans_artifacts/user_clusters.csv"
S3_MBA_RULES     = "models/mba_rules.json"

FEATURES = [
    'total_spend', 'purchase_frequency', 'avg_basket_value',
    'unique_products', 'category_diversity', 'price_sensitivity'
]

# ── Load artifacts from S3 ─────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)

def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    print(f"  [S3] Loading s3://{bucket}/{key} ...")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def load_json_from_s3(bucket: str, key: str) -> dict:
    print(f"  [S3] Loading s3://{bucket}/{key} ...")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

print("\nLoading artifacts from S3...")
df_users  = load_csv_from_s3(BUCKET, S3_USER_CLUSTERS)
mba_rules = load_json_from_s3(BUCKET, S3_MBA_RULES)
print("  ✅ Artifacts loaded.\n")


# ── Endpoint ───────────────────────────────────────────────────────────────────
def call_endpoint(feature_values: list) -> dict:
    """SageMaker endpoint invoke karo."""
    runtime  = boto3.client("sagemaker-runtime", region_name=REGION)
    payload  = {"features": feature_values}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# ── User features ──────────────────────────────────────────────────────────────
def get_user_features(user_id: str) -> list:
    """Persona ke basis pe representative feature values return karo."""
    row = df_users[df_users['user_id'] == user_id]
    if row.empty:
        raise ValueError(f"user_id '{user_id}' not found in user_clusters.csv")

    persona = row['persona'].iloc[0]
    persona_features = {
        "Champion":       [2500.0, 25.0, 100.0, 20.0, 5.0, 0.3],
        "High-Spender":   [2000.0, 10.0, 200.0, 15.0, 4.0, 0.2],
        "Frequent-Buyer": [800.0,  30.0,  26.0, 12.0, 3.0, 0.5],
        "Occasional":     [200.0,   5.0,  40.0,  5.0, 2.0, 0.8],
    }
    return persona_features.get(persona, [500.0, 12.0, 41.7, 8.0, 3.0, 0.6])


# ── Recommendations ────────────────────────────────────────────────────────────
def get_recommendations(cluster_id: int, persona: str, top_n: int = 5) -> list:
    """MBA rules se top N products recommend karo persona ke basis pe."""
    persona_weights = {
        "Champion":       {"confidence": 0.6, "lift": 0.4},
        "High-Spender":   {"confidence": 0.5, "lift": 0.5},
        "Frequent-Buyer": {"confidence": 0.7, "lift": 0.3},
        "Occasional":     {"confidence": 0.4, "lift": 0.6},
    }
    weights = persona_weights.get(persona, {"confidence": 0.5, "lift": 0.5})

    scored = {}
    for antecedent, rules in mba_rules.items():
        for rule in rules:
            product  = rule['product']
            weighted = rule['confidence'] * weights['confidence'] + rule['lift'] * weights['lift']
            if product not in scored or weighted > scored[product]['score']:
                scored[product] = {
                    "product":    product,
                    "score":      round(weighted, 4),
                    "confidence": round(rule['confidence'], 4),
                    "lift":       round(rule['lift'], 4),
                }

    return sorted(scored.values(), key=lambda x: x['score'], reverse=True)[:top_n]


# ── Demo ───────────────────────────────────────────────────────────────────────
def demo(user_id: str, top_n: int = 5):
    print(f"\n{'='*55}")
    print(f"  PRODUCT RECOMMENDATION DEMO")
    print(f"{'='*55}")

    # Step 1: User info
    row = df_users[df_users['user_id'] == user_id]
    if row.empty:
        print(f"  ❌ user_id '{user_id}' not found in user_clusters.csv")
        return

    # Step 2: Feature values
    feature_values = get_user_features(user_id)
    print(f"\n  [STEP 1] User Info")
    print(f"  User ID       : {user_id}")
    print(f"  Features sent : {feature_values}")

    # Step 3: Call endpoint
    print(f"\n  [STEP 2] Calling SageMaker Endpoint...")
    endpoint_result = call_endpoint(feature_values)
    cluster_id = endpoint_result['cluster_id']
    persona    = endpoint_result['persona']
    print(f"  ✅ Endpoint Response:")
    print(f"     Cluster ID  : {cluster_id}")
    print(f"     Persona     : {persona}")
    if "centroid_distance" in endpoint_result:
        print(f"     Distance    : {endpoint_result['centroid_distance']}")

    # Step 4: Recommendations
    print(f"\n  [STEP 3] Generating Recommendations (MBA Rules)...")
    recommendations = get_recommendations(cluster_id, persona, top_n)

    print(f"\n{'='*55}")
    print(f"  TOP {top_n} RECOMMENDED PRODUCTS FOR {user_id} ({persona})")
    print(f"{'='*55}")
    print(f"  {'#':<4} {'Product':<10} {'Confidence':<14} {'Lift':<10} {'Score'}")
    print(f"  {'-'*50}")
    for i, p in enumerate(recommendations, 1):
        print(f"  {i:<4} {p['product']:<10} {p['confidence']:<14} {p['lift']:<10} {p['score']}")
    print(f"{'='*55}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=True, help="e.g. U0007")
    parser.add_argument("--top_n",   type=int, default=5,     help="Kitne products chahiye")
    args = parser.parse_args()
    demo(args.user_id, args.top_n)