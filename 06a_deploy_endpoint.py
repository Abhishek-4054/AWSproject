#!/usr/bin/env python
import boto3, json, io, argparse
import pandas as pd

ENDPOINT_NAME    = "hybrid-rec-endpoint"
REGION           = "ap-south-1"
BUCKET           = "sagemaker-product-ap-south-1"
S3_USER_CLUSTERS = "models/kmeans_artifacts/user_clusters.csv"
S3_KMEANS_INPUT  = "data/processed/kmeans_input.csv"
S3_MBA_RULES     = "models/mba_rules.json"
FEATURES = ['total_spend','purchase_frequency','avg_basket_value',
            'unique_products','category_diversity','price_sensitivity']

s3 = boto3.client("s3", region_name=REGION)
def load_csv(key):
    print(f"  [S3] {key}")
    return pd.read_csv(io.BytesIO(s3.get_object(Bucket=BUCKET,Key=key)['Body'].read()))
def load_json(key):
    print(f"  [S3] {key}")
    return json.loads(s3.get_object(Bucket=BUCKET,Key=key)['Body'].read().decode('utf-8'))

print("\nLoading from S3...")
df_clusters = load_csv(S3_USER_CLUSTERS)
df_features = load_csv(S3_KMEANS_INPUT)
mba_rules   = load_json(S3_MBA_RULES)
print(f"  clusters: {len(df_clusters)} | features: {len(df_features)} | personas: {list(mba_rules.keys())}")

def call_endpoint(features):
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json', Accept='application/json',
        Body=json.dumps({"features": features})
    )
    return json.loads(resp['Body'].read().decode('utf-8'))

def get_recommendations(persona, top_n=5):
    if persona not in mba_rules:
        print(f"  ⚠️  Persona '{persona}' not in rules. Available: {list(mba_rules.keys())}")
        return []

    persona_rules = mba_rules[persona]
    cold_key      = f'__persona_{persona}__'
    all_scored    = {}

    for antecedent, rules_list in persona_rules.items():
        if antecedent == cold_key:
            continue
        for rule in rules_list:
            p     = rule['product']
            score = rule['confidence'] * 0.5 + rule['lift'] * 0.5
            if p not in all_scored or score > all_scored[p]['score']:
                all_scored[p] = {'product': p, 'score': round(score, 4),
                                 'confidence': round(rule['confidence'], 4),
                                 'lift': round(rule['lift'], 4),
                                 'category': rule.get('category', '')}

    if cold_key in persona_rules:
        for rule in persona_rules[cold_key]:
            p = rule['product']
            if p not in all_scored:
                all_scored[p] = {'product': p, 'score': round(rule['lift'] * 0.1, 4),
                                 'confidence': round(rule['confidence'], 4),
                                 'lift': round(rule['lift'], 4),
                                 'category': rule.get('category', '')}

    return sorted(all_scored.values(), key=lambda x: x['score'], reverse=True)[:top_n]

def demo(user_id, top_n=5):
    row = df_clusters[df_clusters['user_id'] == user_id]
    if row.empty:
        print(f"❌ '{user_id}' not found"); return
    feat_row = df_features[df_features['user_id'] == user_id]
    if feat_row.empty:
        print(f"❌ features not found for '{user_id}'"); return

    scaled      = feat_row[FEATURES].iloc[0].tolist()
    csv_persona = row['persona'].iloc[0]
    csv_cluster = row['cluster'].iloc[0]
    result      = call_endpoint(scaled)
    persona     = result['persona']
    cluster     = result['cluster_id']
    distance    = result.get('centroid_distance', 'N/A')
    recs        = get_recommendations(persona, top_n)

    print(f"\n{'='*62}")
    print(f"  RECOMMENDATION — {user_id}")
    print(f"  CSV  → Cluster: {csv_cluster} | Persona: {csv_persona}")
    print(f"  Live → Cluster: {cluster}  | Persona: {persona} | Dist: {distance}")
    print(f"  Scaled features: {[round(f,3) for f in scaled]}")
    print(f"{'='*62}")
    if not recs:
        print("  ⚠️  No recommendations found."); return
    print(f"  {'#':<3} {'Product':<32} {'Category':<15} {'Conf':<7} Score")
    print(f"  {'-'*60}")
    for i, p in enumerate(recs, 1):
        print(f"  {i:<3} {p['product']:<32} {p['category']:<15} {p['confidence']:<7} {p['score']}")
    print(f"{'='*62}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    demo(args.user_id, args.top_n)