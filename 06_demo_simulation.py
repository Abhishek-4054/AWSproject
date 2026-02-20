#!/usr/bin/env python
import boto3, json, io, argparse
import pandas as pd

ENDPOINT_NAME="hybrid-rec-endpoint"
REGION="ap-south-1"
BUCKET="sagemaker-product-ap-south-1"
FEATURES=['total_spend','purchase_frequency','avg_basket_value','unique_products','category_diversity','price_sensitivity']

s3=boto3.client("s3",region_name=REGION)
def lcsv(key): return pd.read_csv(io.BytesIO(s3.get_object(Bucket=BUCKET,Key=key)['Body'].read()))
def ljson(key): return json.loads(s3.get_object(Bucket=BUCKET,Key=key)['Body'].read().decode('utf-8'))

print("Loading S3...")
DFC=lcsv("models/kmeans_artifacts/user_clusters.csv")
DFF=lcsv("data/processed/kmeans_input.csv")
MBA=ljson("models/mba_rules.json")
print(f"personas in rules: {list(MBA.keys())}")

def endpoint(f):
    r=boto3.client("sagemaker-runtime",region_name=REGION)
    return json.loads(r.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Accept='application/json',Body=json.dumps({"features":f}))['Body'].read().decode('utf-8'))

def recs(persona,n=5):
    if persona not in MBA: return []
    pr=MBA[persona]; ck=f'__persona_{persona}__'; sc={}
    for ant,rl in pr.items():
        if ant==ck: continue
        for r in rl:
            p=r['product']; s=r['confidence']*0.5+r['lift']*0.5
            if p not in sc or s>sc[p]['score']:
                sc[p]={'product':p,'score':round(s,4),'confidence':round(r['confidence'],4),'lift':round(r['lift'],4),'category':r.get('category','')}
    if ck in pr:
        for r in pr[ck]:
            p=r['product']
            if p not in sc: sc[p]={'product':p,'score':round(r['lift']*0.1,4),'confidence':round(r['confidence'],4),'lift':round(r['lift'],4),'category':r.get('category','')}
    return sorted(sc.values(),key=lambda x:x['score'],reverse=True)[:n]

def demo(uid,n=5):
    row=DFC[DFC['user_id']==uid]
    if row.empty: print(f"Not found: {uid}"); return
    fr=DFF[DFF['user_id']==uid]
    if fr.empty: print("Features not found"); return
    scaled=fr[FEATURES].iloc[0].tolist()
    res=endpoint(scaled)
    persona=res['persona']
    rs=recs(persona,n)
    print(f"\n{'='*60}")
    print(f"  {uid} | Cluster:{res['cluster_id']} | Persona:{persona}")
    print(f"  Dist:{res.get('centroid_distance')} | Scaled:{[round(f,2) for f in scaled]}")
    print(f"{'='*60}")
    for i,p in enumerate(rs,1):
        print(f"  {i}. {p['product']:<30} {p['category']:<12} score:{p['score']}")
    print(f"{'='*60}\n")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--user_id",required=True)
    ap.add_argument("--top_n",type=int,default=5)
    a=ap.parse_args()
    demo(a.user_id,a.top_n)
