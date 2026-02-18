#!/usr/bin/env python
# coding: utf-8

# # Step 3 — Market Basket Analysis (FP-Growth)
# Discovers products frequently bought together and builds a recommendation dictionary.

# In[ ]:

#!/usr/bin/env python
# coding: utf-8
# # Step 3 — Market Basket Analysis (FP-Growth)
# Discovers products frequently bought together and builds a recommendation dictionary.
#!/usr/bin/env python
# coding: utf-8
# # Step 3 — Market Basket Analysis (FP-Growth)
# Discovers products frequently bought together and builds a recommendation dictionary.

import pandas as pd
import boto3, json, os
from mlxtend.frequent_patterns import fpgrowth, association_rules

BUCKET = 'sagemaker-product-ap-south-1'
s3 = boto3.client('s3')

# Create all required local directories upfront
os.makedirs('../data/processed/', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# ── Download basket matrix from S3 ──────────────────────────────────────────
print("Downloading basket matrix from S3...")
s3.download_file(BUCKET, 'data/processed/basket_matrix.csv', '../data/processed/basket_matrix.csv')

basket_df = pd.read_csv('../data/processed/basket_matrix.csv')

# Ensure all values are boolean (required by fpgrowth)
basket_df = basket_df.astype(bool)

print(f'Basket matrix: {basket_df.shape[0]} transactions x {basket_df.shape[1]} products')

# ── Quick data sanity check ──────────────────────────────────────────────────
items_per_transaction = basket_df.sum(axis=1)
print(f'\nData quality check:')
print(f'  Avg products per transaction : {items_per_transaction.mean():.2f}')
print(f'  Transactions with 2+ products: {(items_per_transaction >= 2).sum()}')
print(f'  Transactions with 1  product : {(items_per_transaction == 1).sum()}')
print(f'  Empty transactions           : {(items_per_transaction == 0).sum()}')

if (items_per_transaction >= 2).sum() < 100:
    print('\n⚠️  WARNING: Very few multi-item transactions. MBA results may be weak.')
    print('   Consider fixing your data generation script to include more co-purchases.')

# ── Run FP-Growth ────────────────────────────────────────────────────────────
# min_support=0.005 means a pattern must appear in at least 0.5% of transactions.
# Lowered from 0.02 to capture product pairs in sparse transaction data.
print("\nRunning FP-Growth (min_support=0.005)...")
frequent_itemsets = fpgrowth(basket_df, min_support=0.005, use_colnames=True)

multi_item_sets = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) > 1]
print(f'Frequent itemsets found : {len(frequent_itemsets)}')
print(f'Multi-item itemsets     : {len(multi_item_sets)}  <- must be > 0 for rules')
print(frequent_itemsets.sort_values('support', ascending=False).head(10))

if len(frequent_itemsets) == 0:
    raise ValueError(
        "No frequent itemsets found at min_support=0.005. "
        "Try lowering further to 0.001 or check your basket data."
    )

if len(multi_item_sets) == 0:
    raise ValueError(
        "Only single-item itemsets found — no product pairs are frequent enough. "
        "Either lower min_support further (try 0.001) or check that transactions "
        "actually contain multiple products."
    )

# ── Generate Association Rules ───────────────────────────────────────────────
# lift >= 0.5 and confidence >= 0.10 to cast a wide net first.
# Tighten these thresholds once you confirm rules are generating.
print("\nGenerating association rules (lift>=0.5, confidence>=0.10)...")
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.5)
rules = rules[rules['confidence'] >= 0.10].sort_values('lift', ascending=False).reset_index(drop=True)
print(f'Rules generated: {len(rules)}')
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))

if len(rules) == 0:
    raise ValueError(
        "No rules generated even with relaxed thresholds. "
        "Your transaction data likely lacks genuine co-purchase patterns. "
        "Check the data generation script."
    )

# ── Build recommendation dictionary ─────────────────────────────────────────
# Format: {product_id: [{product, score, confidence, lift}, ...]}
print("\nBuilding recommendation dictionary...")
rec_dict = {}

for _, row in rules.iterrows():
    for ant in row['antecedents']:
        for con in row['consequents']:
            score = float(row['lift']) * float(row['confidence'])
            rec_dict.setdefault(ant, []).append({
                'product':    con,
                'score':      round(score, 4),
                'confidence': round(float(row['confidence']), 4),
                'lift':       round(float(row['lift']), 4)
            })

# Sort each product's recs by score descending
for prod in rec_dict:
    rec_dict[prod] = sorted(rec_dict[prod], key=lambda x: x['score'], reverse=True)

print(f'Products with MBA recommendations: {len(rec_dict)}')

# Show a sample
example_prod = list(rec_dict.keys())[0]
print(f'\nSample recs for product "{example_prod}":')
for r in rec_dict[example_prod][:5]:
    print(f"  -> {r['product']}  (lift={r['lift']}, conf={r['confidence']}, score={r['score']})")

# ── Save locally & upload to S3 ──────────────────────────────────────────────
print("\nSaving and uploading to S3...")

with open('../models/mba_rules.json', 'w') as f:
    json.dump(rec_dict, f, indent=2)

# Convert frozensets -> strings before saving CSV
rules_csv = rules.copy()
rules_csv['antecedents'] = rules_csv['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
rules_csv['consequents'] = rules_csv['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
rules_csv.to_csv('../models/association_rules.csv', index=False)

s3.upload_file('../models/mba_rules.json',       BUCKET, 'models/mba_rules.json')
s3.upload_file('../models/association_rules.csv', BUCKET, 'models/association_rules.csv')

print(f'\nMBA complete!')
print(f'  mba_rules.json        -> s3://{BUCKET}/models/mba_rules.json')
print(f'  association_rules.csv -> s3://{BUCKET}/models/association_rules.csv')