#!/usr/bin/env python
# coding: utf-8

# # Step 1 — Synthetic Dataset Creation
# Generates 50 products, 1000 users, 10000 transactions and uploads to S3.

# In[ ]:


# In[ ]:
#!/usr/bin/env python
# coding: utf-8

# # Step 1 — Synthetic Dataset Creation
# Generates 50 products, 1000 users, 10000 transactions and uploads to S3.

import pandas as pd
import numpy as np
import random, boto3, os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BUCKET = 'sagemaker-product-ap-south-1'
# ─────────────────────────────────────────────────────────────────────────────


# ── PRODUCTS ──────────────────────────────────────────────────────────────────
categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
products = []
for i in range(1, 51):
    cat = random.choice(categories)
    price_ranges = {
        'Electronics': (50, 800), 'Clothing': (15, 150),
        'Books': (8, 60),         'Home': (20, 300), 'Sports': (25, 400)
    }
    lo, hi = price_ranges[cat]
    products.append({
        'product_id':  f'P{i:03d}',
        'name':        f'{cat}_Item_{i}',
        'category':    cat,
        'price':       round(random.uniform(lo, hi), 2),
        'avg_rating':  round(random.uniform(3.0, 5.0), 1)
    })
products_df = pd.DataFrame(products)
print('Products:', products_df.shape)


# ── USERS ─────────────────────────────────────────────────────────────────────
regions = ['North', 'South', 'East', 'West']
users = []
for i in range(1, 1001):
    persona = random.choice(['budget', 'regular', 'premium'])
    spend_map = {'budget': (15, 80),   'regular': (60, 250),  'premium': (200, 1200)}
    order_map = {'budget': (1, 5),     'regular': (5, 20),    'premium': (15, 50)}
    lo, hi = spend_map[persona]
    ol, oh = order_map[persona]
    users.append({
        'user_id':            f'U{i:04d}',
        'age':                random.randint(18, 65),
        'avg_order_value':    round(random.uniform(lo, hi), 2),
        'total_orders':       random.randint(ol, oh),
        'region':             random.choice(regions),
        'preferred_category': random.choice(categories),
        'persona':            persona
    })
users_df = pd.DataFrame(users)
print('Users:', users_df.shape)


# ── TRANSACTIONS (with injected co-purchase combos) ───────────────────────────
# Each COMBO is a basket of products bought TOGETHER in one transaction.
# 40% of transactions use a full combo → multiple rows share the same transaction_id
# 60% of transactions are single random products

COMBOS = [
    ['P001', 'P002', 'P003'],
    ['P010', 'P011'],
    ['P020', 'P021', 'P022'],
    ['P030', 'P031'],
    ['P040', 'P041', 'P042'],
]

transactions = []
txn_id       = 1
start_date   = datetime(2024, 1, 1)
product_ids  = products_df['product_id'].tolist()

for _ in range(10000):
    user     = users_df.sample(1).iloc[0]
    txn_date = start_date + timedelta(days=random.randint(0, 364))
    txn_key  = f'T{txn_id:06d}'

    # 40% chance → pick a full combo (2–3 products in one transaction)
    if random.random() < 0.40:
        basket = random.choice(COMBOS)
    else:
        basket = [random.choice(product_ids)]   # single-item transaction

    for pid in basket:
        prod = products_df[products_df['product_id'] == pid].iloc[0]
        transactions.append({
            'transaction_id': txn_key,
            'user_id':        user['user_id'],
            'product_id':     pid,
            'quantity':       random.randint(1, 5),
            'price':          prod['price'],
            'region':         user['region'],
            'timestamp':      txn_date.strftime('%Y-%m-%d')
        })

    txn_id += 1

transactions_df = pd.DataFrame(transactions)
print('Transactions:', transactions_df.shape)

# Quick sanity check
items_per_txn = transactions_df.groupby('transaction_id')['product_id'].count()
print(f'\nTransaction sanity check:')
print(f'  Total rows            : {len(transactions_df)}')
print(f'  Unique transactions   : {transactions_df["transaction_id"].nunique()}')
print(f'  Avg products per txn  : {items_per_txn.mean():.2f}')
print(f'  Txns with 2+ products : {(items_per_txn >= 2).sum()}')


# ── SAVE LOCALLY ──────────────────────────────────────────────────────────────
os.makedirs('../data/raw', exist_ok=True)
products_df.to_csv('../data/raw/products.csv', index=False)
users_df.to_csv('../data/raw/users.csv', index=False)
transactions_df.to_csv('../data/raw/transactions.csv', index=False)
print('\nSaved locally to data/raw/')


# ── UPLOAD TO S3 ──────────────────────────────────────────────────────────────
s3 = boto3.client('s3')
for fname in ['products.csv', 'users.csv', 'transactions.csv']:
    local_path = f'../data/raw/{fname}'
    s3_key     = f'data/raw/{fname}'
    s3.upload_file(local_path, BUCKET, s3_key)
    print(f'Uploaded: s3://{BUCKET}/{s3_key}')
print('Done!')