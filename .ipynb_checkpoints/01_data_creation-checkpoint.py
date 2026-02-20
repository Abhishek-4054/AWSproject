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

BUCKET = 'sagemaker-product-ap-south-1'

# ── PRODUCTS (real names, no IDs) ─────────────────────────────────────────────
product_catalog = [
    # Electronics
    ("Samsung Galaxy S23",       "Electronics", 799),
    ("Apple AirPods Pro",        "Electronics", 249),
    ("Sony WH-1000XM5",          "Electronics", 349),
    ("iPad Air 5th Gen",         "Electronics", 599),
    ("Logitech MX Master 3",     "Electronics", 99),
    ("Dell 27inch Monitor",      "Electronics", 329),
    ("Anker PowerBank 20000mAh", "Electronics", 55),
    ("JBL Flip 6 Speaker",       "Electronics", 129),
    ("Kindle Paperwhite",        "Electronics", 139),
    ("GoPro Hero 12",            "Electronics", 399),

    # Clothing
    ("Levi's 511 Slim Jeans",    "Clothing", 69),
    ("Nike Air Max 270",         "Clothing", 149),
    ("Adidas Ultraboost 22",     "Clothing", 179),
    ("Zara Casual Blazer",       "Clothing", 89),
    ("H&M Cotton T-Shirt Pack",  "Clothing", 29),
    ("Puma Running Shorts",      "Clothing", 35),
    ("Woodland Leather Boots",   "Clothing", 119),
    ("US Polo Assn Polo Shirt",  "Clothing", 45),
    ("Peter England Formal Shirt","Clothing", 55),
    ("Van Heusen Chinos",        "Clothing", 79),

    # Books
    ("Atomic Habits",            "Books", 18),
    ("The Psychology of Money",  "Books", 15),
    ("Rich Dad Poor Dad",        "Books", 12),
    ("Zero to One",              "Books", 16),
    ("Deep Work",                "Books", 14),
    ("Sapiens",                  "Books", 20),
    ("The Lean Startup",         "Books", 17),
    ("Think and Grow Rich",      "Books", 11),
    ("Ikigai",                   "Books", 13),
    ("The Alchemist",            "Books", 10),

    # Home
    ("Philips Air Fryer",        "Home", 89),
    ("Instant Pot Duo 7-in-1",   "Home", 99),
    ("Dyson V11 Vacuum",         "Home", 499),
    ("IKEA KALLAX Shelf",        "Home", 149),
    ("Milton Water Bottle Set",  "Home", 25),
    ("Prestige Induction Cooktop","Home", 59),
    ("Godrej Microwave 25L",     "Home", 139),
    ("Pigeon Electric Kettle",   "Home", 29),
    ("Asian Paints Wall Putty",  "Home", 19),
    ("Wipro LED Bulb Pack",      "Home", 15),

    # Sports
    ("Yonex Badminton Racket",   "Sports", 49),
    ("Cosco Football Size 5",    "Sports", 29),
    ("Nivia Cricket Bat",        "Sports", 89),
    ("Boldfit Gym Gloves",       "Sports", 19),
    ("Strauss Yoga Mat",         "Sports", 35),
    ("Vector X Skipping Rope",   "Sports", 12),
    ("Reebok Gym Bag",           "Sports", 59),
    ("SG Cricket Helmet",        "Sports", 79),
    ("Decathlon Swimming Goggles","Sports", 25),
    ("Nivia Basketball",         "Sports", 45),
]

products = []
for i, (name, cat, base_price) in enumerate(product_catalog, 1):
    price = round(base_price * random.uniform(0.9, 1.1), 2)
    products.append({
        'product_name': name,
        'category':     cat,
        'price':        price,
        'avg_rating':   round(random.uniform(3.5, 5.0), 1)
    })

products_df = pd.DataFrame(products)
print('Products:', products_df.shape)
print(products_df.head())


# ── USERS (real Indian names) ─────────────────────────────────────────────────
first_names = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Ayaan","Krishna","Ishaan",
    "Shaurya","Atharv","Advik","Pranav","Advait","Dhruv","Kabir","Ritvik","Aarush","Shaan",
    "Priya","Ananya","Sneha","Pooja","Divya","Meera","Sanya","Riya","Aisha","Naina",
    "Kavya","Shruti","Tanya","Simran","Jasmine","Neha","Isha","Shreya","Aditi","Kritika",
    "Rahul","Rohan","Amit","Vijay","Suresh","Rajesh","Deepak","Ankur","Varun","Nikhil",
    "Mohit","Rajan","Saurabh","Tarun","Gaurav","Harish","Pankaj","Manoj","Karan","Aryan",
    "Sunita","Rekha","Geeta","Seema","Usha","Nisha","Mamta","Preeti","Kavita","Anjali",
    "Sakshi","Pallavi","Swati","Jyoti","Aarti","Vandana","Archana","Sonal","Monika","Shweta",
    "Yash","Harsh","Vikas","Sandeep","Naveen","Praveen","Ajay","Sanjay","Vijayalakshmi","Arun",
    "Tanmay","Parth","Rishab","Siddhant","Mihir","Samarth","Naman","Lakshya","Devraj","Chirag"
]

last_names = [
    "Sharma","Verma","Singh","Kumar","Gupta","Agarwal","Joshi","Patel","Shah","Mehta",
    "Nair","Reddy","Rao","Iyer","Pillai","Menon","Naidu","Chavan","Patil","Desai",
    "Mishra","Pandey","Tiwari","Dubey","Shukla","Tripathi","Srivastava","Yadav","Chauhan","Rajput"
]

regions = ['North', 'South', 'East', 'West']
users   = []

for i in range(1, 1001):
    fname   = random.choice(first_names)
    lname   = random.choice(last_names)
    name    = f"{fname} {lname}"
    persona = random.choice(['budget', 'regular', 'premium'])
    spend_map = {'budget': (15, 80),   'regular': (60, 250),  'premium': (200, 1200)}
    order_map = {'budget': (1, 5),     'regular': (5, 20),    'premium': (15, 50)}
    lo, hi  = spend_map[persona]
    ol, oh  = order_map[persona]
    users.append({
        'user_id':            f'U{i:04d}',
        'user_name':          name,
        'age':                random.randint(18, 65),
        'avg_order_value':    round(random.uniform(lo, hi), 2),
        'total_orders':       random.randint(ol, oh),
        'region':             random.choice(regions),
        'preferred_category': random.choice(['Electronics','Clothing','Books','Home','Sports']),
        'persona':            persona
    })

users_df = pd.DataFrame(users)
print('\nUsers:', users_df.shape)
print(users_df.head())


# ── TRANSACTIONS (co-purchase combos with real product names) ─────────────────
COMBOS = [
    ["Samsung Galaxy S23",  "Apple AirPods Pro",   "Logitech MX Master 3"],
    ["Instant Pot Duo 7-in-1", "Philips Air Fryer"],
    ["Atomic Habits",       "The Psychology of Money", "Deep Work"],
    ["Yonex Badminton Racket", "Cosco Football Size 5"],
    ["Nike Air Max 270",    "Puma Running Shorts",  "Boldfit Gym Gloves"],
    ["Dyson V11 Vacuum",    "Wipro LED Bulb Pack"],
    ["Kindle Paperwhite",   "Sapiens",              "The Alchemist"],
    ["Nivia Cricket Bat",   "SG Cricket Helmet",    "Cosco Football Size 5"],
]

product_names = products_df['product_name'].tolist()
transactions  = []
txn_id        = 1
start_date    = datetime(2024, 1, 1)

for _ in range(10000):
    user     = users_df.sample(1).iloc[0]
    txn_date = start_date + timedelta(days=random.randint(0, 364))
    txn_key  = f'T{txn_id:06d}'

    if random.random() < 0.40:
        basket = random.choice(COMBOS)
    else:
        basket = [random.choice(product_names)]

    for pname in basket:
        prod = products_df[products_df['product_name'] == pname].iloc[0]
        transactions.append({
            'transaction_id': txn_key,
            'user_id':        user['user_id'],
            'user_name':      user['user_name'],
            'product_name':   pname,
            'category':       prod['category'],
            'quantity':       random.randint(1, 5),
            'price':          prod['price'],
            'region':         user['region'],
            'timestamp':      txn_date.strftime('%Y-%m-%d')
        })
    txn_id += 1

transactions_df = pd.DataFrame(transactions)
print('\nTransactions:', transactions_df.shape)

items_per_txn = transactions_df.groupby('transaction_id')['product_name'].count()
print(f'  Total rows            : {len(transactions_df)}')
print(f'  Unique transactions   : {transactions_df["transaction_id"].nunique()}')
print(f'  Avg products per txn  : {items_per_txn.mean():.2f}')
print(f'  Txns with 2+ products : {(items_per_txn >= 2).sum()}')


# ── SAVE LOCALLY ──────────────────────────────────────────────────────────────
os.makedirs('../data/raw', exist_ok=True)
products_df.to_csv('../data/raw/products.csv',         index=False)
users_df.to_csv('../data/raw/users.csv',               index=False)
transactions_df.to_csv('../data/raw/transactions.csv', index=False)
print('\nSaved locally to ../data/raw/')


# ── UPLOAD TO S3 ──────────────────────────────────────────────────────────────
s3 = boto3.client('s3')
for fname in ['products.csv', 'users.csv', 'transactions.csv']:
    local_path = f'../data/raw/{fname}'
    s3_key     = f'data/raw/{fname}'
    s3.upload_file(local_path, BUCKET, s3_key)
    print(f'  ✅ Uploaded: s3://{BUCKET}/{s3_key}')

print('\nDone! Run 02_preprocessing_job.py next.')