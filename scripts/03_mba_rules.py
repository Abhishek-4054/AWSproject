"""
scripts/03_mba_rules.py
MBA rules generate karta hai — pipeline ke Step C mein run hota hai.
Local paths use karta hai (SageMaker Processing Job ke liye).
"""

import json, io, itertools, os
import pandas as pd
from collections import defaultdict

# SageMaker Processing paths
INPUT_RAW    = '/opt/ml/processing/input/raw'
INPUT_MODELS = '/opt/ml/processing/input/models'
OUTPUT_DIR   = '/opt/ml/processing/output'

def run():
    # ── Load data ──────────────────────────────────────────────────────────────
    txn         = pd.read_csv(f'{INPUT_RAW}/transactions.csv')
    df_clusters = pd.read_csv(f'{INPUT_MODELS}/user_clusters.csv')
    print(f'Transactions: {len(txn)} | Users: {len(df_clusters)}')
    print(f'Personas: {df_clusters["persona"].value_counts().to_dict()}')

    # Merge persona
    txn = txn.merge(df_clusters[['user_id', 'persona']], on='user_id', how='left')

    PERSONAS = ['Champion', 'High-Spender', 'Frequent-Buyer', 'Occasional']

    PERSONA_CATEGORY_BOOST = {
        'Champion':       {'Electronics': 2.0, 'Sports': 1.8, 'Clothing': 1.5, 'Home': 1.2, 'Books': 0.8},
        'High-Spender':   {'Electronics': 2.5, 'Home': 2.0,  'Clothing': 1.8, 'Sports': 1.2, 'Books': 0.6},
        'Frequent-Buyer': {'Home': 2.2, 'Books': 2.0, 'Clothing': 1.5, 'Sports': 1.3, 'Electronics': 1.0},
        'Occasional':     {'Books': 2.5, 'Home': 1.8, 'Sports': 1.5, 'Clothing': 1.2, 'Electronics': 0.7},
    }

    mba_rules = {}

    for persona in PERSONAS:
        print(f'\n── {persona} ──')
        persona_txn = txn[txn['persona'] == persona]
        if len(persona_txn) == 0:
            print(f'  No transactions found, skipping.')
            continue

        product_freq = persona_txn.groupby('product_name').agg(
            count    = ('transaction_id', 'count'),
            category = ('category', 'first')
        ).reset_index()

        total_txns = persona_txn['transaction_id'].nunique()
        product_freq['support'] = product_freq['count'] / total_txns

        baskets = persona_txn.groupby('transaction_id')['product_name'].apply(list)
        multi   = baskets[baskets.apply(len) > 1]

        pair_counts = defaultdict(int)
        for basket in multi:
            for a, b in itertools.combinations(sorted(set(basket)), 2):
                pair_counts[(a, b)] += 1

        product_support  = dict(zip(product_freq['product_name'], product_freq['support']))
        product_category = dict(zip(product_freq['product_name'], product_freq['category']))
        category_boost   = PERSONA_CATEGORY_BOOST[persona]

        rules = []
        for (ant, cons), pair_count in pair_counts.items():
            if pair_count < 2:
                continue
            ant_sup  = product_support.get(ant,  0.001)
            cons_sup = product_support.get(cons, 0.001)
            pair_sup = pair_count / total_txns
            conf     = pair_sup / ant_sup
            lift     = (conf / cons_sup) * category_boost.get(product_category.get(cons,''), 1.0)
            if conf >= 0.05 and lift >= 1.0:
                rules.append({'antecedent': ant, 'product': cons,
                              'confidence': round(conf,4), 'lift': round(lift,4),
                              'support': round(pair_sup,4),
                              'category': product_category.get(cons,'')})
            conf_r = pair_sup / cons_sup if cons_sup > 0 else 0
            lift_r = (conf_r / ant_sup) * category_boost.get(product_category.get(ant,''), 1.0) if ant_sup > 0 else 0
            if conf_r >= 0.05 and lift_r >= 1.0:
                rules.append({'antecedent': cons, 'product': ant,
                              'confidence': round(conf_r,4), 'lift': round(lift_r,4),
                              'support': round(pair_sup,4),
                              'category': product_category.get(ant,'')})

        # Cold-start: top products per persona
        cold_key = f'__persona_{persona}__'
        top_prods = product_freq.sort_values('support', ascending=False).head(20)
        for _, row in top_prods.iterrows():
            cat   = row['category']
            boost = category_boost.get(cat, 1.0)
            rules.append({'antecedent': cold_key, 'product': row['product_name'],
                          'confidence': round(float(row['support']),4),
                          'lift':       round(float(row['support'])*boost*10, 4),
                          'support':    round(float(row['support']),4),
                          'category':   cat})

        rule_dict = defaultdict(list)
        for rule in rules:
            rule_dict[rule['antecedent']].append({
                'product':    rule['product'],
                'confidence': rule['confidence'],
                'lift':       rule['lift'],
                'support':    rule['support'],
                'category':   rule['category']
            })
        for ant in rule_dict:
            rule_dict[ant] = sorted(rule_dict[ant], key=lambda x: x['lift'], reverse=True)

        mba_rules[persona] = dict(rule_dict)
        total = sum(len(v) for v in rule_dict.values())
        print(f'  Rules: {total} across {len(rule_dict)} antecedents')

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f'{OUTPUT_DIR}/mba_rules.json', 'w') as f:
        json.dump(mba_rules, f, indent=2)
    print(f'\n✅ mba_rules.json saved → will upload to s3://models/')

if __name__ == '__main__':
    run()