# Hybrid Product Recommendation System — AWS Free Tier

A complete end-to-end recommendation engine using:
- **Amazon S3** — data storage
- **Amazon SageMaker** — processing, training, endpoint
- **Market Basket Analysis** (FP-Growth / Apriori)
- **KMeans Clustering** — user segmentation
- **Hybrid Scoring** — MBA + Cluster + Rating + Price

## Run Order
```
python
01_data_creation.py
02_preprocessing_job.py
03_market_basket.py
04_kmeans_training.py
05_pipeline.py
06_demo_simulation.py
1. Deploy karo (ek baar):
bashpython 06a_deploy_endpoint.py
2. Jab endpoint ready ho, demo run karo:
bashpython 06_demo_simulation.py --user_id U0007
```

## Setup
1. Open AWS SageMaker Studio
2. Clone this repo
3. Run `pip install -r requirements.txt`
4. Set your S3 bucket name in each notebook (`BUCKET = 'your-bucket-name'`)
5. Run notebooks in order

## Folder Structure
```
hybrid-rec-system/
├── notebooks/          # Jupyter notebooks (run in order)
├── scripts/            # SageMaker job scripts
├── data/raw/           # Raw CSVs (auto-generated)
├── data/processed/     # Feature-engineered outputs
├── models/             # Saved model artifacts
└── requirements.txt
```
