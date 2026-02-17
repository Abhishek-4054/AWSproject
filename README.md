# Hybrid Product Recommendation System — AWS Free Tier

A complete end-to-end recommendation engine using:
- **Amazon S3** — data storage
- **Amazon SageMaker** — processing, training, endpoint
- **Market Basket Analysis** (FP-Growth / Apriori)
- **KMeans Clustering** — user segmentation
- **Hybrid Scoring** — MBA + Cluster + Rating + Price

## Run Order
```
01_data_creation.ipynb
02_preprocessing_job.ipynb
03_market_basket.ipynb
04_kmeans_training.ipynb
05_pipeline.ipynb
06_demo_simulation.ipynb
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
