
import os, pickle, json, numpy as np
from io import BytesIO

FEATURES = [
    "total_spend", "purchase_frequency", "avg_basket_value",
    "unique_products", "category_diversity", "price_sensitivity"
]

PERSONA_MAP = None
MODEL       = None

def model_fn(model_dir):
    global MODEL, PERSONA_MAP
    with open(os.path.join(model_dir, "kmeans_model.pkl"), "rb") as f:
        MODEL = pickle.load(f)
    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics     = json.load(f)
        PERSONA_MAP = {int(k): v for k, v in metrics["persona_map"].items()}
    print("Model loaded.")
    return MODEL

def input_fn(request_body, content_type="application/json"):
    data = json.loads(request_body)
    return np.array(data["features"]).reshape(1, -1)

def predict_fn(input_data, model):
    cluster_id = int(model.predict(input_data)[0])
    distances  = model.transform(input_data)[0]
    persona    = PERSONA_MAP.get(cluster_id, "Unknown")
    return {
        "cluster_id":        cluster_id,
        "persona":           persona,
        "centroid_distance": round(float(distances[cluster_id]), 4)
    }

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
