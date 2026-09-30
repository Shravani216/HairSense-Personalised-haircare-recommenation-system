# train_kmeans.py
import pandas as pd
import numpy as np
import os, pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# ---------- Paths ----------
CANDIDATES = [
    "data/haisen_data.xlsx",
    "haisen_data.xlsx",
    "data/haisen_data.csv",
    "haisen_data.csv"
]
OUT = "model_artifacts"
os.makedirs(OUT, exist_ok=True)

# ---------- Load dataset ----------
data_path = None
for p in CANDIDATES:
    if os.path.exists(p):
        data_path = p
        break
if data_path is None:
    raise SystemExit("Dataset not found. Put haisen_data.xlsx in data/ or project root.")

ext = data_path.split(".")[-1].lower()
if ext in ("xlsx", "xls"):
    raw = pd.read_excel(data_path)
else:
    raw = pd.read_csv(data_path)

print("Loaded:", data_path, raw.shape)

# ---------- Column mapping (uses your actual headers) ----------
# These keys represent the logical features used by the app.
# They map to your sheet column names (from your posted header list).
col_map = {
    "age": "Age:",
    "gender": "Gender:",
    "location": "Location:",
    "wash_freq": "  How often do you wash your hair per week?  ",
    "texture": "What is your natural hair texture?",
    "density": "What is your hair density?",
    "porosity": "Hair porosity level (how well your hair absorbs moisture)?",
    "scalp_type": "Scalp Type:",
    "hair_concerns": "What are your current hair concerns? (Check all that apply)",
    "color_or_treat": "  Do you color, chemically treat, or heat-style your hair regularly?  ",
    "family_history": "Any family history of hair loss or scalp conditions?",
    "medical_condition": "Do you have any medical condition (e.g., PCOS, thyroid issues, anemia) that affects hair health?(if yes,please specify)",
    "medication": "Are you currently on any medication that affects your hair?",
    "hair_loss_stress": "Have you noticed hair loss increasing during stressful periods?",
    "stress_level": "How would you describe your daily stress level?",
    "sleep_hours": "How many hours of sleep do you get on average?",
    "smoke_alcohol": "  Do you smoke or consume alcohol?  ",
    "wear_caps": "  Do you wear helmets, caps, or cover your hair frequently?  ",
    "tight_tie": "   Do you tie your hair tightly (ponytails, buns, braids) on a regular basis?  ",
    "exercise": "Do you exercise regularly?",
    "occupation": "What is your occupation or typical daily activity level?",
    "diet": "How would you describe your regular diet?",
    "supplements": "Do you take any supplements? (Select all that apply)",
    "fruits_freq": "How often do you eat these foods weekly? (Scale: Never – Rarely – Sometimes – Often – Daily) [Fruits]",
    "greens_freq": "How often do you eat these foods weekly? (Scale: Never – Rarely – Sometimes – Often – Daily) [Green vegetables]",
    "nuts_freq": "How often do you eat these foods weekly? (Scale: Never – Rarely – Sometimes – Often – Daily) [Nuts/seeds]",
    "junk_freq": "How often do you eat these foods weekly? (Scale: Never – Rarely – Sometimes – Often – Daily) [Junk food]",
    "dairy_freq": "How often do you eat these foods weekly? (Scale: Never – Rarely – Sometimes – Often – Daily) [Dairy]",
    "water_liters": "How many liters of water do you drink per day?",
    "products_used": "Which products do you use? (Check all that apply)",
    "routine": "Do you follow any specific haircare routine?",
    "prefer_natural": "Would you prefer natural/home remedies over chemical products?",
    "recommendation_interest": "What kind of recommendations are you interested in? (Check all that apply)",
    "comments": "Any additional comments or problems you’d like to share?"
}

# ---------- Build cleaned DataFrame ----------
clean = pd.DataFrame()
for key, colname in col_map.items():
    if colname in raw.columns:
        clean[key] = raw[colname]
    else:
        # missing column: fill sensible default
        if key in ["age", "sleep_hours", "water_liters", "wash_freq"]:
            clean[key] = np.nan
        else:
            clean[key] = "unknown"

# clean multi-select 'hair_concerns' into comma-separated lower-case strings
if "hair_concerns" in clean.columns:
    clean["hair_concerns"] = clean["hair_concerns"].fillna("unknown").astype(str).str.lower()

# normalize strings
for c in clean.select_dtypes(include=["object"]).columns:
    clean[c] = clean[c].astype(str).str.strip().str.lower()

# numeric conversions & fill with median
for n in ["age", "sleep_hours", "water_liters", "wash_freq"]:
    if n in clean.columns:
        clean[n] = pd.to_numeric(clean[n].astype(str).str.replace(r"[^\d\.]", "", regex=True), errors="coerce")
        median = clean[n].median() if not clean[n].dropna().empty else 0
        clean[n] = clean[n].fillna(median)

print("Clean shape:", clean.shape)

# ---------- Label encoders ----------
encoders = {}
for col in clean.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    clean[col] = clean[col].fillna("unknown")
    le.fit(clean[col])
    clean[col] = le.transform(clean[col])
    encoders[col] = le

# save encoders
with open(os.path.join(OUT, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

# ---------- Feature columns & scaler ----------
feature_cols = clean.columns.tolist()
X = clean[feature_cols].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open(os.path.join(OUT, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---------- KMeans ----------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
kmeans.fit(X_scaled)
with open(os.path.join(OUT, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)

# ---------- Build cluster profiles (readable) ----------
decoded = pd.DataFrame()
for col in feature_cols:
    if col in encoders:
        classes = list(encoders[col].classes_)
        decoded[col] = clean[col].apply(lambda x: classes[int(x)] if 0 <= int(x) < len(classes) else "unknown")
    else:
        decoded[col] = clean[col]
decoded["cluster"] = kmeans.predict(X_scaled)

cluster_profiles = {}
for c in sorted(decoded["cluster"].unique()):
    sub = decoded[decoded["cluster"] == c]
    prof = {}
    for col in decoded.columns:
        if col == "cluster":
            continue
        if sub[col].dtype == object:
            prof[col] = {"mode": sub[col].mode().iloc[0] if not sub[col].mode().empty else "unknown", "count": len(sub)}
        else:
            prof[col] = {"mean": float(sub[col].mean()), "count": len(sub)}
    cluster_profiles[c] = prof

with open(os.path.join(OUT, "cluster_profiles.pkl"), "wb") as f:
    pickle.dump(cluster_profiles, f)

with open(os.path.join(OUT, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

print("Training complete. Artifacts saved to", OUT)
