# recommend.py
import os, pickle, numpy as np

ART = "model_artifacts"
required = ["encoders.pkl", "scaler.pkl", "kmeans_model.pkl", "cluster_profiles.pkl", "feature_cols.pkl"]
for r in required:
    if not os.path.exists(os.path.join(ART, r)):
        raise SystemExit(f"Missing {r} in {ART} — run train_kmeans.py first.")

with open(os.path.join(ART, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(ART, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(ART, "kmeans_model.pkl"), "rb") as f:
    kmeans = pickle.load(f)
with open(os.path.join(ART, "cluster_profiles.pkl"), "rb") as f:
    cluster_profiles = pickle.load(f)
with open(os.path.join(ART, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)

# rule maps
HAIR_PROBLEM_MAP = {
    "dandruff": {"shampoo":"Anti-dandruff (ketoconazole/zinc pyrithione)", "mask":"Neem + yogurt mask", "remedy":"Neem rinse weekly", "drink":"Amla juice", "supplement":"Zinc", "dos":["Use anti-dandruff shampoo 2x/week"], "donts":["Avoid heavy oils on scalp"]},
    "hair fall": {"shampoo":"Strengthening shampoo (biotin)", "mask":"Onion + curd mask", "remedy":"Warm oil massage 2-3x/week", "drink":"Aloe vera drink", "supplement":"Biotin, Omega-3", "dos":["Protein rich diet"], "donts":["Avoid tight hairstyles"]},
    "dry hair": {"shampoo":"Moisturizing shampoo (argan/shea)", "mask":"Avocado + olive oil", "remedy":"Hot oil massage", "drink":"Coconut water", "supplement":"Vitamin E", "dos":["Deep condition weekly"], "donts":["Avoid frequent shampooing"]},
    "oily scalp": {"shampoo":"Clarifying shampoo (salicylic/tea tree)", "mask":"Clay mask", "remedy":"Apple-cider vinegar rinse (diluted)", "drink":"Green tea", "supplement":"Reduce processed fats", "dos":["Wash regularly"], "donts":["Avoid heavy oils near roots"]},
    "frizzy hair": {"shampoo":"Smoothing shampoo", "mask":"Banana + honey", "remedy":"Rice-water rinse", "drink":"Banana smoothie", "supplement":"Collagen", "dos":["Use anti-frizz serum"], "donts":["Avoid heat"]},
    "split ends": {"shampoo":"Repair/keratin shampoo", "mask":"Egg + olive oil", "remedy":"Trim regularly", "drink":"Protein shakes", "supplement":"Protein", "dos":["Trim regularly"], "donts":["Avoid excessive brushing"]}
}
DEFAULTS = {"shampoo":"Mild sulfate-free shampoo", "mask":"Aloe vera mask", "remedy":"Weekly oil massage", "drink":"Water", "supplement":"Multivitamin", "dos":["Hydrate","Balanced diet"], "donts":["Avoid excessive heat"]}

def safe_encode(user):
    """
    Accepts user dict (logical keys used in train_kmeans mapping).
    Returns scaled vector matching feature_cols ordering.
    """
    raw = {}
    for c in feature_cols:
        raw[c] = user.get(c, "unknown") if c in encoders else user.get(c, 0)

    # normalize strings then encode
    enc = {}
    for k, v in raw.items():
        if k in encoders:
            val = str(v).strip().lower()
            le = encoders[k]
            classes = [str(x).lower() for x in le.classes_]
            if val in classes:
                enc[k] = classes.index(val)
            else:
                # fall back to 'unknown' if present, else to mode (index 0)
                if "unknown" in classes:
                    enc[k] = classes.index("unknown")
                else:
                    enc[k] = 0
        else:
            try:
                enc[k] = float(v)
            except:
                enc[k] = 0.0

    arr = np.array([enc[c] for c in feature_cols], dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled

def craft_recommendation(user):
    """Return dictionary with recommendations."""
    # ensure hair_problems is list of lowercase strings
    probs = user.get("hair_problems", [])
    if isinstance(probs, str):
        probs = [p.strip().lower() for p in probs.split(",") if p.strip()]
    else:
        probs = [p.strip().lower() for p in (probs or [])]

    # cluster
    X = safe_encode(user)
    cluster = int(kmeans.predict(X)[0])
    profile = cluster_profiles.get(cluster, {})

    # gather rule-based outputs
    shampoos = []; masks = []; remedies = []; drinks = []; supplements = []; dos = []; donts = []
    for p in probs:
        if p in HAIR_PROBLEM_MAP:
            r = HAIR_PROBLEM_MAP[p]
            shampoos.append(r["shampoo"]); masks.append(r["mask"]); remedies.append(r["remedy"])
            drinks.append(r["drink"]); supplements.append(r["supplement"]); dos.extend(r["dos"]); donts.extend(r["donts"])

    if not shampoos:
        # infer from profile simple heuristic
        scalp_mode = profile.get("scalp_type", {}).get("mode") if profile.get("scalp_type") else None
        if scalp_mode and "oily" in str(scalp_mode).lower():
            shampoos = ["Clarifying shampoo for oily scalp"]
        else:
            shampoos = [DEFAULTS["shampoo"]]
        masks = [DEFAULTS["mask"]]; remedies=[DEFAULTS["remedy"]]; drinks=[DEFAULTS["drink"]]; supplements=[DEFAULTS["supplement"]]; dos=DEFAULTS["dos"]; donts=DEFAULTS["donts"]

    # general tips
    tips = []
    try:
        if float(user.get("sleep_hours", 7) or 7) < 7:
            tips.append("Aim for 7–8 hours sleep.")
    except: pass
    try:
        if float(user.get("water_liters", user.get("water_liters", user.get("water_intake", 2)) or 2)) < 2:
            tips.append("Increase water intake to 2–2.5 L/day.")
    except: pass
    if str(user.get("stress", user.get("stress_level", "") )).lower() in ["high", "very high"]:
        tips.append("Reduce stress with exercise, yoga or meditation.")

    def uniq(lst):
        seen=set(); out=[]
        for x in (lst or []):
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return {
        "cluster": cluster,
        "cluster_profile": profile,
        "shampoo_suggestion": uniq(shampoos),
        "hair_masks": uniq(masks),
        "home_remedies": uniq(remedies),
        "drinks": uniq(drinks),
        "supplements": uniq(supplements),
        "dos": uniq(dos),
        "donts": uniq(donts),
        "recommendation_tips": tips,
        "diet_tips": ["Include protein-rich foods, leafy greens, nuts and seeds."]
    }
