import joblib
import os
import pickle

# ============================================
# STEP 1 - LOAD YOUR ORIGINAL MODELS
# ============================================
print("Loading original models...")
rf_model = joblib.load("models/intrusion_model.pkl")
scaler   = joblib.load("models/scaler.pkl")
encoder  = joblib.load("models/label_encoder.pkl")

# ============================================
# STEP 2 - CHECK ORIGINAL SIZES
# ============================================
def get_size(filepath):
    size = os.path.getsize(filepath)
    return f"{size / (1024*1024):.2f} MB"

print(f"intrusion_model.pkl : {get_size('models/intrusion_model.pkl')}")
print(f"scaler.pkl          : {get_size('models/scaler.pkl')}")
print(f"label_encoder.pkl   : {get_size('models/label_encoder.pkl')}")

# ============================================
# STEP 3 - REDUCE RANDOM FOREST SIZE
# ============================================
print("\nReducing Random Forest size...")

# Method 1 - Reduce number of trees (most effective)
# Original might have 100-500 trees, we reduce to 50
rf_model.n_estimators = 50
rf_model.estimators_ = rf_model.estimators_[:50]

# Method 2 - Remove unnecessary stored data
rf_model.oob_score_ = None if hasattr(rf_model, 'oob_score_') else None

# ============================================
# STEP 4 - SAVE WITH MAXIMUM COMPRESSION
# ============================================
print("Saving compressed models...")

os.makedirs("models_compressed", exist_ok=True)

# Save with highest compression (0=no compression, 9=max compression)
joblib.dump(rf_model, "models_compressed/intrusion_model.pkl", compress=9)
joblib.dump(scaler,   "models_compressed/scaler.pkl",          compress=9)
joblib.dump(encoder,  "models_compressed/label_encoder.pkl",   compress=9)

# ============================================
# STEP 5 - CHECK NEW SIZES
# ============================================
print("\n--- Size Comparison ---")
print(f"intrusion_model.pkl : {get_size('models/intrusion_model.pkl')} → {get_size('models_compressed/intrusion_model.pkl')}")
print(f"scaler.pkl          : {get_size('models/scaler.pkl')} → {get_size('models_compressed/scaler.pkl')}")
print(f"label_encoder.pkl   : {get_size('models/label_encoder.pkl')} → {get_size('models_compressed/label_encoder.pkl')}")
print("\nDone! Check models_compressed/ folder")