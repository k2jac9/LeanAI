import os
from featureform import Featureform

FEATURES_DIR = os.path.join("features")
ff = Featureform()

def register_feature_files():
    for filename in os.listdir(FEATURES_DIR):
        if filename.endswith(".yaml"):
            filepath = os.path.join(FEATURES_DIR, filename)
            try:
                ff.apply(filepath)
                print(f"✅ Registered feature: {filename}")
            except Exception as e:
                print(f"❌ Failed to register {filename}: {e}")

if __name__ == "__main__":
    print("🔧 Registering Featureform features...")
    register_feature_files()
    print("✅ All feature definitions processed.")
