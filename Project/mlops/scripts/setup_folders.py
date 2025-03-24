# scripts/setup_folders.py

import os
import argparse

FOLDERS = [
    "data/kaggle_datasets/body_fat",
    "features",
    "models",
    "reports",
    "logs",
    "scripts",
    "notebooks",
    "dashboards",
    "src",
    "tests",
    "mlruns"
]

def create_folders(with_gitkeep: bool = False):
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
        if with_gitkeep:
            gitkeep_path = os.path.join(folder, ".gitkeep")
            with open(gitkeep_path, "w") as f:
                f.write("")
            print(f"   └─ 🟢 Added: {gitkeep_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📁 Create required project folders.")
    parser.add_argument(
        "--with-gitkeep",
        action="store_true",
        help="Include .gitkeep file in each folder to keep it under version control"
    )
    args = parser.parse_args()
    
    print("🚀 Setting up folders...")
    create_folders(with_gitkeep=args.with_gitkeep)
    print("🎉 All done!")
