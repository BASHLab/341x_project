import os
import shutil
import random
import argparse

def create_test_split(data_dir, split_ratio=0.1):
    """
    Moves a percentage of images from the main dataset to a 'test' folder.
    This ensures the model is trained on completely different data than it is scored on.
    """
    
    # 1. Setup Paths
    # We assume data_dir contains 'person' and 'non_person'
    # We will create 'test/person' and 'test/non_person' inside data_dir
    
    test_dir = os.path.join(data_dir, "test")
    categories = ["person", "non_person"]
    
    if os.path.exists(test_dir):
        print(f"[WARN] Test directory already exists at {test_dir}")
        print("       Skipping split to avoid deleting data or double-splitting.")
        return

    print(f"[INFO] Creating test split ({split_ratio*100}%) from {data_dir}...")

    total_moved = 0

    for category in categories:
        src_path = os.path.join(data_dir, category)
        dest_path = os.path.join(test_dir, category)
        
        if not os.path.exists(src_path):
            print(f"[ERROR] Source folder not found: {src_path}")
            continue
            
        # Create destination folder
        os.makedirs(dest_path, exist_ok=True)
        
        # Get list of all images
        all_files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(all_files)
        
        # Shuffle ensures random selection
        random.shuffle(all_files)
        
        # Calculate how many to move
        num_to_move = int(total_files * split_ratio)
        files_to_move = all_files[:num_to_move]
        
        print(f"       Processing '{category}': Moving {num_to_move} of {total_files} images...")
        
        for fname in files_to_move:
            shutil.move(os.path.join(src_path, fname), os.path.join(dest_path, fname))
            
        total_moved += num_to_move

    print("-" * 40)
    print(f"[SUCCESS] Moved {total_moved} images to '{test_dir}'.")
    print(f"          Training Set: {data_dir} (person/non_person)")
    print(f"          Test Set:     {test_dir} (person/non_person)")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="vw_coco2014_96", help="Root of the dataset")
    parser.add_argument("--split", type=float, default=0.1, help="Fraction of data to set aside (default 0.1)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset not found at {args.data}. Did you run the download command?")
        exit(1)

    create_test_split(args.data, args.split)