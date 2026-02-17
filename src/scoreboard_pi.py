import time
import os
import argparse
import numpy as np
import math
import json
import resource
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("[ERROR] tflite_runtime not found. Run: pip install tflite-runtime")
    exit(1)

def load_and_preprocess_image(image_path, input_shape):
    """
    Resizes and normalizes image.
    NOTE: Time spent here counts towards total latency!
    """
    img = Image.open(image_path).convert('RGB')
    # Resize expects (width, height) -> (96, 96)
    img = img.resize((input_shape[2], input_shape[1]))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def evaluate_model(model_path, data_dir):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    correct_predictions = 0
    total_images = 0
    total_times = [] # Holds (Preprocessing + Inference) times

    class_map = {'non_person': 0, 'person': 1}
    
    print(f"[INFO] Running OFFICIAL EVALUATION on Raspberry Pi...")
    
    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for image_name in os.listdir(class_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(class_dir, image_name)
            
            # --- TIMER STARTS HERE (Preprocessing + Inference) ---
            start_total = time.time()
            
            # 1. Preprocess
            input_data = load_and_preprocess_image(image_path, input_shape)

            # 2. Set Input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # 3. Inference
            interpreter.invoke()
            
            # 4. Get Output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # --- TIMER ENDS HERE ---
            end_total = time.time()
            total_times.append((end_total - start_total) * 1000) # Convert to ms

            # Decode Prediction
            prediction = np.argmax(output_data)
            if prediction == label:
                correct_predictions += 1
            total_images += 1

            if total_images % 50 == 0:
                print(f"Processed {total_images} images...", end='\r')

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    
    # Get Peak RSS Memory (in KB on Linux, convert to MB)
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    
    return accuracy, np.mean(total_times), peak_memory_mb

def calculate_score(accuracy, model_size_mb, macs_m):
    """
    Score = Accuracy - 0.3 * log10(SizeMB) - 0.001 * MegaMACs
    """
    if model_size_mb < 0.0001: model_size_mb = 0.0001
    size_penalty = 0.3 * math.log10(model_size_mb)
    macs_penalty = 0.001 * macs_m
    return accuracy - size_penalty - macs_penalty

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .tflite model")
    parser.add_argument("--data", type=str, default="./vw_coco2014_96/test", help="Path to validation dataset")
    args = parser.parse_args()

    # 1. Load Metadata JSON (Enforced Workflow)
    json_path = args.model.replace(".tflite", ".json")
    
    if not os.path.exists(json_path):
        print(f"\n[ERROR] Metadata JSON not found: {json_path}")
        print("        You must run 'src/scoreboard_cluster.py' on the cluster first!")
        print("        That script verifies MACs and exports the required JSON file.")
        exit(1)
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # 2. Extract Verified MACs
    macs = metadata.get("macs_m", 12.5)
    print(f"[INFO] Loaded Verified MACs: {macs} M")

    # 3. Get Real File Size
    size_mb = os.path.getsize(args.model) / (1024 * 1024)

    # 4. Run Eval (Returns Acc, Latency, Peak Mem)
    acc, latency, peak_rss = evaluate_model(args.model, args.data)

    # 5. Calculate Score
    final_score = calculate_score(acc, size_mb, macs)

    # Output Report
    print("\n" + "="*40)
    print(f"OFFICIAL PI SCOREBOARD REPORT")
    print("="*40)
    print(f"Model:        {os.path.basename(args.model)}")
    print(f"Top-1 Acc:    {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 40)
    print(f"Latency:      {latency:.2f} ms (Prepro + Infer)")
    print(f"Peak RSS:     {peak_rss:.2f} MB")
    print(f"Model Size:   {size_mb:.4f} MB")
    print(f"MACs:         {macs:.4f} M (Verified)")
    print("-" * 40)
    print(f"FINAL SCORE:  {final_score:.4f}")
    print("="*40)
    
    # Final Pass/Fail Check
    if acc < 0.80:
        print("[FAIL] Accuracy is below 80% threshold.")
    else:
        print("[PASS] Meets accuracy requirement.")