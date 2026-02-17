import time
import os
import argparse
import numpy as np
import math
import json  
import resource
import sys
import re
import io
from PIL import Image

try:
    import tensorflow as tf
    # Import profiler options specifically
    from tensorflow.python.profiler import option_builder
except ImportError:
    print("Error: TensorFlow not found. Please install TensorFlow to run this script.")
    exit(1)

def load_and_preprocess_image(image_path, input_shape):
    """Load an image and preprocess it for model input."""
    img = Image.open(image_path).convert('RGB')
    # Resize expects (width, height). TFLite shape is [1, 96, 96, 3] -> (96, 96)
    img = img.resize((input_shape[2], input_shape[1]))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def get_exact_macs(model_path):
    if not model_path or not os.path.exists(model_path):
        return None
        
    try:
        import tflite
        
        with open(model_path, 'rb') as f:
            buf = f.read()
            model = tflite.Model.GetRootAsModel(buf, 0)

        graph = model.Subgraphs(0)
        total_flops = 0.0

        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            op_code = model.OperatorCodes(op.OpcodeIndex())
            builtin_code = op_code.BuiltinCode()

            op_flops = 0.0

            # 1. Standard Convolution
            if builtin_code == tflite.BuiltinOperator.CONV_2D:
                # Output Shape: [Batch, Height, Width, OutChannels]
                out_shape = graph.Tensors(op.Outputs(0)).ShapeAsNumpy()
                # Filter Shape: [OutChannels, KernelH, KernelW, InChannels]
                filter_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
                
                # FLOPs = 2 * H * W * C_out * K_h * K_w * C_in
                op_flops = 2 * out_shape[1] * out_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]

            # 2. Depthwise Convolution
            elif builtin_code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
                out_shape = graph.Tensors(op.Outputs(0)).ShapeAsNumpy()
                filter_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
                
                # FLOPs = 2 * H * W * C_out * K_h * K_w
                op_flops = 2 * out_shape[1] * out_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]

            # 3. Fully Connected (Dense)
            elif builtin_code == tflite.BuiltinOperator.FULLY_CONNECTED:
                # Filter Shape: [N_out, N_in]
                filter_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
                # FLOPs = 2 * N_in * N_out
                op_flops = 2 * filter_shape[0] * filter_shape[1]

            total_flops += op_flops
        print(f"[DEBUG] Total FLOPs: {total_flops}")

        # Convert to MegaMACs
        # 1 MAC = 2 FLOPs
        mega_macs = (total_flops / 2) / 1_000_000
        
        print(f"[INFO] Analyzed TFLite: {total_flops/1e6:.2f} MFLOPs -> {mega_macs:.4f} MegaMACs")
        return mega_macs

    except Exception as e:
        print(f"[WARN] Failed to parse TFLite structure: {e}")
        return None

def evaluate_model(model_path, data_dir):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    correct_predictions = 0
    total_images = 0
    total_times = [] 

    # FIXED: Iterate over Folder Names, not IDs
    class_map = {'non_person': 0, 'person': 1}

    print(f"[INFO] Running sanity check on cluster...")
    print(f"[WARN] Scores here are APPROXIMATE. Latency on Pi will be different.")

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[WARN] Directory {class_dir} does not exist. Skipping.")
            continue

        for image_name in os.listdir(class_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(class_dir, image_name)
            
            # --- Timer: Preprocess + Inference ---
            start_time = time.time()
            input_data = load_and_preprocess_image(image_path, input_shape)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            # -------------------------------------
            
            total_times.append((end_time - start_time) * 1000) # ms

            if np.argmax(output_data) == label:
                correct_predictions += 1
            total_images += 1
            
            if total_images % 100 == 0:
                print(f"Processed {total_images} images...", end='\r')

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    return accuracy, peak_memory, np.mean(total_times) if total_times else 0

def calculate_score(accuracy, model_size_mb, macs_m):
    if model_size_mb < 0.0001: model_size_mb = 0.0001
    # FIXED: Use log10 as per standard scoring formula
    size_penalty = 0.3 * math.log10(model_size_mb)
    macs_penalty = 0.001 * macs_m
    return accuracy - size_penalty - macs_penalty

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VWW model on cluster")
    parser.add_argument("--model", type=str, required=True, help="Path to the TFLite model file")
    parser.add_argument("--data", type=str, default="./vw_coco2014_96/test", help="Path to validation dataset")
    # parser.add_argument("--macs", type=float, default=12.5, help="Manual override for MegaMACs (if no .h5 provided)")
    args = parser.parse_args()

    # 1. Determine MACs
    # if args.source:
    #     calculated_macs = get_exact_macs(args.source)
    #     final_macs = calculated_macs if calculated_macs else args.macs
    # else:
    #     print("[INFO] No .h5 source provided. Using default/manual MACs.")
    #     final_macs = args.macs
    final_macs = get_exact_macs(args.model)

    # 2. Get File Size
    size_mb = os.path.getsize(args.model) / (1024 * 1024)

    # 3. Run Evaluation
    acc, peak_mem, lat = evaluate_model(args.model, args.data)

    # 4. Calculate Score
    score = calculate_score(acc, size_mb, final_macs)

    print("\n" + "="*40)
    print(f"CLUSTER SANITY CHECK REPORT")
    print("="*40)
    print(f"Model:      {os.path.basename(args.model)}")
    print(f"Accuracy:   {acc:.4f} ({acc*100:.2f}%)")
    print(f"Size:       {size_mb:.4f} MB")
    print(f"MACs:       {final_macs:.4f} M")
    print("-" * 40)
    print(f"EST SCORE:  {score:.4f}")
    print("="*40)

    # 5. EXPORT JSON FOR PI
    if acc >= 0.80:
        json_path = args.model.replace(".tflite", ".json")
        metadata = {
            "model_name": os.path.basename(args.model),
            "macs_m": final_macs,
            "accuracy_cluster": acc,
            "size_mb": size_mb,
            "created_at": time.ctime()
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"[SUCCESS] Accuracy passed! Metadata exported to:")
        print(f"          {json_path}")
        print(f"          Transfer BOTH the .tflite and .json to the Pi.")
    else:
        print(f"[FAIL] Accuracy < 80%. Metadata JSON NOT generated.")