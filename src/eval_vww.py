import tensorflow as tf
import numpy as np
import os
import cv2
import glob

# --- CONFIGURATION ---
MODEL_PATH = "./models/vww_96_float.tflite"
# Point this to the folder you just extracted
DATA_DIR = "./vw_coco2014_96" 

def run_eval():
    print(f"Loading Model: {MODEL_PATH}...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details['index']
    output_index = output_details['index']
    
    # Check Normalization Requirements
    is_quantized = input_details['dtype'] == np.int8
    if is_quantized:
        print(">> Mode: Int8 (Input range -128 to 127)")
    else:
        print(">> Mode: Float32 (Input range -1.0 to 1.0)")

    # Define the two folders to check
    # Structure: vw_coco2014_96/person and vw_coco2014_96/non_person
    classes = {
        0: "non_person",
        1: "person"
    }

    correct = 0
    total = 0
    
    # Limit to 500 images per class for speed (1000 total)
    # LIMIT_PER_CLASS = None

    print("\nStarting Evaluation...")

    for label_idx, folder_name in classes.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        print(f"Checking Class '{folder_name}' (Target: {label_idx}). Found {len(image_files)} images.")
        
        for i, img_path in enumerate(image_files):
            # if i >= LIMIT_PER_CLASS: break
            
            # 1. Load Image (Already 96x96!)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # 2. Convert BGR to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Preprocess
            img = np.expand_dims(img, axis=0) # Add batch dimension

            if is_quantized:
                # Int8 Normalization
                input_data = (img.astype(np.float32) - 128).astype(np.int8)
            else:
                input_data = img.astype(np.float32) / 255.0

            # 4. Inference
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)[0]

            # 5. Decode Prediction
            # Output is [Score_NonPerson, Score_Person]
            if output.shape[-1] == 2:
                prediction = 1 if output[1] > output[0] else 0
            else:
                prediction = 1 if output[0] > 0 else 0

            if prediction == label_idx:
                correct += 1
            total += 1

    print("="*40)
    print(f"FINAL ACCURACY: {correct/total:.2%} (on {total} images)")
    print("="*40)

if __name__ == "__main__":
    run_eval()