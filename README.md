# ECE341x Final Project 

This project implements a Person Detection model (MobileNetV1 0.25 96x96). The goal is to maximize the project score by balancing high accuracy with minimal model size and complexity on a Raspberry Pi.

## Quick Start: Environment Setup
To avoid version conflicts (especially with Keras 3 or GLIBCXX), use the provided conda environment:

```bash
conda env create -f environment.yml
conda activate vww_env
```
## Dataset Preparation

We use the vw_coco2014_96 dataset. This contains pre-cropped 96x96 images suitable for MobileNet architecture. 

1. Download:

```bash
wget [https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz](https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz)
tar -xvf vw_coco2014_96.tar.gz
```

2. Create Hold-Out Test Set:
Run this script to randomly move 10% of the images into a test/ folder. This ensures the training script enver sees the images for your final score.

```bash
python src/create_test_set.py
```

2. Structure: Ensure your directory structure looks like this:

```bash
341x_project/
├── src/
│   ├── train_vww.py
│   └── vww_model.py
├── vw_coco2014_96/
│   ├── person/
│   └── non_person/
│   └── test/
└── models/
```

## Training
A common issue found when testing on the campus cluster is TensorFlow not recgnizing or finding the GPU libraries. Not using a GPU when training could increase training time by more than 2x. These commands helped:

```bash
# Point to your conda environment's library folder
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/vww_env/lib:$LD_LIBRARY_PATH
# Help XLA find CUDA
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
```

Make sure that you are using a GPU either through sinteractive or a content: slurm job when training!


## Evaluation and Socring

Use the provided scoreboard.py to evaluate your final .tflite model. 

```bash
python src/scoreboard.py --model PATH-TO-YOUR-MODEL --data PATH-TO-YOUR-TEST-SET

```

The above scoreboard.py is for you to do a sanity check of your project on the cluster, and to calculate MACs, since MAC calculations can be strenous on a Raspberry Pi. The above scoreboard.py outputs a json file the same name as your model. 

Once you run scoreboard.py, please run scoreboard_pi.py on your Raspberry Pi:
```bash
python src/scoreboard_pi.py --model PATH-TO-YOUR-MODEL --data PATH-TO-YOUR-TEST-SET
```

Scoring formula

$$Score = Accuracy - 0.3 \times \log_{10}(\text{ModelSize}_{MB}) - 0.001 \times \text{MACs}$$

| Error | Cause | Potential Fix |
| :--- | :--- | :--- |
| **GLIBCXX_3.4.29 not found** | OS uses old C++ libraries. | Export the `LD_LIBRARY_PATH` to your conda `/lib` folder. |
| **Unrecognized keyword arguments** | Keras 3 version mismatch. | Make sure to use the verson of TensorFlow in the yml file. |
| **ModuleNotFoundError: PIL** | Pillow is missing in env. | `pip install Pillow` |
| **Skipping registering GPU...** | CUDA libraries not found. | Ensure you are on a GPU node and `LD_LIBRARY_PATH` is exported. |

## Hints

A standard "dense" model will likely get a poor score. You are expected to explore different strategies. You will need to research and modify your training/conversion code to implement these.

### Weight Pruning (Sparsity)

Pruning zeros out "unimportant" weights. You can use Pruning-Aware Training (PAT) to help the model maintain accuracy while becoming sparse.

Hint: Look into the tensorflow_model_optimization (tfmot) library, specifically prune_low_magnitude.

### Quantization:

Standard models use 32-bit floats. Converting to 8-bit integers (INT8) reduces size by 4x.

Hint: Look at the TFLiteConverter options for optimizations and representative_dataset.

## train_vww.py

The training happens all in this train_vww.py. In order to apply techniques learned in class, you will need to update this train_vww.py file. 

Script Overview
Configurations: The script defines IMAGE_SIZE, BATCH_SIZE, and EPOCHS.

Note: Ensure the dataset path points to your training data, not the held-out test set.

Data Pipelines: The datagen, train_generator, and val_generator blocks handle the 90/10 training/validation split. These can be used as-is, or augmented to improve generalization.

Main Loop: The main() function is the engine room. You are encouraged to add command-line arguments (e.g., using argparse or absl.flags) to make testing different hyperparameters (like pruning ratios or learning rates) faster.

The Training Engine: train_epochs() handles the actual compilation and fitting. This is where you can experiment with different optimizers, loss functions, and metrics.

In general, the train_vww.py will save a .keras model, which is ***NOT*** compatible with a RaspberryPi (the scoreboard.py and scoreboard_pi.py will not run with that), so you will have to convert it. 

The snippet below shows the general way of going about that:
```bash
from tensorflow.contrib import lite
converter = lite.TFLiteConverter.from_keras_model_file( 'model.h5' ) # Your model's name
model = converter.convert()
file = open( 'model.tflite' , 'wb' ) 
file.write( model )

```
