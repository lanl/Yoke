# ONNX Deployment

This tutorial shows how to save and load/evaluate a trained model using ONNX, which is loaded from a .pth file. This is useful for deploying trained models so that it can be easily used.

---

## Table of Contents

- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Usage](#usage)  

---

## Features

- **Feature 1**: Save a model in ONNX format from a .pth file.  
- **Feature 2**: Evaluate the ONNX model.  

---

## Prerequisites

Ensure onnx and onnxruntime are installed. pyproject.toml was updated to include these dependencies when installing yoke. You can also install these dependencies with:
   ```bash
   pip install onnx onnxruntime
   ```

---

## Usage

1. **Save ONNX model**  
   In run_onnx_model.py, ensure the variables model_filepath and model_name correctly point to the model you want to load in .pth format.
   Ensure the variables onnx_model_savepath and onnx_model_name correctly point to the .onx file you want to save.
   Run the script with:
   ```bash
   python save_onnx_model.py
   ```
   This will save the onnx model with the same name and file path with an .onx extension.

2. **Load and evaluate ONNX model**  
   In run_onnx_model.py, ensure the variables onnx_model_path and onnx_model_name correctly point to the model you want to load in .onx format.
   Ensure the data you want to evaluate the model with is correctly set up.
   The data in this example is the test set of the MNIST data set.
   Run the script with:
   ```bash
   python run_onnx_model.py
   ```
   This will load and evaluate the onnx model.
