# run_onnx_module.py: example of evaluating a model in ONNX format

import sys
import os

import torch
from torchvision import datasets, transforms

from yoke.models.mnist_model import mnist_CNN
import yoke.torch_training_utils as tr


# inputs

# path to load ONNX model
onnx_model_path = './data/'
os.makedirs(onnx_model_path, exist_ok=True)

# name of ONNX model
onnx_model_name = "study001_modelState"

# path to test data
data_dir = './data/MNIST' # args.data_dir

# batch size
batch_size = 64 # args.batch_size


# load data to evaluate ONNX model

test_kwargs = {"batch_size": batch_size}
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
test_ds = datasets.MNIST(
    data_dir, train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)


# evaluate onnx model

# full path to load ONNX model
onnx_model_fullpath = onnx_model_path + onnx_model_name + '.onx'

# create ONNX object
onx = tr.onnx_module(onnx_model_fullpath)

# evaluate ONNX model, test accuracy with test set by computing percentage of correctly identified classes
correct = 0
total = 0
from tqdm import tqdm
import numpy as np

with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Evaluating"):
        # data: torch.Tensor of shape [B,1,28,28], target: [B]
        
        # Convert to numpy (float32) for ONNX Runtime
        # If your model expects a flattened vector, you can reshape here:
        # data_np = data.view(data.size(0), -1).cpu().numpy().astype(np.float32)
        data_np = data.cpu().numpy().astype(np.float32)
        
        # Run the ONNX model
        outputs = onx.evaluate(data_np, verbose=False)
        
        # Get predicted class
        preds = np.argmax(outputs[0], axis=1)
        
        # Compare to ground truth
        correct += (preds == target.cpu().numpy()).sum()
        total += target.size(0)

# Compute accuracy percentage
accuracy = correct / total * 100
print(f"MNIST test accuracy: {accuracy:.2f}%")
