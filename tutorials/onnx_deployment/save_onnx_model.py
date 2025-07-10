"""Example of saving a model in ONNX format."""

import os

import torch
import torch.optim as optim
from torchvision import datasets, transforms

from yoke.models.mnist_model import mnist_CNN
import yoke.torch_training_utils as tr

# inputs

# path to load model in .pth format
model_filepath = r"../../applications/harnesses/mnist_surrogate/runs/study_001/"

# name of model to be loaded in .pth format
model_name = "study001_modelState"  # .pth

# path to test (or train) data, only need one data point to save an ONNX model
data_dir = "./data/MNIST"  # args.data_dir

# batch size
batch_size = 64  # args.batch_size

# device
device = "cpu"

# model arguments for model_name
lr = 1.0
model_args = {"conv1_size": 16, "conv2_size": 28, "conv3_size": 64, "conv4_size": 128}

model_basic = mnist_CNN(conv1_size=16, conv2_size=28, conv3_size=64, conv4_size=128).to(
    device
)

optimizer = optim.Adadelta(model_basic.parameters(), lr=lr)

available_models = {"mnist_CNN": mnist_CNN}

# load model with load_model_and_optimizer
print("Loading model from load_model_and_optimizer ...")
model, checkpoint = tr.load_model_and_optimizer(
    model_filepath + model_name + ".pth", optimizer, available_models, device=device
)


# load data, one data point needed to get structure of model to
# save on onnx format
test_kwargs = {"batch_size": batch_size}
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)


# save model in ONNX format

# path to save ONNX model
onnx_model_savepath = "./data/"
os.makedirs(onnx_model_savepath, exist_ok=True)

# name of ONNX model
onnx_model_name = "study001_modelState"

# full path to ONNX model
onnx_model_path = onnx_model_savepath + onnx_model_name + ".onx"

# get an example input
images, _ = next(iter(test_loader))
example_input = images[:1].to(device)  # shape = [1, 1, 28, 28]

# create object for ONNX model handling
onx = tr.onnx_module(onnx_model_path)

# save model in ONNX format
print("Saving model in ONNX format ...")
onx.save(model, example_input)
