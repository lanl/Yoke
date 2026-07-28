import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from yoke.models.vit.swin.bomberman import LodeRunner
#from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
#from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch
#from yoke.utils.restart import continuation_setup
#from yoke.utils.dataload import make_distributed_dataloader
#from yoke.utils.checkpointing import load_model_and_optimizer
#from yoke.utils.checkpointing import save_model_and_optimizer
#from yoke.lr_schedulers import CosineWithWarmupScheduler
#from yoke.helpers import cli

from train_LodeRunner_ddp import Kilonova_lc_img_DataSet
from torch.utils.data import DataLoader

# FIXME remove if restructure
#from torch.utils.data import Dataset, DataLoader, random_split
import glob
import random

import torch

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pdb

# matplotlib.use('MacOSX')
# matplotlib.use('pdf')
# Get rid of type 3 fonts in figures
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)

#ckpt = torch.load(
#    "runs/study_005/study005_modelState_epoch0100.pth",
#    map_location="cpu",
#    weights_only=False,   # trusted checkpoint
#)

#model = LodeRunner(**ckpt["model_args"])
#model.load_state_dict(ckpt["model_state_dict"])
#model.eval()

#file_prefix_list = sorted(glob.glob(f"/net/sescratch1/atoivonen/data/KN_lightcurves/uniform_dataset_20000/lc_*.npz"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#ckpt = torch.load("runs/study_005/study005_modelState_epoch0100.pth", map_location=device, weights_only=False)
ckpt = torch.load("runs/study_006/study006_modelState_epoch0028.pth", map_location=device, weights_only=False)

model = LodeRunner(**ckpt["model_args"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
#eval_dataset = Kilonova_lc_img_DataSet(max_timeIDX_offset=2, half_image=False, N_imgs=1)
eval_dataset = Kilonova_lc_img_DataSet(half_image=False, N_imgs=1)

xs, targets, Dts = [], [], []
idxs = []

loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
for idx, (x, target, Dt) in enumerate(loader):
    xs.append(x.mean().item())
    targets.append(target.mean().item())
    idxs.append(idx)

plt.figure()
plt.plot(idxs, xs)
plt.savefig('plot_val_lc_seq_new.png')

xs_pred, targets_pred, Dts_pred = [], [], []
preds, idxs_pred = [], []

in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
for idx, (x, target, Dt) in enumerate(loader):
    xs_pred.append(x.mean().item())
    targets_pred.append(target.mean().item())
    idxs_pred.append(idx)

    with torch.no_grad():
        pred_image = model(x, in_vars, out_vars, Dt)   # [1, 8, 1120, 400]
    pred_plot = pred_image.squeeze().detach().cpu().numpy().mean()
    preds.append(pred_plot)

print(idxs_pred, len(idxs_pred))
print(preds, len(preds))
print(targets_pred, len(targets_pred))

plt.figure()
plt.scatter(idxs_pred, preds, label='Predictions')
plt.scatter(idxs_pred, targets_pred, label='Truth')
plt.legend()
plt.gca().invert_yaxis()
plt.savefig('pred_vs_truth_seq_new.png')


loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
it = iter(loader)

x0, target0, Dt0 = next(it)
prev = x0

xs_seq = []
preds_seq = []
idxs_seq = []

for idx, (x, target, Dt) in enumerate(loader):
    xs_seq.append(x.mean().item())
    idxs_seq.append(idx)

    with torch.no_grad():
        pred_image = model(prev, in_vars, out_vars, Dt)
        prev = pred_image

    preds_seq.append(pred_image.mean().item())

plt.figure()
plt.scatter(idxs_seq, preds_seq, label='Predictions')
plt.scatter(idxs_seq, xs_seq, label='Truth')
plt.legend()
plt.gca().invert_yaxis()
plt.savefig('pred_vs_truth_seq_new2.png')

pred_plot = pred_image.squeeze().mean(dim=0).detach().cpu().numpy() #.mean(dim=0)
true_plot = target.squeeze().mean(dim=0).detach().cpu().numpy()
error_plot = pred_plot - true_plot

# --- consistent color scale for pred + truth ---
vmin = min(pred_plot.min(), true_plot.min())
vmax = max(pred_plot.max(), true_plot.max())

# --- error scale (symmetric around 0 looks better) ---
err_max = np.max(np.abs(error_plot))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

# Prediction
im1 = ax1.imshow(pred_plot, origin="lower", vmin=vmin, vmax=vmax)
ax1.set_title(f"Prediction")

# Truth
im2 = ax2.imshow(true_plot, origin="lower", vmin=vmin, vmax=vmax)
ax2.set_title(f"Truth")

# Error
im3 = ax3.imshow(error_plot, origin="lower", vmin=-err_max, vmax=err_max)
ax3.set_title("Error (Pred - Truth)")

# --- shared colorbar for pred + truth ---
cbar = fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8)
cbar.set_label("Field value")

# --- separate colorbar for error ---
cbar_err = fig.colorbar(im3, ax=ax3, shrink=0.8)
cbar_err.set_label("Error")

# Clean up axes
for ax in (ax1, ax2, ax3):
    ax.axis("off")

plt.tight_layout()
savefile = 'img_comp_test_new.png'
plt.savefig(savefile, bbox_inches="tight")

n_times = 5   # number of frames/samples to plot

pred_means = []
true_means = []
dts = []

'''
for idx in range(n_times):
    x, target, Dt = eval_dataset[idx]

    x = x.unsqueeze(0).to(device)              # [1, 8, 1120, 400]
    lead_times = Dt.unsqueeze(0).to(device)

    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

    with torch.no_grad():
        pred_image = model(x, in_vars, out_vars, lead_times)   # [1, 8, 1120, 400]

    # select one channel and average over the whole image
    pred_mean = pred_image[0, channel].mean().item()
    true_mean = target[channel].mean().item()

    pred_means.append(pred_mean)
    true_means.append(true_mean)
    dts.append(Dt.item())

times = np.arange(n_times)

plt.figure(figsize=(10, 5))
plt.plot(times, pred_means, marker="o", label="Prediction")
plt.plot(times, true_means, marker="s", label="Truth")
plt.xlabel("Sample index")
plt.ylabel(f"Mean image value, channel {channel}")
plt.title("Mean-value time series")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('series_test_dt.png')
plt.show()

times = np.cumsum([0.0] + dts[:-1])

plt.figure(figsize=(10, 5))
plt.plot(times, pred_means, marker="o", label="Prediction")
plt.plot(times, true_means, marker="s", label="Truth")
plt.xlabel("Time")
plt.ylabel(f"Mean image value, channel {channel}")
plt.title("Mean-value time series")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('series_test.png')
plt.show()
'''



'''
idx = 0
x, target, Dt = eval_dataset[idx]

x = x.unsqueeze(0).to(device)          # [1, 8, 1120, 400]
lead_times = Dt.unsqueeze(0).to(device)

in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

with torch.no_grad():
    pred_image = model(x, in_vars, out_vars, lead_times)   # [1, 8, 1120, 400]

print("x shape:", x.shape)
print("target shape:", target.shape)
print("pred_image shape:", pred_image.shape)

channel = 0

pred_plot = pred_image[0, channel].detach().cpu().numpy()
true_plot = target[channel].detach().cpu().numpy()

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

img1 = ax1.imshow(pred_plot, origin="lower")
img2 = ax2.imshow(true_plot, origin="lower")

ax1.set_title(f"Prediction channel {channel}")
ax2.set_title(f"Truth channel {channel}")

plt.colorbar(img1, ax=ax1)
plt.colorbar(img2, ax=ax2)
plt.show()

idx = 0
sample = eval_dataset[idx]
x, target, Dt = sample   # adapt this to your dataset's actual return format

x = x.unsqueeze(0).to(device)        # add batch dimension
lead_times = Dt.unsqueeze(0).to(device)

in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

with torch.no_grad():
    pred_image = model(x, in_vars, out_vars, lead_times)

print(pred_image)

#with torch.no_grad():
#    out = model(x)   # x should already be on the same device

#fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sim_params = x
true_image = target

#pred_plot = pred_image[0, 0].detach().cpu().numpy()
#true_plot = true_image[0, 0].detach().cpu().numpy()

pred_plot = pred_image[0].detach().cpu().numpy()
true_plot = target.detach().cpu().numpy()

print("x shape:", x.shape)
print("target shape:", target.shape)
print("pred_image shape:", pred_image.shape)

img1 = ax1.imshow(pred_plot, origin="lower")
img2 = ax2.imshow(true_plot, origin="lower")

# Reshape for plotting
sim_params = sim_params.numpy()
true_image = np.squeeze(true_image.numpy())
# Predictions from network must be detached from gradients in order to be
# written to numpy arrays.
pred_image = np.squeeze(pred_image.detach().numpy())
# print('Shape of image prediction:', pred_image.shape)

# Plot Truth/Prediction/Discrepancy panel.
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
#fig1.suptitle(f"Time={sim_params[-1]:.3f}us", fontsize=18)
img1 = ax1.imshow(
    true_image,
    aspect="equal",
    origin="lower",
    cmap="jet",
    vmin=true_image.min(),
    vmax=true_image.max(),
)
ax1.set_ylabel("Z-axis", fontsize=16)
ax1.set_xlabel("R-axis", fontsize=16)
ax1.set_title("True", fontsize=18)

# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes('right', size='10%', pad=0.1)
# fig1.colorbar(img1,
#               cax=cax1).set_label('Density',
#                                   fontsize=14)


img2 = ax2.imshow(
    pred_image,
    aspect="equal",
    origin="lower",
    cmap="jet",
    vmin=true_image.min(),
    vmax=true_image.max(),
)
ax2.set_title("Predicted", fontsize=18)
ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="10%", pad=0.1)
fig1.colorbar(img2, cax=cax2).set_label("Density (g/cc)", fontsize=14)

discrepancy = np.abs(true_image - pred_image)
img3 = ax3.imshow(
    discrepancy,
    aspect="equal",
    origin="lower",
    cmap="hot",
    vmin=discrepancy.min(),
    vmax=dscale * discrepancy.max(),
)
ax3.set_title("Discrepancy", fontsize=18)
ax3.tick_params(axis="y", which="both", left=False, labelleft=False)

divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="10%", pad=0.1)
fig1.colorbar(img3, cax=cax3).set_label("Discrepancy", fontsize=14)
'''

