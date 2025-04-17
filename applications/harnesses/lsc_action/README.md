Study GPU optimization approaches:
==================================

Study_001:
----------

Introduced timing of the training for each epoch.

Per epoch time ~2.7 minutes

Study_002:
----------

In `makedataloader`, set `shuffle=True`, `pin_memory=True`. During
move of data to 'cuda' set `non_blocking=True`.

NO APPARENT SPEED UP PER EPOCH with batchsize=124.

Perhaps if multiple batches could be loaded on GPU
simultaneously. Second attempt at Study_002 with batchsize=8 and
batch_per_epoch=77 to see if GPU memory usage is near 100% during
training. This setup gives roughly the same number of samples as the
original batchsize=124 and batch_per_epoch=5.

Per epoch time ~0.5105 minutes!!


Study_003:
----------

Convert model into `torch.jit.script`. Needed to modify `forward`
method in model to not rely on `range` and indexing into a
`nn.moduleList` and instead enumerate the `nn.moduleList`.

Per epoch time ~0.5154 minutes. Slightly slower...???


Study_004:
----------

Compile the scripted model with `torch.compile`.

First epoch time ~0.5120 minutes. Model fails to load from checkpoint however.

If the `torch.jit.script` is removed previous to compile epoch time
slows to ~0.6775. Model will still not load from checkpoint though.

`save_model_and_optimizer_hdf5` modified to work with compiled
model. Now per epoch time 0.5138 and 0.5104 minutes for 2 different
epochs.

NOTE 1: Setting `mode='reduced-overhead'` in `torch.compile` does not
seem to speed up training.

NOTE 2: Including `fullgraph=True` along with
`mode='reduced-overhead'` may have evidence of minimal speed
up. Per-epoch times of 0.5123 and 0.5078 minutes

NOTE 3: Adding `set_to_none=True` in the gradient zeroing in the
optimizer also seems to result in a potential speed up. Per-epoch
times of 0.5092 and 0.5101 minutes.

Study_???:
----------

Still need to manually implement the cuda graph using
`torch.cuda.stream`, `torch.cuda.graph`, and `replay`.


We can fill the GPU memory with multiple batches using separate CUDA
streams. However, that does not seem to speed up per-epoch
computation.

We have to be careful of the number of workers and the amount of
memory per job as well. Notes are in the slurm scripts.