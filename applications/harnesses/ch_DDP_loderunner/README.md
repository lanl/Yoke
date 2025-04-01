LodeRunner Training - DDP - Chicoma
===================================

An example setup training LodeRunner using PyTorch
`DistributedDataParallel`.

The training system works currently within limitations but seems more
stable than `lightning.fabric`.

1. On Venado, it seems that beyond 4 Nodes communication conflicts arise
intermittently.

```
RuntimeError: CUDA error: uncorrectable ECC error encountered

CUDA kernel errors might be asynchronously reported at some other API
call, so the stacktrace below might be incorrect.

For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

2. On Venado, with 8 `lsc240420` fields each GPU can only fit 5 samples at a
time.

3. On Chicoma, the Giant size LodeRunner model will not fit with DDP training.

4. On Chicoma, the Big size LodeRunner model will handle per-GPU batchsizes of 10.

