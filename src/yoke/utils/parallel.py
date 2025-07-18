"""Yoke module to assist GPU-parallel training.

Some models within Yoke require specific modifications to PyTorch multi-GPU
training utilities.

"""

import torch
import torch.nn as nn


# Custom nn.DataParallel class to handle input to LodeRunner that should not be
# split by batch.
class LodeRunner_DataParallel(nn.DataParallel):
    """Handle unique GPU splitting of LodeRunner inputs.

    Since LodeRunner's *forward* method has multiple inputs consisting of
    several different shapes, some of which include a batch dimension and some
    of which do not, we must handle the splitting of data across multiple GPUs
    explicitly.

    """

    def __init__(self, model: nn.Module) -> None:
        """Get it initialized using parent."""
        super().__init__(model)

    def forward(self, *inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Handle explicit GPU splitting."""
        # Input is (start_img, in_vars, out_vars, Dt)
        image_input = inputs[0]
        in_vars = inputs[1]
        out_vars = inputs[2]
        Dt_input = inputs[3]

        # Split batchsize-dependent inputs and replicate fixed inputs
        if self.device_ids:
            # Copy model to device
            replicas = self.replicate(self.module, self.device_ids)

            # Split batchsize-dependent inputs
            inputs_split = nn.parallel.scatter((image_input, Dt_input), self.device_ids)

            # Replicate non-batchsize-dependent inputs
            in_vars_replicas = [in_vars.to(device) for device in self.device_ids]

            out_vars_replicas = [out_vars.to(device) for device in self.device_ids]

            # Combine splits and replicas
            inputs_combined = [
                (split_inputs[0], in_vars, out_vars, split_inputs[1])
                for split_inputs, in_vars, out_vars in zip(
                    inputs_split, in_vars_replicas, out_vars_replicas
                )
            ]

            # Forward pass with replicas and custom splits
            outputs = nn.parallel.parallel_apply(replicas, inputs_combined)

            return nn.parallel.gather(outputs, self.output_device)
        else:
            return self.module(*inputs, **kwargs)
