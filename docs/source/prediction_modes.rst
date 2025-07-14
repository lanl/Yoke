Prediction Modes Guide
=====================

This guide explains the different prediction modes available in the LodeRunner animation script and helps you choose the appropriate mode for your use case.

Overview
--------

The prediction modes determine how the model makes predictions across multiple timesteps. The mode selection can be a little confusing, so here is an example showing how this works.

Basic Setup
-----------

Assume the initial image is at time zero.
Assume you are trying to predict the n'th timestep.
This would mean that you are most likely trying to predict time k * n, for some constant k.

Mode Descriptions
-----------------

Single Mode
~~~~~~~~~~~

In single mode, this prediction would be made by taking the TRUE input for the n-1 timestep.
The model would then use that, along with timestep k to predict timestep n.

**Advantages:**
- No propagating error.
- Requires one model call to predict. (O(1) model calls)
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**
- Cannot predict multiple timesteps ahead.
- This mode takes one true timestep and predicts the next.
- Uses a constant dt, so some parameters may be ignored.

Chained Mode
~~~~~~~~~~~~

In chained mode, the following relation is used: P(n) = RM(I, n, k)

- P(n) is the prediction of the n'th timestep.
- RM(I, n, k) would mean run repeated predictions of the model.

Here is a simplified version of the code for this function::

    def RM(inp, steps, dt):
        current = inp
        for _ in range(n):  # underscore used because it doesn't matter
            current = M(current, dt)  # where M is the model.
        return current

**Advantages:**
- Requires only the initial timestep to generate predictions.
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**
- Prediction of a timestep far ahead can be costly.
- Predicting timestep n requires n calls to the model. (O(n) model calls)
- Predictions lose accuracy across the rollout.
- The prediction quality decays significantly as the rollout progresses forward.
- Uses a constant dt, so some parameters may be ignored.

Timestep Mode
~~~~~~~~~~~~~

In timestep mode, the initial image at time zero is used for ALL predictions.
The timestep by itself is used to determine how far forward to predict.

**Advantages:**
- Requires only the initial timestep to generate predictions.
- Does not cause the dt related parameters to be wasted.
- Allows for predictions of a specific timestep to only require one model call. (O(1) model calls)

**Consideration:**
- Requires a model trained with a variety of timesteps.
- Training the model to predict one timestep ahead (exclusively) will NOT work effectively in this mode.

Which Mode Should You Use?
--------------------------

You must determine that, but here are some tips that may help:

Choose **Single Mode** when:
- You want to avoid propagating errors
- You need fast predictions (O(1) model calls)
- Your model was trained to predict only one timestep ahead
- You only need to predict the next immediate timestep

Choose **Chained Mode** when:
- You need to predict multiple timesteps ahead
- You only have the initial timestep available
- Your model was trained to predict one timestep ahead
- You can tolerate increasing prediction errors over time

Choose **Timestep Mode** when:
- Your model was trained with variable timesteps
- You want efficient predictions for specific future timesteps
- You want to preserve dt-related parameter information
- You can provide the initial timestep for all predictions
