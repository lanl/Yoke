Prediction Modes Guide
======================

This guide explains the different prediction modes available in the LodeRunner animation script and helps you choose the appropriate mode for your use case.

Overview
--------

The prediction modes determine how the model makes predictions across multiple timesteps. The mode selection can be a little confusing, so here is an example showing how this works.

Initial Assumptions & Definitions
---------------------------------

Here are some initial assumptions that will be used throughout this document. They are designed to make it easier to understand the differences in modes.

- Assume the initial image is at time zero.
- Assume you are trying to predict the n'th timestep. For simplicity, n is always a positive integer.
- Assume there is a constant spacing between timesteps, and call that `k`.
- This would mean that you are most likely trying to predict time `n * k`.
- Define P(n) as the prediction of the n'th timestep.
- Define M(I, dt) to be the model that takes an input image `I` and a time delta `dt` to produce a prediction.
- Define T(n) to be the true state at timestep `n`.

Mode Descriptions
-----------------

Single Mode
~~~~~~~~~~~

In single mode, the model uses the TRUE value for the n-1 timestep along with 1*k as the time delta to predict the n'th timestep. The relation is as follows: `P(n) = M(T(n-1), 1 * k)`

**Advantages:**

- No propagating error.
- Requires one model call to predict. (`O(1)` model calls)
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**

- Cannot predict multiple timesteps ahead. Therefore, it requires the full true rollout to predict.
- Uses a constant dt, so some parts of the model may be wasted.

Chained Mode
~~~~~~~~~~~~

In chained mode, the following relation is used: `P(n) = M(P(n-1), 1 * k)`, with a base case of `P(0) = I`, where `I` is the initial image at time zero.

**Advantages:**

- Requires only the initial timestep to generate predictions.
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**

- Prediction of a timestep far ahead can be costly. It costs `O(n)` model calls to predict the n'th timestep.
- Predictions lose accuracy across the rollout; the error accumulates over time.
- Uses a constant dt, so some parts of the model may be wasted.

Timestep Mode
~~~~~~~~~~~~~

In timestep mode, the initial image at time zero is used for ALL predictions.
The timestep by itself is used to determine how far forward to predict. The relationship would look like this: `P(n) = M(T(0), n * k)`.

**Advantages:**

- Requires only the initial timestep to generate predictions.
- Does not cause the dt related parameters to be wasted.
- Allows for predictions of a specific timestep to only require one model call. (`O(1)` model calls)

**Consideration:**

- Training the model to predict one timestep ahead (exclusively) will NOT work effectively in this mode. You need to train with a variety of timesteps, so that the model will properly learn how to predict for different time deltas.

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
- You want efficient predictions (O(1) model calls) for specific future timesteps
- You want to preserve dt-related parameter information
- You only have the initial timestep available.
