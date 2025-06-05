LSC Action Training
==================================
The LSC Action Training Pipeline is designed to train a Transpose-CNN (TCNN) surrogate model that maps layered shaped charge (LSC) simulation geometry parameters to density images.
This will be used as the action network in the Reinforcement Learning Network for LSC.
----------

Learning Rate Scheduling: Integrates a cosine learning rate scheduler with warmup (via CosineWithWarmupScheduler) for dynamic learning rate adaptation.

train_lsc_action.py: Main training script that initiates and manages the training process (model initialization, dataset preparation, training loop, LR scheduling, checkpointing, and Slurm integration)


