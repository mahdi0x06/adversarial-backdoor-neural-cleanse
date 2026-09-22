# Adversarial Backdoor Neural Cleanse

An experimental study of how adversarial training affects BadNet backdoors
and their detectability using Neural Cleanse.

## Experiment

The project compares three models:

1. Clean model
2. BadNet model
3. Adversarially trained BadNet model

The models will be evaluated using:

- Clean accuracy
- Backdoor attack success rate (ASR)
- Adversarial robustness
- Neural Cleanse trigger reconstruction
- Neural Cleanse anomaly index

## Project Status

Work in progress.

The final experiments will be executed on Google Colab.
Model checkpoints and experiment artifacts will be exported before the
Colab runtime terminates.
