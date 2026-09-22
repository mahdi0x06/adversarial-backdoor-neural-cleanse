# Experiment Design

## Research Question

How does adversarial training affect the detectability of BadNet backdoors
using Neural Cleanse?

## Models

### 1. Clean Model

Trained only on clean data.

### 2. BadNet Model

Trained using a mixture of clean and BadNet-poisoned samples.

### 3. Adversarial + BadNet Model

Trained using BadNet-poisoned data together with adversarial training.

## Evaluation

Each relevant model will be evaluated using:

- Clean accuracy
- Backdoor attack success rate (ASR)
- Adversarial robustness

Neural Cleanse will then be applied to the trained models to compare:

- Reconstructed trigger patterns
- Reconstructed trigger masks
- Mask norms
- Anomaly indices
- Detected target classes
