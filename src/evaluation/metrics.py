import torch


@torch.no_grad()
def evaluate_accuracy(
    model,
    loader,
    device,
):
    model.eval()

    total_correct = 0
    total_samples = 0


    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)


        logits = model(images)


        predictions = logits.argmax(
            dim=1
        )


        total_correct += (
            predictions == labels
        ).sum().item()


        total_samples += labels.size(0)


    return (
        total_correct /
        total_samples
    )


@torch.no_grad()
def evaluate_asr(
    model,
    loader,
    target_class,
    device,
):
    model.eval()

    total_success = 0
    total_samples = 0


    for images, _ in loader:

        images = images.to(device)


        logits = model(images)


        predictions = logits.argmax(
            dim=1
        )


        success = (
            predictions == target_class
        ).sum().item()


        total_success += success

        total_samples += images.size(0)


    return (
        total_success /
        total_samples
    )

import numpy as np


def neural_cleanse_anomaly_indices(
    mask_norms,
):

    values = np.asarray(
        list(
            mask_norms.values()
        ),
        dtype=np.float64,
    )

    median = np.median(
        values
    )

    mad = np.median(
        np.abs(
            values - median
        )
    )

    normalized_mad = (
        1.4826 * mad
    )

    denominator = max(
        normalized_mad,
        1e-12,
    )

    scores = {}


    for class_id, value in (
        mask_norms.items()
    ):

        scores[class_id] = (
            median - value
        ) / denominator


    return {
        "median":
            float(median),

        "mad":
            float(mad),

        "scores":
            scores,
    }