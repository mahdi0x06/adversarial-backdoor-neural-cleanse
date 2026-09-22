from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


OUTPUT_ROOT = Path(
    "outputs/neural_cleanse"
)

FIGURE_ROOT = Path(
    "outputs/figures"
)

MODELS = [
    "clean",
    "badnet",
    "adv_badnet",
]

NUM_CLASSES = 10


def save_mask_norm_plot(
    model_name,
    summary,
):

    figure_dir = (
        FIGURE_ROOT /
        model_name
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        summary["target_class"],
        summary["mask_norm"],
    )

    plt.xlabel(
        "Target class"
    )

    plt.ylabel(
        "Mask norm"
    )

    plt.title(
        f"{model_name} - Neural Cleanse Mask Norm"
    )

    plt.xticks(
        range(NUM_CLASSES)
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir /
        "mask_norms.png",
        dpi=200,
    )

    plt.close()


def save_anomaly_plot(
    model_name,
    summary,
):

    figure_dir = (
        FIGURE_ROOT /
        model_name
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        summary["target_class"],
        summary["anomaly_score"],
    )

    plt.xlabel(
        "Target class"
    )

    plt.ylabel(
        "Anomaly score"
    )

    plt.title(
        f"{model_name} - Neural Cleanse Anomaly Scores"
    )

    plt.xticks(
        range(NUM_CLASSES)
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir /
        "anomaly_scores.png",
        dpi=200,
    )

    plt.close()


def save_trigger_grid(
    model_name,
):

    figure_dir = (
        FIGURE_ROOT /
        model_name
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        NUM_CLASSES,
        3,
        figsize=(8, 24),
    )

    for class_id in range(
        NUM_CLASSES
    ):

        result_path = (
            OUTPUT_ROOT /
            model_name /
            f"class_{class_id}.pt"
        )

        result = torch.load(
            result_path,
            map_location="cpu",
        )

        mask = (
            result["mask"]
            .squeeze(0)
            .squeeze(0)
            .numpy()
        )

        pattern = (
            result["pattern"]
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

        masked_pattern = (
            pattern
            *
            mask[..., None]
        )

        axes[class_id, 0].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[class_id, 1].imshow(
            pattern,
            vmin=0,
            vmax=1,
        )

        axes[class_id, 2].imshow(
            masked_pattern,
            vmin=0,
            vmax=1,
        )

        axes[class_id, 0].set_ylabel(
            f"Class {class_id}"
        )

        for column in range(3):
            axes[class_id, column].set_xticks([])
            axes[class_id, column].set_yticks([])

    axes[0, 0].set_title(
        "Mask"
    )

    axes[0, 1].set_title(
        "Pattern"
    )

    axes[0, 2].set_title(
        "Masked Pattern"
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir /
        "reconstructed_triggers.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def main():

    for model_name in MODELS:

        summary_path = (
            OUTPUT_ROOT /
            model_name /
            "summary.csv"
        )

        summary = pd.read_csv(
            summary_path
        )

        save_mask_norm_plot(
            model_name,
            summary,
        )

        save_anomaly_plot(
            model_name,
            summary,
        )

        save_trigger_grid(
            model_name,
        )

        print(
            f"Saved figures for {model_name}"
        )


if __name__ == "__main__":
    main()