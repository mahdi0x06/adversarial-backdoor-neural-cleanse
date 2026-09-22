from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    create_cifar10_datasets,
)

from src.detection.neural_cleanse import (
    detect_all_classes,
)

from src.evaluation.metrics import (
    neural_cleanse_anomaly_indices,
)

from src.models.classifier import (
    create_classifier,
)

from src.utils.seed import (
    set_seed,
)


SEED = 14

DATA_ROOT = "data"

NUM_CLASSES = 10

BATCH_SIZE = 128

NUM_WORKERS = 2

NC_STEPS = 500

NC_LEARNING_RATE = 0.1

NC_LAMBDA_MASK = 0.01


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


CHECKPOINTS = {
    "clean":
        "outputs/checkpoints/clean_best.pt",

    "badnet":
        "outputs/checkpoints/badnet_best.pt",

    "adv_badnet":
        "outputs/checkpoints/adv_badnet_best.pt",
}


OUTPUT_ROOT = Path(
    "outputs/neural_cleanse"
)


def load_model(
    checkpoint_path,
):

    model = create_classifier(
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


def save_class_result(
    model_name,
    class_id,
    result,
):

    model_output_dir = (
        OUTPUT_ROOT /
        model_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "target_class":
                class_id,

            "mask":
                result["mask"],

            "pattern":
                result["pattern"],

            "mask_norm":
                result["mask_norm"],

            "attack_success":
                result["attack_success"],
        },
        model_output_dir /
        f"class_{class_id}.pt",
    )


def main():

    set_seed(
        SEED
    )

    _, val_dataset, _ = (
        create_cifar10_datasets(
            root=DATA_ROOT,
            seed=SEED,
        )
    )

    generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    all_rows = []

    for model_name, checkpoint_path in (
        CHECKPOINTS.items()
    ):

        print()
        print(
            f"Running Neural Cleanse on {model_name}"
        )

        model = load_model(
            checkpoint_path
        )

        results = detect_all_classes(
            model=model,
            loader=val_loader,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            steps=NC_STEPS,
            lr=NC_LEARNING_RATE,
            lambda_mask=NC_LAMBDA_MASK,
        )

        mask_norms = {
            class_id:
                result["mask_norm"]

            for class_id, result
            in results.items()
        }

        anomaly = (
            neural_cleanse_anomaly_indices(
                mask_norms
            )
        )

        model_rows = []

        for class_id, result in (
            results.items()
        ):

            save_class_result(
                model_name=model_name,
                class_id=class_id,
                result=result,
            )

            row = {
                "model":
                    model_name,

                "target_class":
                    class_id,

                "mask_norm":
                    result[
                        "mask_norm"
                    ],

                "attack_success":
                    result[
                        "attack_success"
                    ],

                "anomaly_score":
                    anomaly[
                        "scores"
                    ][class_id],
            }

            model_rows.append(
                row
            )

            all_rows.append(
                row
            )

        model_results = pd.DataFrame(
            model_rows
        )

        model_output_dir = (
            OUTPUT_ROOT /
            model_name
        )

        model_results.to_csv(
            model_output_dir /
            "summary.csv",
            index=False,
        )

        detected_class = max(
            anomaly["scores"],
            key=anomaly["scores"].get,
        )

        print(
            f"Median mask norm: "
            f"{anomaly['median']:.4f}"
        )

        print(
            f"MAD: "
            f"{anomaly['mad']:.4f}"
        )

        print(
            f"Highest anomaly class: "
            f"{detected_class}"
        )

        print(
            f"Highest anomaly score: "
            f"{anomaly['scores'][detected_class]:.4f}"
        )

    all_results = pd.DataFrame(
        all_rows
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results.to_csv(
        OUTPUT_ROOT /
        "all_models_summary.csv",
        index=False,
    )

    print()
    print(
        "Saved Neural Cleanse results to:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()