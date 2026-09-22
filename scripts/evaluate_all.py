from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.attacks.adversarial import (
    pgd_linf,
)

from src.attacks.badnet import (
    create_square_trigger,
)

from src.data.dataset import (
    create_cifar10_datasets,
)

from src.data.poisoning import (
    BackdoorTestDataset,
)

from src.evaluation.metrics import (
    evaluate_accuracy,
    evaluate_asr,
)

from src.models.classifier import (
    create_classifier,
)

from src.training.engine import (
    evaluate_adversarial,
)


SEED = 14

DATA_ROOT = "data"

TARGET_CLASS = 0

NUM_CLASSES = 10

BATCH_SIZE = 128

NUM_WORKERS = 2


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


PGD_CONFIG = {
    "epsilon": 8 / 255,
    "alpha": 2 / 255,
    "steps": 20,
    "random_start": True,
}


RESULTS_PATH = Path(
    "outputs/metrics/model_metrics.csv"
)


def load_model(path):

    model = create_classifier(
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        path,
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


def main():

    _, _, test_dataset = (
        create_cifar10_datasets(
            root=DATA_ROOT,
            seed=SEED,
        )
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


    trigger = create_square_trigger(
        trigger_size=4,
        value=1.0,
    )


    backdoor_test_dataset = (
        BackdoorTestDataset(
            dataset=test_dataset,
            trigger=trigger,
            target_class=TARGET_CLASS,
        )
    )


    backdoor_test_loader = DataLoader(
        backdoor_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


    rows = []


    for model_name, checkpoint_path in (
        CHECKPOINTS.items()
    ):

        model = load_model(
            checkpoint_path
        )


        clean_accuracy = (
            evaluate_accuracy(
                model=model,
                loader=test_loader,
                device=DEVICE,
            )
        )


        backdoor_asr = (
            evaluate_asr(
                model=model,
                loader=backdoor_test_loader,
                target_class=TARGET_CLASS,
                device=DEVICE,
            )
        )


        robust_metrics = (
            evaluate_adversarial(
                model=model,
                loader=test_loader,
                device=DEVICE,
                attack_fn=pgd_linf,
                attack_kwargs=PGD_CONFIG,
            )
        )


        rows.append(
            {
                "model":
                    model_name,

                "clean_accuracy":
                    clean_accuracy,

                "backdoor_asr":
                    backdoor_asr,

                "robust_accuracy":
                    robust_metrics[
                        "accuracy"
                    ],
            }
        )


        print(
            f"{model_name} | "
            f"Clean: {clean_accuracy:.4f} | "
            f"ASR: {backdoor_asr:.4f} | "
            f"Robust: "
            f"{robust_metrics['accuracy']:.4f}"
        )


    results = pd.DataFrame(
        rows
    )


    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    results.to_csv(
        RESULTS_PATH,
        index=False,
    )


    print()

    print(results)

    print()

    print(
        "Saved results to:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()