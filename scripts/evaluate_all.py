from pathlib import Path

import torch

from torch.utils.data import DataLoader

from src.data.dataset import (
    create_cifar10_datasets,
)

from src.data.poisoning import (
    BackdoorTestDataset,
)

from src.attacks.badnet import (
    create_square_trigger,
)

from src.models.classifier import (
    create_classifier,
)

from src.evaluation.metrics import (
    evaluate_accuracy,
    evaluate_asr,
)


SEED = 14

TARGET_CLASS = 0

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


def load_model(path):

    model = create_classifier()

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model.to(DEVICE)