from pathlib import Path

import torch
import torch.optim as optim

from src.data.dataset import (
    create_cifar10_datasets,
    create_data_loaders,
)

from src.data.poisoning import (
    PoisonedDataset,
)

from src.attacks.badnet import (
    create_square_trigger,
)

from src.models.classifier import (
    create_classifier,
)

from src.training.engine import (
    train_one_epoch,
    evaluate,
)

from src.utils.seed import (
    set_seed,
)


SEED = 14

DATA_ROOT = "data"

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/badnet_best.pt"
)

NUM_CLASSES = 10

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 0.1

POISON_RATE = 0.1

TARGET_CLASS = 0


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    set_seed(SEED)


    train_dataset, val_dataset, test_dataset = (
        create_cifar10_datasets(
            root=DATA_ROOT,
            seed=SEED,
        )
    )


    trigger = create_square_trigger(
        trigger_size=4,
        value=1.0,
    )


    poisoned_train_dataset = PoisonedDataset(
        dataset=train_dataset,
        trigger=trigger,
        poison_rate=POISON_RATE,
        target_class=TARGET_CLASS,
        seed=SEED,
    )


    train_loader, val_loader, test_loader = (
        create_data_loaders(
            poisoned_train_dataset,
            val_dataset,
            test_dataset,
            batch_size=BATCH_SIZE,
        )
    )


    model = create_classifier(
        num_classes=NUM_CLASSES,
    )

    model = model.to(device)


    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
    )


    best_accuracy = 0.0


    for epoch in range(1, EPOCHS + 1):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )


        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )


        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )


        if val_metrics["accuracy"] > best_accuracy:

            best_accuracy = val_metrics["accuracy"]

            CHECKPOINT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_accuracy,
                },
                CHECKPOINT_PATH,
            )


    print(
        "Best validation accuracy:",
        best_accuracy,
    )


if __name__ == "__main__":
    main()