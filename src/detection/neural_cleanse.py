import torch
import torch.nn.functional as F

from src.detection.optimizer import (
    TriggerOptimizer,
)


def reverse_engineer_trigger(
    model,
    loader,
    target_class,
    device,
    steps=500,
    lr=0.1,
    lambda_mask=0.01,
):

    model.eval()

    trigger_optimizer = TriggerOptimizer().to(
        device
    )

    optimizer = torch.optim.Adam(
        trigger_optimizer.parameters(),
        lr=lr,
    )

    loader_iterator = iter(loader)

    final_attack_success = 0.0

    for _ in range(steps):

        try:
            images, labels = next(
                loader_iterator
            )

        except StopIteration:
            loader_iterator = iter(loader)

            images, labels = next(
                loader_iterator
            )

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        keep = (
            labels != target_class
        )

        if keep.sum().item() == 0:
            continue

        images = images[
            keep
        ]

        targets = torch.full(
            (
                images.size(0),
            ),
            target_class,
            dtype=torch.long,
            device=device,
        )

        poisoned, mask, pattern = (
            trigger_optimizer(
                images
            )
        )

        logits = model(
            poisoned
        )

        loss_ce = F.cross_entropy(
            logits,
            targets,
        )

        loss_mask = mask.abs().sum()

        loss = (
            loss_ce
            +
            lambda_mask * loss_mask
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            predictions = logits.argmax(
                dim=1
            )

            final_attack_success = (
                predictions == targets
            ).float().mean().item()

    with torch.no_grad():

        mask = (
            trigger_optimizer
            .get_mask()
            .detach()
            .cpu()
        )

        pattern = (
            trigger_optimizer
            .get_pattern()
            .detach()
            .cpu()
        )

        mask_norm = (
            mask.abs()
            .sum()
            .item()
        )

    return {
        "target_class":
            target_class,

        "mask":
            mask,

        "pattern":
            pattern,

        "mask_norm":
            mask_norm,

        "attack_success":
            final_attack_success,
    }


def detect_all_classes(
    model,
    loader,
    num_classes,
    device,
    steps=500,
    lr=0.1,
    lambda_mask=0.01,
):

    results = {}


    for target_class in range(
        num_classes
    ):

        result = reverse_engineer_trigger(
            model=model,
            loader=loader,
            target_class=target_class,
            device=device,
            steps=steps,
            lr=lr,
            lambda_mask=lambda_mask,
        )


        results[target_class] = (
            result
        )


        print(
            f"Class {target_class} | "
            f"Mask norm: "
            f"{result['mask_norm']:.4f} | "
            f"Attack success: "
            f"{result['attack_success']:.4f}"
        )


    return results