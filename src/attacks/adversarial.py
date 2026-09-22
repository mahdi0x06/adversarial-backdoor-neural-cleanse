import torch
import torch.nn.functional as F


def pgd_linf(
    model,
    images,
    labels,
    epsilon,
    alpha,
    steps,
    random_start=True,
):
    original_images = images.detach()

    if random_start:
        perturbation = torch.empty_like(
            original_images
        ).uniform_(
            -epsilon,
            epsilon,
        )

        images_adv = torch.clamp(
            original_images + perturbation,
            0.0,
            1.0,
        )
    else:
        images_adv = original_images.clone()

    was_training = model.training

    model.eval()

    try:
        for _ in range(steps):

            images_adv = images_adv.detach()
            images_adv.requires_grad_(True)

            logits = model(
                images_adv
            )

            loss = F.cross_entropy(
                logits,
                labels,
            )

            gradient = torch.autograd.grad(
                loss,
                images_adv,
                only_inputs=True,
            )[0]

            with torch.no_grad():

                images_adv = (
                    images_adv
                    +
                    alpha * gradient.sign()
                )

                perturbation = (
                    images_adv
                    -
                    original_images
                )

                perturbation = torch.clamp(
                    perturbation,
                    -epsilon,
                    epsilon,
                )

                images_adv = torch.clamp(
                    original_images + perturbation,
                    0.0,
                    1.0,
                )

    finally:
        model.train(
            was_training
        )

    return images_adv.detach()