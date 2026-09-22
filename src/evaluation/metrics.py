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