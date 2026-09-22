import torch
import torch.nn.functional as F


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        logits = model(images)


        loss = F.cross_entropy(
            logits,
            labels,
        )


        loss.backward()


        optimizer.step()


        batch_size = labels.size(0)


        total_loss += (
            loss.item()
            *
            batch_size
        )


        predictions = logits.argmax(
            dim=1
        )


        total_correct += (
            predictions == labels
        ).sum().item()


        total_samples += batch_size


    return {
        "loss":
            total_loss / total_samples,

        "accuracy":
            total_correct / total_samples,
    }

@torch.no_grad()
def evaluate(
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


    return {
        "accuracy":
            total_correct / total_samples
    }