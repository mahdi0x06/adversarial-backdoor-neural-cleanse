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

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

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


def train_one_epoch_adversarial(
    model,
    loader,
    optimizer,
    device,
    attack_fn,
    attack_kwargs,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        images_adv = attack_fn(
            model=model,
            images=images,
            labels=labels,
            **attack_kwargs,
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images_adv
        )

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

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        loss = F.cross_entropy(
            logits,
            labels,
        )

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


def evaluate_adversarial(
    model,
    loader,
    device,
    attack_fn,
    attack_kwargs,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        images_adv = attack_fn(
            model=model,
            images=images,
            labels=labels,
            **attack_kwargs,
        )

        with torch.no_grad():

            logits = model(
                images_adv
            )

            loss = F.cross_entropy(
                logits,
                labels,
            )

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