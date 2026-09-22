from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_cifar10_datasets(
    root: str | Path,
    val_size: int = 5000,
    seed: int = 14,
    download: bool = True,
):

    root = Path(root)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    eval_transform = transforms.ToTensor()

    train_full = datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transform,
        download=download,
    )

    val_full = datasets.CIFAR10(
        root=root,
        train=True,
        transform=eval_transform,
        download=False,
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=eval_transform,
        download=download,
    )

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_full),
        generator=generator,
    )

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(
        train_full,
        train_indices.tolist(),
    )

    val_dataset = Subset(
        val_full,
        val_indices.tolist(),
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 128,
    num_workers: int = 1,
    pin_memory: bool = True,
    seed: int = 14,
):

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
