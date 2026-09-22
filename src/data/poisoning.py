import random

from torch.utils.data import Dataset, Subset

from src.attacks.badnet import apply_trigger


def _get_label(dataset, index: int) -> int:

    if isinstance(dataset, Subset):
        original_index = dataset.indices[index]

        return _get_label(
            dataset.dataset,
            original_index,
        )

    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])

    _, label = dataset[index]

    return int(label)


class PoisonedDataset(Dataset):

    def __init__(
        self,
        dataset,
        trigger,
        poison_rate: float = 0.1,
        target_class: int = 0,
        seed: int = 14,
        exclude_target_class: bool = True,
    ):
        self.dataset = dataset
        self.trigger = trigger
        self.poison_rate = poison_rate
        self.target_class = target_class
        self.seed = seed
        self.exclude_target_class = exclude_target_class

        candidate_indices = []

        for index in range(len(dataset)):
            label = _get_label(
                dataset,
                index,
            )

            if (
                exclude_target_class
                and label == target_class
            ):
                continue

            candidate_indices.append(index)

        number_of_poisoned_samples = int(
            len(candidate_indices) * poison_rate
        )

        rng = random.Random(seed)

        self.poison_indices = set(
            rng.sample(
                candidate_indices,
                number_of_poisoned_samples,
            )
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if index in self.poison_indices:
            image = apply_trigger(
                image,
                self.trigger,
            )

            label = self.target_class

        return image, label

class BackdoorTestDataset(Dataset):

    def __init__(
        self,
        dataset,
        trigger,
        target_class,
    ):
        self.dataset = dataset
        self.trigger = trigger
        self.target_class = target_class


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):

        image, _ = self.dataset[index]


        image = apply_trigger(
            image,
            self.trigger,
        )


        return (
            image,
            self.target_class,
        )