import torch


def create_square_trigger(
    channels: int = 3,
    trigger_size: int = 4,
    value: float = 1.0,
) -> torch.Tensor:

    trigger = torch.full(
        (
            channels,
            trigger_size,
            trigger_size,
        ),
        fill_value=value,
        dtype=torch.float32,
    )

    return trigger


def apply_trigger(
    image: torch.Tensor,
    trigger: torch.Tensor,
) -> torch.Tensor:

    channels, height, width = image.shape

    trigger_channels, trigger_height, trigger_width = trigger.shape

    poisoned_image = image.clone()

    poisoned_image[
        :,
        height - trigger_height : height,
        width - trigger_width : width,
    ] = trigger.to(
        device=image.device,
        dtype=image.dtype,
    )

    return poisoned_image