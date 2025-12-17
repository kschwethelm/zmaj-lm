"""Simple visualization utilities for attention masks."""

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def plot_attention_mask(
    mask: torch.Tensor,
    title: str = "Attention Mask",
    save_path: str | None = None,
) -> Figure | None:
    """Plot a single attention mask.

    Args:
        mask: 2D or 3D tensor. If 3D, uses the first item in batch.
        title: Title for the plot.
        save_path: Optional path to save the image. If None, returns the figure for display.

    Returns:
        The matplotlib Figure if save_path is None, otherwise None.
    """
    # Handle 3D masks by taking first batch item
    if mask.ndim == 3:
        mask = mask[0]

    # Convert to numpy and CPU
    mask_np = mask.cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(mask_np, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Mask Value")
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_mask_comparison(
    masks: list[torch.Tensor],
    titles: list[str],
    save_path: str | None = None,
) -> Figure | None:
    """Plot multiple masks side-by-side for comparison.

    Args:
        masks: List of 2D or 3D tensors. If 3D, uses first batch item.
        titles: List of titles for each mask (must match length of masks).
        save_path: Optional path to save the image. If None, returns the figure for display.

    Returns:
        The matplotlib Figure if save_path is None, otherwise None.
    """
    n_masks = len(masks)

    # Create subplots
    fig, axes = plt.subplots(1, n_masks, figsize=(6 * n_masks, 5))

    # Handle single mask case
    if n_masks == 1:
        axes = [axes]

    for ax, mask, title in zip(axes, masks, titles):
        # Handle 3D masks
        if mask.ndim == 3:
            mask = mask[0]

        mask_np = mask.cpu().numpy()

        ax.imshow(mask_np, cmap="Blues", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig
