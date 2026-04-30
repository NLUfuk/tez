"""Visualization functions for Conway-Izhikevich simulation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import imageio


def save_gol_frame(state: np.ndarray, filepath: Path, cmap: str = 'gray'):
    """
    Save Conway Game of Life state as PNG.
    
    Args:
        state: Binary grid (H, W)
        filepath: Output file path
        cmap: Colormap name
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(state, cmap=cmap, interpolation='nearest')
    plt.title('Conway Game of Life State')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


def save_v_heatmap(v: np.ndarray, filepath: Path, cmap: str = 'viridis'):
    """
    Save membrane potential as heatmap.
    
    Args:
        v: Membrane potential array (H, W)
        filepath: Output file path
        cmap: Colormap name
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(v, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Membrane Potential (mV)')
    plt.title('Membrane Potential Heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


def save_spike_raster(spike_history: List[np.ndarray], filepath: Path):
    """
    Save spike raster plot (time x cell index).
    
    Args:
        spike_history: List of spike masks (H, W) bool arrays
        filepath: Output file path
    """
    if not spike_history:
        return
    
    H, W = spike_history[0].shape
    N = H * W
    T = len(spike_history)
    
    # Flatten to (T, N) raster
    raster = np.zeros((T, N), dtype=bool)
    for t, spikes in enumerate(spike_history):
        raster[t, :] = spikes.flatten()
    
    plt.figure(figsize=(12, 8))
    # Plot only where spikes occur
    spike_times, spike_indices = np.where(raster)
    plt.scatter(spike_times, spike_indices, s=0.5, c='black', alpha=0.6)
    plt.xlabel('Time Step')
    plt.ylabel('Cell Index')
    plt.title('Spike Raster Plot')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_csv(metrics_list: List[dict], filepath: Path):
    """
    Save metrics to CSV file.
    
    Args:
        metrics_list: List of metric dictionaries
        filepath: Output CSV file path
    """
    import csv
    
    if not metrics_list:
        return
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
        writer.writeheader()
        writer.writerows(metrics_list)


def create_gif(frame_paths: List[Path], output_path: Path, duration: float = 0.1):
    """
    Create GIF from frame images.
    
    Args:
        frame_paths: List of image file paths
        output_path: Output GIF path
        duration: Frame duration in seconds
    """
    images = []
    for path in frame_paths:
        if path.exists():
            images.append(imageio.imread(path))
    
    if images:
        imageio.mimsave(output_path, images, duration=duration)

