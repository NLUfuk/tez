#!/usr/bin/env python3
"""Create side-by-side comparison visualization."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


def create_side_by_side_comparison():
    """Create side-by-side comparison of final states."""
    demo_dir = Path("outputs/sunum_demo")
    feedback_dir = Path("outputs/sunum_feedback")
    output_dir = Path("outputs/sunum_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    gol_demo = Image.open(demo_dir / "final_gol.png")
    gol_feedback = Image.open(feedback_dir / "final_gol.png")
    v_demo = Image.open(demo_dir / "final_v.png")
    v_feedback = Image.open(feedback_dir / "final_v.png")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('Simulation Comparison: No Feedback vs With Feedback', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Top row: Conway states
    axes[0, 0].imshow(gol_demo)
    axes[0, 0].set_title('Conway Game of Life\n(No Feedback)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gol_feedback)
    axes[0, 1].set_title('Conway Game of Life\n(With Feedback)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom row: Membrane potentials
    axes[1, 0].imshow(v_demo)
    axes[1, 0].set_title('Membrane Potential\n(No Feedback)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(v_feedback)
    axes[1, 1].set_title('Membrane Potential\n(With Feedback)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_dir / "side_by_side_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Side-by-side comparison saved: {output_path}")


if __name__ == "__main__":
    create_side_by_side_comparison()

