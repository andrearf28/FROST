import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_comparison(original_file, interpolated_file, title, bin_width_original=6.5):
    """
    Load original and interpolated temporal profiles and plot them for comparison.
    
    Args:
        original_file: path to original txt file
        interpolated_file: path to interpolated txt file
        title: plot title
        bin_width_original: original bin width in ns (default: 6.5)
    """
    # Load original data
    original_data = np.loadtxt(original_file)
    n_bins_original = len(original_data)
    times_original = np.arange(n_bins_original) * bin_width_original
    
    # Load interpolated data
    interpolated_data = np.loadtxt(interpolated_file)
    n_bins_interpolated = len(interpolated_data)
    
    # Calculate interpolated bin width (should be bin_width_original / 10)
    bin_width_interpolated = (times_original[-1] - times_original[0]) / (n_bins_interpolated - 1)
    times_interpolated = np.linspace(times_original[0], times_original[-1], n_bins_interpolated)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Full comparison
    ax1.plot(times_original, original_data, 'o-', label='Original', linewidth=2, markersize=4, alpha=0.7)
    ax1.plot(times_interpolated, interpolated_data, '-', label='Interpolated (10x)', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('Photon counts', fontsize=12)
    ax1.set_title(f'{title} - Full range', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom on first 200 ns to see interpolation detail
    mask_original = times_original <= 200
    mask_interpolated = times_interpolated <= 200
    
    ax2.plot(times_original[mask_original], original_data[mask_original], 
             'o-', label='Original', linewidth=2, markersize=6, alpha=0.7)
    ax2.plot(times_interpolated[mask_interpolated], interpolated_data[mask_interpolated], 
             '-', label='Interpolated (10x)', linewidth=1, alpha=0.8)
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('Photon counts', fontsize=12)
    ax2.set_title(f'{title} - Zoom (0-200 ns)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Original:")
    print(f"  - Bins: {n_bins_original}")
    print(f"  - Bin width: {bin_width_original:.3f} ns")
    print(f"  - Time range: {times_original[0]:.1f} - {times_original[-1]:.1f} ns")
    print(f"  - Total counts: {np.sum(original_data):.1f}")
    print(f"  - Max value: {np.max(original_data):.6f}")
    
    print(f"\nInterpolated:")
    print(f"  - Bins: {n_bins_interpolated}")
    print(f"  - Bin width: {bin_width_interpolated:.3f} ns")
    print(f"  - Time range: {times_interpolated[0]:.1f} - {times_interpolated[-1]:.1f} ns")
    print(f"  - Total counts: {np.sum(interpolated_data):.1f}")
    print(f"  - Max value: {np.max(interpolated_data):.6f}")
    
    print(f"\nInterpolation factor: {n_bins_interpolated / n_bins_original:.2f}x")
    print(f"Bin width ratio: {bin_width_original / bin_width_interpolated:.2f}x")
    
    return fig


if __name__ == "__main__":
    # Compare 3Q (Xenon)
    fig1 = load_and_plot_comparison(
        original_file="Data/3Q.txt",
        interpolated_file="Data/interpolated/3Q_interp10x.txt",
        title="Xenon temporal profile (3Q.txt)"
    )
    plt.savefig("comparison_3Q_interpolation.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: comparison_3Q_interpolation.png")
    
    # Compare 3nonQ (Total: Ar + Xe)
    fig2 = load_and_plot_comparison(
        original_file="Data/3nonQ.txt",
        interpolated_file="Data/interpolated/3nonQ_interp10x.txt",
        title="Total temporal profile (3nonQ.txt)"
    )
    plt.savefig("comparison_3nonQ_interpolation.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: comparison_3nonQ_interpolation.png")
    
    plt.show()
