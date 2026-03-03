#!/usr/bin/env python3
"""
Visualize Before/After Low-Pass Filtering for Your Actual Data

This script loads raw parquet files from a bundle and shows:
1. Before filtering: Raw voltage/current traces (with noise)
2. After filtering: Clean voltage/current traces
3. Frequency spectra comparison
4. Zoomed-in time windows to see noise removal clearly

Usage:
    python plot_filter_before_after.py /path/to/bundle_dir

Example:
    python plot_filter_before_after.py /path/to/sub_131113
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import sys
import json

# Import the filter function
from lowpass_filter import apply_butterworth_lowpass


def load_parquet_data(bundle_dir, data_type='mV'):
    """Load voltage or current data from parquet files.
    
    Args:
        bundle_dir: Path to bundle directory
        data_type: 'mV' for voltage or 'pA' for current
        
    Returns:
        DataFrame with data in appropriate format
    """
    bundle_path = Path(bundle_dir)
    
    # Find the matching parquet file (check both current dir and subdirs)
    pattern = f"{data_type}_*.parquet"
    parquet_files = list(bundle_path.glob(pattern))
    
    # If not found in current dir, look in subdirectories
    if not parquet_files:
        parquet_files = list(bundle_path.rglob(pattern))
    
    if not parquet_files:
        raise FileNotFoundError(f"No {pattern} files found in {bundle_dir} or subdirectories")
    
    parquet_file = parquet_files[0]
    print(f"Loading: {parquet_file.relative_to(bundle_path)}")
    
    df = pd.read_parquet(parquet_file)
    
    # If data is in long format (has 'sweep', 'value' columns), pivot it
    if 'sweep' in df.columns and 'value' in df.columns and 't_s' in df.columns:
        print("  Converting from long format to wide format...")
        # Pivot to get sweeps as columns
        df = df.pivot_table(index='t_s', columns='sweep', values='value', aggfunc='first')
        df.columns = [f'sweep_{int(col)}' for col in df.columns]
    
    return df


def filter_data(data, fs=200000, cutoff=5000, order=4):
    """Apply 5 kHz Butterworth low-pass filter.
    
    Args:
        data: 1D numpy array of signal
        fs: Sampling frequency (Hz)
        cutoff: Cutoff frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered data (same shape as input)
    """
    # Design filter
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    # Apply forward-backward filter (filtfilt)
    filtered = signal.filtfilt(b, a, data)
    
    return filtered


def plot_sweep_comparison(sweep_data_raw, sweep_data_filtered, sweep_num, fs=200000, title_prefix=''):
    """Plot before/after filtering for a single sweep.
    
    Args:
        sweep_data_raw: Raw voltage/current data
        sweep_data_filtered: Filtered voltage/current data
        sweep_num: Sweep number (for labeling)
        fs: Sampling frequency
        title_prefix: Prefix for plot title (e.g., 'Voltage' or 'Current')
    """
    # Time array (in milliseconds)
    duration = len(sweep_data_raw) / fs
    time_ms = np.linspace(0, duration * 1000, len(sweep_data_raw))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Trace Comparison - Sweep {sweep_num}\n(Before vs After 5 kHz Low-Pass Filter)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Full sweep - Raw data
    ax1 = axes[0]
    ax1.plot(time_ms, sweep_data_raw, 'r-', alpha=0.7, linewidth=0.8, label='Before filter (raw)')
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('1️⃣ Full Trace - Raw Data (WITH NOISE)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, duration * 1000])
    
    # Add note
    ax1.text(0.02, 0.95, 'Notice the high-frequency noise\n(rapid oscillations)', 
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))
    
    # Plot 2: Full sweep - Filtered data
    ax2 = axes[1]
    ax2.plot(time_ms, sweep_data_filtered, 'b-', alpha=0.7, linewidth=0.8, label='After filter (clean)')
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title('2️⃣ Full Trace - Filtered Data (NOISE REMOVED)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, duration * 1000])
    
    # Add note
    ax2.text(0.02, 0.95, 'Much smoother!\nHigh-frequency noise is gone', 
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot 3: Zoomed window - both overlaid for direct comparison
    ax3 = axes[2]
    
    # Choose a window in the middle
    window_start_idx = len(sweep_data_raw) // 2
    window_duration_ms = 50  # Show 50 ms window
    window_samples = int(window_duration_ms * fs / 1000)
    window_end_idx = min(window_start_idx + window_samples, len(sweep_data_raw))
    
    window_time = time_ms[window_start_idx:window_end_idx]
    window_raw = sweep_data_raw[window_start_idx:window_end_idx]
    window_filtered = sweep_data_filtered[window_start_idx:window_end_idx]
    
    ax3.plot(window_time, window_raw, 'r-', alpha=0.6, linewidth=1.5, label='Before filter')
    ax3.plot(window_time, window_filtered, 'b-', alpha=0.8, linewidth=1.5, label='After filter')
    ax3.set_xlabel('Time (ms)', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    ax3.set_title(f'3️⃣ Zoomed Window ({window_duration_ms} ms) - Noise is Small but Present', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Add note
    ax3.text(0.02, 0.95, 'If traces look nearly identical:\n✅ Your data is already clean!\nNoise is small (0.1-5% of signal)\nBut filter still removes it.',
            transform=ax3.transAxes, fontsize=8.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    return fig


def plot_frequency_comparison(sweep_data_raw, sweep_data_filtered, sweep_num, fs=200000, title_prefix=''):
    """Plot frequency spectrum before/after filtering.
    
    Args:
        sweep_data_raw: Raw voltage/current data
        sweep_data_filtered: Filtered voltage/current data
        sweep_num: Sweep number (for labeling)
        fs: Sampling frequency
        title_prefix: Prefix for plot title
    """
    # Compute FFTs
    fft_raw = np.abs(np.fft.fft(sweep_data_raw))
    fft_filtered = np.abs(np.fft.fft(sweep_data_filtered))
    freqs = np.fft.fftfreq(len(sweep_data_raw), 1/fs)
    
    # Only plot positive frequencies up to 50 kHz
    mask = (freqs > 0) & (freqs < 50000)
    freqs_plot = freqs[mask]
    fft_raw_plot = fft_raw[mask]
    fft_filtered_plot = fft_filtered[mask]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'{title_prefix} Frequency Spectrum Comparison - Sweep {sweep_num}\n(Before vs After 5 kHz Low-Pass Filter)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Linear scale
    ax1 = axes[0]
    ax1.plot(freqs_plot, fft_raw_plot, 'r-', alpha=0.6, linewidth=1.5, label='Before filter')
    ax1.plot(freqs_plot, fft_filtered_plot, 'b-', alpha=0.8, linewidth=1.5, label='After filter')
    ax1.axvline(5000, color='green', linestyle='--', linewidth=2, label='5 kHz cutoff')
    ax1.fill_between([0, 5000], 0, max(fft_raw_plot)*1.1, alpha=0.1, color='green', label='Kept')
    ax1.fill_between([5000, 50000], 0, max(fft_raw_plot)*1.1, alpha=0.1, color='red', label='Removed')
    ax1.set_ylabel('Magnitude', fontsize=10)
    ax1.set_title('Linear Scale - See Low Frequencies Clearly', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim([0, 50000])
    
    # Plot 2: Log scale
    ax2 = axes[1]
    ax2.semilogy(freqs_plot, fft_raw_plot, 'r-', alpha=0.6, linewidth=1.5, label='Before filter')
    ax2.semilogy(freqs_plot, fft_filtered_plot, 'b-', alpha=0.8, linewidth=1.5, label='After filter')
    ax2.axvline(5000, color='green', linestyle='--', linewidth=2, label='5 kHz cutoff')
    ax2.fill_between([0, 5000], 1, 1e10, alpha=0.1, color='green', label='Kept')
    ax2.fill_between([5000, 50000], 1, 1e10, alpha=0.1, color='red', label='Removed')
    ax2.set_xlabel('Frequency (Hz)', fontsize=10)
    ax2.set_ylabel('Magnitude (log scale)', fontsize=10)
    ax2.set_title('Log Scale - See High Frequencies and Attenuation Clearly', fontsize=11, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_xlim([0, 50000])
    ax2.set_ylim([1, 1e10])
    
    # Add annotation
    ax2.text(0.02, 0.98, 'Notice how BLUE drops below RED\nafter 5 kHz = noise being removed', 
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig


def main():
    """Main function to create before/after filter plots."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_filter_before_after.py /path/to/bundle_dir [--sweep SWEEP_NUM]")
        print("\nExample:")
        print("  python plot_filter_before_after.py /path/to/sub_131113          # Plot sweep 0")
        print("  python plot_filter_before_after.py /path/to/sub_131113 --sweep 5  # Plot sweep 5")
        sys.exit(1)
    
    bundle_dir = sys.argv[1]
    
    # Optional: specify which sweep to plot
    sweep_to_plot = 0
    if '--sweep' in sys.argv:
        idx = sys.argv.index('--sweep')
        if idx + 1 < len(sys.argv):
            sweep_to_plot = int(sys.argv[idx + 1])
    
    bundle_path = Path(bundle_dir)
    
    if not bundle_path.exists():
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("BEFORE/AFTER FILTER VISUALIZATION")
    print("="*70)
    print(f"Bundle: {bundle_path}")
    print(f"Plotting sweep {sweep_to_plot}")
    
    try:
        # Load voltage data
        print("\nLoading voltage (mV) data...")
        df_mv = load_parquet_data(bundle_dir, 'mV')
        
        # Get the sweep
        sweep_col = f'sweep_{sweep_to_plot}'
        if sweep_col not in df_mv.columns:
            print(f"ERROR: {sweep_col} not found in mV data")
            print(f"Available sweeps: {', '.join([c for c in df_mv.columns if c.startswith('sweep_')])}")
            sys.exit(1)
        
        sweep_mv_raw = df_mv[sweep_col].values.astype(np.float64)
        
        # Apply filter
        print("Applying 5 kHz Butterworth filter...")
        sweep_mv_filtered = filter_data(sweep_mv_raw)
        
        # Create plots
        print("Generating voltage trace comparison plots...")
        fig1 = plot_sweep_comparison(sweep_mv_raw, sweep_mv_filtered, sweep_to_plot, 
                                    title_prefix='VOLTAGE (mV)')
        output_dir = bundle_path / 'filter_visualizations'
        output_dir.mkdir(exist_ok=True)
        output_file1 = output_dir / f'voltage_before_after_sweep_{sweep_to_plot}.png'
        fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file1.name}")
        plt.close(fig1)
        
        print("Generating voltage frequency spectrum comparison plots...")
        fig2 = plot_frequency_comparison(sweep_mv_raw, sweep_mv_filtered, sweep_to_plot,
                                        title_prefix='VOLTAGE (mV)')
        output_file2 = output_dir / f'voltage_spectrum_before_after_sweep_{sweep_to_plot}.png'
        fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file2.name}")
        plt.close(fig2)
        
        # Load and plot current data
        print("\nLoading current (pA) data...")
        df_pa = load_parquet_data(bundle_dir, 'pA')
        
        sweep_pa_raw = df_pa[sweep_col].values.astype(np.float64)
        
        print("Applying 5 kHz Butterworth filter...")
        sweep_pa_filtered = filter_data(sweep_pa_raw)
        
        print("Generating current trace comparison plots...")
        fig3 = plot_sweep_comparison(sweep_pa_raw, sweep_pa_filtered, sweep_to_plot,
                                    title_prefix='CURRENT (pA)')
        output_file3 = output_dir / f'current_before_after_sweep_{sweep_to_plot}.png'
        fig3.savefig(output_file3, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file3.name}")
        plt.close(fig3)
        
        print("Generating current frequency spectrum comparison plots...")
        fig4 = plot_frequency_comparison(sweep_pa_raw, sweep_pa_filtered, sweep_to_plot,
                                        title_prefix='CURRENT (pA)')
        output_file4 = output_dir / f'current_spectrum_before_after_sweep_{sweep_to_plot}.png'
        fig4.savefig(output_file4, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file4.name}")
        plt.close(fig4)
        
        # Print summary
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE")
        print("="*70)
        print(f"\nCreated 4 plots for sweep {sweep_to_plot}:")
        print(f"  1. Voltage trace comparison (time domain)")
        print(f"  2. Voltage frequency spectrum (before vs after)")
        print(f"  3. Current trace comparison (time domain)")
        print(f"  4. Current frequency spectrum (before vs after)")
        print(f"\nAll files saved to: {output_dir}")
        print(f"\nWhat to look for:")
        print(f"  • Time domain: RED wiggly line (noise) vs BLUE smooth line (clean signal)")
        print(f"  • Frequency: RED line high above 5 kHz, BLUE line drops off")
        print(f"  • Zoomed window: Clear difference between noisy and filtered signal")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
