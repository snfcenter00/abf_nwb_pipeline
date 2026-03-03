"""
Pre-processing low-pass filter module

This module provides a butterworth low-pass filter implementation
to remove high-frequency noise from electrophysiology recordings
before analysis begins.

The filter operates on raw voltage and current data immediately
after loading, before any spike detection or analysis steps.
"""

import pandas as pd
import numpy as np
from scipy import signal
import json
from pathlib import Path


def apply_butterworth_lowpass(data_array: np.ndarray, 
                              sampling_rate: float,
                              cutoff_hz: float = 5000,
                              order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to remove high-frequency noise.
    
    Args:
        data_array: Input signal (1D numpy array)
        sampling_rate: Sampling rate in Hz
        cutoff_hz: Cutoff frequency in Hz (default 5000 Hz = 5 kHz)
        order: Filter order (default 4)
        
    Returns:
        Filtered signal as numpy array
        
    Notes:
        - Butterworth filters have a flat passband (no ripple)
        - Cutoff frequency is -3dB point
        - Higher order = steeper rolloff but more phase distortion
        - Order 4 is a good compromise between steepness and stability
    """
    # Design the filter
    # Normalize frequency: cutoff_hz / (sampling_rate / 2)
    nyquist = sampling_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"Cutoff frequency ({cutoff_hz} Hz) must be less than Nyquist frequency ({nyquist} Hz)")
    
    normalized_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    # Apply filter using forward-backward filtering (filtfilt)
    # This gives zero phase distortion and doubles the filter order
    filtered_data = signal.filtfilt(b, a, data_array)
    
    return filtered_data


def filter_sweep_data(df_sweep: pd.DataFrame, 
                      sampling_rate: float,
                      cutoff_hz: float = 5000) -> pd.DataFrame:
    """
    Apply low-pass filter to a single sweep's data.
    
    Args:
        df_sweep: DataFrame with columns like 'value' for one sweep
        sampling_rate: Sampling rate in Hz
        cutoff_hz: Cutoff frequency in Hz
        
    Returns:
        DataFrame with filtered 'value' column
    """
    df_filtered = df_sweep.copy()
    
    if len(df_sweep) > 0:
        filtered_values = apply_butterworth_lowpass(
            df_sweep['value'].values,
            sampling_rate,
            cutoff_hz
        )
        df_filtered['value'] = filtered_values
    
    return df_filtered


def apply_lowpass_filter_to_bundle(bundle_dir: str, 
                                   cutoff_hz: float = 5000,
                                   inplace: bool = True,
                                   verbose: bool = False) -> dict:
    """
    Apply low-pass filter to voltage and current data in a bundle.
    
    This function reads the raw parquet files, applies a butterworth
    low-pass filter to remove high-frequency noise, and optionally
    overwrites the original files.
    
    Args:
        bundle_dir: Path to the bundle directory
        cutoff_hz: Cutoff frequency in Hz (default 5000)
        inplace: If True, overwrites parquet files. If False, returns filtered data only.
        verbose: Print progress information
        
    Returns:
        Dictionary with:
        - 'df_mv': Filtered voltage dataframe
        - 'df_pa': Filtered current dataframe
        - 'cutoff_hz': Cutoff frequency used
        - 'n_sweeps_mv': Number of sweeps in voltage data
        - 'n_sweeps_pa': Number of sweeps in current data
        
    Notes:
        - Expects 'manifest.json' in bundle_dir
        - Modifies files in-place if inplace=True (recommended for pipeline)
        - Returns filtered data for inspection if inplace=False
    """
    p = Path(bundle_dir)
    
    # Load manifest to find parquet file paths
    manifest_path = p / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {bundle_dir}")
    
    man = json.loads(manifest_path.read_text())
    
    # Get sampling rates
    meta = man.get("meta", {})
    if isinstance(meta.get("rate"), list):
        # Multiple rates - use max for now (will handle per-sweep later if needed)
        fs_mv = float(meta.get("rate")[0]) if meta.get("rate") else 200000.0
        fs_pa = float(meta.get("rate")[0]) if meta.get("rate") else 200000.0
    else:
        fs_mv = float(meta.get("rate", 200000.0))
        fs_pa = float(meta.get("rate", 200000.0))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOW-PASS FILTER PRE-PROCESSING")
        print(f"{'='*70}")
        print(f"Cutoff frequency: {cutoff_hz} Hz ({cutoff_hz/1000:.1f} kHz)")
        print(f"Voltage sampling rate: {fs_mv} Hz")
        print(f"Current sampling rate: {fs_pa} Hz")
    
    # Load voltage data
    mv_path = p / man["tables"]["mv"]
    if verbose:
        print(f"\nLoading voltage data from {mv_path.name}...")
    df_mv = pd.read_parquet(mv_path)
    n_sweeps_mv = df_mv['sweep'].nunique()
    
    if verbose:
        print(f"  Loaded: {len(df_mv):,} samples, {n_sweeps_mv} sweeps")
        print(f"  Filtering each sweep individually...")
    
    # Apply filter to each sweep separately
    df_mv_filtered = df_mv.copy()
    for sweep_id in df_mv['sweep'].unique():
        mask = df_mv['sweep'] == sweep_id
        df_sweep = df_mv[mask]
        
        filtered_vals = apply_butterworth_lowpass(
            df_sweep['value'].values,
            fs_mv,
            cutoff_hz
        )
        df_mv_filtered.loc[mask, 'value'] = filtered_vals
        
        if verbose and sweep_id % max(1, n_sweeps_mv // 10) == 0:
            print(f"    Progress: Sweep {sweep_id}/{n_sweeps_mv}...")
    
    if verbose:
        print(f"  ✓ Voltage filtering complete")
    
    # Load current data
    pa_path = p / man["tables"]["pa"]
    if verbose:
        print(f"\nLoading current data from {pa_path.name}...")
    df_pa = pd.read_parquet(pa_path)
    n_sweeps_pa = df_pa['sweep'].nunique()
    
    if verbose:
        print(f"  Loaded: {len(df_pa):,} samples, {n_sweeps_pa} sweeps")
        print(f"  Filtering each sweep individually...")
    
    # Apply filter to each sweep separately
    df_pa_filtered = df_pa.copy()
    for sweep_id in df_pa['sweep'].unique():
        mask = df_pa['sweep'] == sweep_id
        df_sweep = df_pa[mask]
        
        filtered_vals = apply_butterworth_lowpass(
            df_sweep['value'].values,
            fs_pa,
            cutoff_hz
        )
        df_pa_filtered.loc[mask, 'value'] = filtered_vals
        
        if verbose and sweep_id % max(1, n_sweeps_pa // 10) == 0:
            print(f"    Progress: Sweep {sweep_id}/{n_sweeps_pa}...")
    
    if verbose:
        print(f"  ✓ Current filtering complete")
    
    # Save filtered data back to parquet if inplace=True
    if inplace:
        if verbose:
            print(f"\nSaving filtered data...")
        df_mv_filtered.to_parquet(mv_path, index=False)
        df_pa_filtered.to_parquet(pa_path, index=False)
        if verbose:
            print(f"  ✓ Filtered voltage saved to {mv_path.name}")
            print(f"  ✓ Filtered current saved to {pa_path.name}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ LOW-PASS FILTER COMPLETE")
        print(f"{'='*70}\n")
    
    return {
        'df_mv': df_mv_filtered,
        'df_pa': df_pa_filtered,
        'cutoff_hz': cutoff_hz,
        'n_sweeps_mv': n_sweeps_mv,
        'n_sweeps_pa': n_sweeps_pa,
        'fs_mv': fs_mv,
        'fs_pa': fs_pa
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python lowpass_filter.py <bundle_directory> [cutoff_hz]")
        print("\nExample:")
        print("  python lowpass_filter.py /path/to/bundle")
        print("  python lowpass_filter.py /path/to/bundle 5000")
        sys.exit(1)
    
    bundle_dir = sys.argv[1]
    cutoff_hz = float(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    # Apply filter with verbose output
    result = apply_lowpass_filter_to_bundle(bundle_dir, cutoff_hz, inplace=True, verbose=True)
    print(f"Result: Filtered {result['n_sweeps_mv']} voltage sweeps and {result['n_sweeps_pa']} current sweeps")
