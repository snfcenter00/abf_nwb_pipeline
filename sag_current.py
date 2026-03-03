"""
Calculate sag current from hyperpolarizing current sweeps.

Sag is the voltage response during hyperpolarizing current injection,
caused by HCN (hyperpolarization-activated cyclic nucleotide-gated) channels.

Theory:
    When negative current is injected:
    1. Voltage initially hyperpolarizes (becomes more negative)
    2. Over time, HCN channels open, allowing positive current to flow back in
    3. Voltage "sags" or relaxes back toward less negative values
    4. The amount of sag indicates HCN channel activity

Measurements:
    - Sag voltage (mV): V_steady_state - V_min
    - Sag ratio (dimensionless): sag_voltage / (V_baseline - V_min)
      * 0 = no sag (no HCN channels)
      * 1 = complete relaxation back to baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path


def find_hyperpolarizing_sweeps(analysis_parquet: pd.DataFrame, threshold_pA: float = 0) -> list:
    """
    Identify sweeps with negative (hyperpolarizing) current injection.
    
    Args:
        analysis_parquet: DataFrame from analysis.parquet with 'avg_injected_current_pA' column
        threshold_pA: Only include sweeps with current < this value (default: 0 pA)
    
    Returns:
        List of sweep numbers with hyperpolarizing current
    """
    hyper_sweeps = analysis_parquet[
        analysis_parquet['avg_injected_current_pA'] < threshold_pA
    ]['sweep'].tolist()
    
    return sorted(hyper_sweeps)


def measure_voltage_response(
    mv_data: pd.DataFrame,
    sweep: int,
    sweep_config: dict = None,
    sampling_rate: float = 200000,  # Hz
    steady_state_window: float = 0.050  # Last 50 ms (steady state)
) -> dict:
    """
    Measure key voltage points during a hyperpolarizing sweep.
    
    Uses sweep_config.json windows if available for accurate baseline and stimulus timing.
    
    Args:
        mv_data: DataFrame with columns ['sweep', 't_s', 'value'] (voltage in mV)
        sweep: Sweep number to analyze
        sweep_config: Dict from sweep_config.json (optional, will auto-load if None)
        sampling_rate: Sampling rate in Hz
        steady_state_window: Duration of window for steady-state measurement (seconds)
    
    Returns:
        Dictionary with:
        - 'v_baseline': Mean voltage during baseline window (mV)
        - 'v_min': Most negative voltage reached during stimulus (mV)
        - 'v_steady': Mean voltage during steady-state window (mV)
        - 't_v_min': Time when V_min was reached (seconds)
    """
    sweep_data = mv_data[mv_data['sweep'] == sweep].copy()
    
    if len(sweep_data) == 0:
        return None
    
    times = sweep_data['t_s'].values
    voltages = sweep_data['value'].values
    
    # Get baseline and stimulus windows from sweep_config
    if sweep_config is None:
        # Try to auto-load from bundle
        sweep_config = {}  # Will use defaults below
    
    sweep_str = str(int(sweep))
    if sweep_str in sweep_config:
        windows = sweep_config[sweep_str].get('windows', {})
        baseline_start = windows.get('baseline_start_s', 0.0)
        baseline_end = windows.get('baseline_end_s', 0.01)
        stimulus_start = windows.get('stimulus_start_s', 0.01)
        stimulus_end = windows.get('stimulus_end_s', None)
    else:
        # Fallback to defaults
        baseline_start = 0.0
        baseline_end = 0.01
        stimulus_start = 0.01
        stimulus_end = None
    
    # Extract baseline voltage
    baseline_mask = (times >= baseline_start) & (times <= baseline_end)
    if baseline_mask.sum() > 0:
        v_baseline = np.mean(voltages[baseline_mask])
    else:
        v_baseline = np.mean(voltages[:int(0.01 * sampling_rate)])
    
    # Extract stimulus phase voltage
    stimulus_mask = times >= stimulus_start
    if stimulus_mask.sum() > 0:
        stimulus_voltages = voltages[stimulus_mask]
        stimulus_times = times[stimulus_mask]
        
        # Minimum voltage during stimulus
        v_min = np.min(stimulus_voltages)
        t_v_min = stimulus_times[np.argmin(stimulus_voltages)]
        
        # Steady-state: last 50 ms of stimulus
        steady_start = stimulus_times[-1] - steady_state_window
        steady_mask = stimulus_times >= steady_start
        v_steady = np.mean(stimulus_voltages[steady_mask])
    else:
        return None
    
    return {
        'v_baseline': v_baseline,
        'v_min': v_min,
        'v_steady': v_steady,
        't_v_min': t_v_min,
    }


def calculate_sag(voltage_response: dict) -> dict:
    """
    Calculate sag metrics from voltage response measurements.
    
    Args:
        voltage_response: Dict from measure_voltage_response()
    
    Returns:
        Dictionary with:
        - 'sag_voltage_mV': Absolute sag (V_steady - V_min, in mV)
        - 'sag_ratio': Normalized sag (0-1)
                      0 = no recovery, 1 = complete recovery
        - 'sag_percent': Sag as percentage of total hyperpolarization
    """
    if voltage_response is None:
        return None
    
    v_baseline = voltage_response['v_baseline']
    v_min = voltage_response['v_min']
    v_steady = voltage_response['v_steady']
    
    # Total hyperpolarization (how far voltage dropped)
    total_hyperpol = v_baseline - v_min
    
    # Sag voltage (how much it recovered)
    sag_voltage = v_steady - v_min
    
    # Sag ratio (what fraction of the drop did it recover from?)
    if total_hyperpol != 0:
        sag_ratio = sag_voltage / total_hyperpol
    else:
        sag_ratio = 0
    
    # Sag as percentage
    sag_percent = sag_ratio * 100
    
    return {
        'sag_voltage_mV': sag_voltage,
        'sag_ratio': sag_ratio,
        'sag_percent': sag_percent,
        'total_hyperpol_mV': total_hyperpol,
        'v_baseline_mV': v_baseline,
        'v_min_mV': v_min,
        'v_steady_mV': v_steady,
    }


def calculate_sag_for_bundle(
    bundle_dir: str,
    verbose: bool = True
) -> dict:
    """
    Calculate sag for all hyperpolarizing sweeps in a bundle.
    
    Args:
        bundle_dir: Path to bundle directory
        verbose: Print progress information
    
    Returns:
        Dictionary with results:
        - 'hyper_sweeps': List of hyperpolarizing sweep numbers
        - 'sag_results': Dict mapping sweep_num → sag measurements
        - 'mean_sag': Mean sag ratio across all hyperpolarizing sweeps
        - 'summary': Summary statistics
    """
    bundle_path = Path(bundle_dir)
    
    # Find parquet files
    mv_files = list(bundle_path.rglob("mV_*.parquet"))
    analysis_files = list(bundle_path.rglob("analysis.parquet"))
    sweep_config_files = list(bundle_path.rglob("sweep_config.json"))
    
    if not mv_files or not analysis_files:
        if verbose:
            print(f"⚠ Missing parquet files in {bundle_dir}")
        return None
    
    # Load data
    mv_data = pd.read_parquet(mv_files[0])
    analysis_data = pd.read_parquet(analysis_files[0])
    
    # Load sweep_config if available
    sweep_config = {}
    if sweep_config_files:
        import json
        with open(sweep_config_files[0], 'r') as f:
            config_data = json.load(f)
            if 'sweeps' in config_data:
                sweep_config = config_data['sweeps']
    
    # Find hyperpolarizing sweeps
    hyper_sweeps = find_hyperpolarizing_sweeps(analysis_data)
    
    if verbose:
        print(f"\n[Sag Current Analysis]")
        print(f"  Found {len(hyper_sweeps)} hyperpolarizing sweeps: {hyper_sweeps}")
    
    # Measure sag for each hyperpolarizing sweep
    sag_results = {}
    sag_ratios = []
    
    for sweep in hyper_sweeps:
        voltage_response = measure_voltage_response(mv_data, sweep, sweep_config=sweep_config)
        if voltage_response is not None:
            sag_measurements = calculate_sag(voltage_response)
            sag_results[sweep] = sag_measurements
            sag_ratios.append(sag_measurements['sag_ratio'])
            
            if verbose:
                current = analysis_data[analysis_data['sweep'] == sweep]['avg_injected_current_pA'].iloc[0]
                print(f"\n  Sweep {sweep} ({current:.0f} pA):")
                print(f"    V_baseline: {sag_measurements['v_baseline_mV']:.2f} mV")
                print(f"    V_min:      {sag_measurements['v_min_mV']:.2f} mV")
                print(f"    V_steady:   {sag_measurements['v_steady_mV']:.2f} mV")
                print(f"    Sag voltage: {sag_measurements['sag_voltage_mV']:.2f} mV")
                print(f"    Sag ratio:   {sag_measurements['sag_ratio']:.3f} ({sag_measurements['sag_percent']:.1f}%)")
    
    # Summary statistics
    if sag_ratios:
        mean_sag = np.mean(sag_ratios)
        std_sag = np.std(sag_ratios)
        summary = {
            'n_sweeps': len(sag_results),
            'mean_sag_ratio': mean_sag,
            'std_sag_ratio': std_sag,
            'min_sag_ratio': np.min(sag_ratios),
            'max_sag_ratio': np.max(sag_ratios),
        }
    else:
        summary = None
    
    if verbose and summary:
        print(f"\n  ─── SUMMARY ───")
        print(f"  Mean sag ratio: {summary['mean_sag_ratio']:.3f} ± {summary['std_sag_ratio']:.3f}")
        print(f"  Range: {summary['min_sag_ratio']:.3f} - {summary['max_sag_ratio']:.3f}")
    
    return {
        'hyper_sweeps': hyper_sweeps,
        'sag_results': sag_results,
        'summary': summary,
    }


def add_sag_to_analysis_parquet(
    bundle_dir: str,
    sag_results: dict = None
) -> pd.DataFrame:
    """
    Add sag measurements to the analysis parquet file.
    
    Args:
        bundle_dir: Path to bundle directory
        sag_results: Output from calculate_sag_for_bundle() (if None, will calculate)
    
    Returns:
        Updated analysis DataFrame with new columns:
        - 'sag_voltage_mV': Sag magnitude (mV)
        - 'sag_ratio': Sag as fraction of hyperpolarization
        - 'sag_percent': Sag as percentage
    """
    bundle_path = Path(bundle_dir)
    analysis_files = list(bundle_path.rglob("analysis.parquet"))
    
    if not analysis_files:
        print(f"⚠ No analysis.parquet found in {bundle_dir}")
        return None
    
    analysis_data = pd.read_parquet(analysis_files[0])
    
    # Calculate sag if not provided
    if sag_results is None:
        sag_results = calculate_sag_for_bundle(bundle_dir, verbose=False)
    
    # Add sag columns (NaN for non-hyperpolarizing sweeps)
    analysis_data['sag_voltage_mV'] = np.nan
    analysis_data['sag_ratio'] = np.nan
    analysis_data['sag_percent'] = np.nan
    
    # Fill in values for hyperpolarizing sweeps
    for sweep, measurements in sag_results['sag_results'].items():
        mask = analysis_data['sweep'] == sweep
        analysis_data.loc[mask, 'sag_voltage_mV'] = measurements['sag_voltage_mV']
        analysis_data.loc[mask, 'sag_ratio'] = measurements['sag_ratio']
        analysis_data.loc[mask, 'sag_percent'] = measurements['sag_percent']
    
    # Optionally save back to file
    # analysis_data.to_parquet(analysis_files[0], index=False)
    
    return analysis_data


if __name__ == "__main__":
    # Test on test_bundle2
    bundle_dir = "test_bundle2/sub-131113"
    results = calculate_sag_for_bundle(bundle_dir, verbose=True)
    
    if results:
        print("\n" + "="*70)
        print("Results can be integrated into analysis pipeline")
        print("="*70)
