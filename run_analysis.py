import json
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis import resting_vm_per_sweep, attach_manifest_to_analysis
from spike_detection_new import run_spike_detection
from sav_gol_filter import run_sav_gol
from input_resistance import get_input_resistance
from lowpass_filter import apply_lowpass_filter_to_bundle
from sag_current import calculate_sag_for_bundle

# Set to True to enable verbose/debug output in terminal
VERBOSE = False


def visualize_filter_all_sweeps(bundle_dir: str, skip_plots: bool = False, max_sweeps: int = 4):
    """
    Create before/after filter visualizations for all (or selected) sweeps in a bundle.
    
    Args:
        bundle_dir: Path to bundle directory
        skip_plots: If True, skip visualization (for faster processing)
        max_sweeps: Maximum number of sweeps to visualize (default 12 for good coverage)
                   Set to None to visualize all sweeps
    """
    if skip_plots:
        return
    
    try:
        import subprocess
        from pathlib import Path
        
        bundle_path = Path(bundle_dir)
        
        # Get list of parquet files
        mv_files = list(bundle_path.rglob("mV_*.parquet"))
        if not mv_files:
            return
        
        # Count sweeps
        df = pd.read_parquet(mv_files[0])
        
        if 'sweep' in df.columns:
            n_sweeps = int(df['sweep'].max()) + 1
        else:
            n_sweeps = len(df.columns)
        
        # Determine how many to plot
        if max_sweeps is None:
            sweeps_to_plot = n_sweeps
        else:
            sweeps_to_plot = min(max_sweeps, n_sweeps)
        
        print(f"  Generating before/after filter visualizations...")
        print(f"  Creating plots for {sweeps_to_plot} sweeps (of {n_sweeps} total)...")
        
        # Get current working directory to find plot script
        script_path = Path(__file__).parent / "plot_filter_before_after.py"
        if not script_path.exists():
            print(f"  ⚠ plot_filter_before_after.py not found, skipping visualizations")
            return
        
        for sweep_num in range(sweeps_to_plot):
            try:
                # Run the visualization script
                cmd = [
                    "python",
                    str(script_path),
                    str(bundle_dir),
                    "--sweep", str(sweep_num)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"  ✓ Sweep {sweep_num}")
                else:
                    if VERBOSE:
                        print(f"  ⚠ Sweep {sweep_num} failed: {result.stderr[:100]}")
            except subprocess.TimeoutExpired:
                print(f"  ⚠ Sweep {sweep_num} timed out")
            except Exception as e:
                if VERBOSE:
                    print(f"  ⚠ Error with sweep {sweep_num}: {e}")
        
        # Print summary
        viz_dir = bundle_path / "filter_visualizations"
        if viz_dir.exists():
            n_plots = len(list(viz_dir.glob("*.png")))
            print(f"  ✓ {n_plots} visualization files created")
            print(f"    Location: {viz_dir}")
        
    except Exception as e:
        if VERBOSE:
            print(f"  ⚠ Could not generate filter visualizations: {e}")


def checkpoint_with_resume(stage_name: str, bundle_dir: str = None) -> bool:
    """
    Checkpoint that prompts user to continue or pause for inspection.
    Allows seamless resume within the same pipeline execution.
    
    Args:
        stage_name: Description of the completed stage
        bundle_dir: Path to bundle (for displaying file location)
        
    Returns:
        True to proceed, False to pause (but stay in function for resume)
    """
    print("\n" + "="*70)
    print(f"✓ CHECKPOINT: {stage_name}")
    print("="*70)
    if bundle_dir:
        print(f"Bundle: {Path(bundle_dir).name}")
    print("\nYou can now inspect the results before proceeding.")
    if bundle_dir:
        print(f"Location: {bundle_dir}")
    
    response = input("\nContinue to next step? (y/n): ").strip().lower()
    return response == 'y'


def detect_hardware_malfunction(bundle_dir: str):
    """
    Detect if hardware malfunction occurred: both channels recorded as mV (empty pA).
    
    Args:
        bundle_dir: Path to the bundle
    
    Returns:
        True if malfunction detected (empty pA), False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    
    try:
        df_pa = pd.read_parquet(p / man["tables"]["pa"])
        # Malfunction if pA is empty or has very few data points
        return len(df_pa) == 0 or df_pa.shape[0] < 100
    except:
        return False

def fix_hardware_malfunction_mV(bundle_dir: str):
    """
    When hardware malfunction occurs, two mV channels are recorded (correct + nonsense).
    This function identifies and keeps only the correct mV channel by checking signal stability.
    The correct channel should have consistent morphology across sweeps.
    The nonsense channel will have random/inconsistent data.
    
    Args:
        bundle_dir: Path to the bundle
    
    Returns:
        True if fix successful, False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    
    try:
        mv_path = p / man["tables"]["mv"]
        df_mv = pd.read_parquet(mv_path)
        
        # Check if there are multiple channels
        if "channel_index" not in df_mv.columns:
            return False
        
        channels = df_mv["channel_index"].unique()
        if len(channels) != 2:
            return False
        
        if VERBOSE: print(f"  Detected {len(channels)} mV channels. Identifying the correct one...")
        
        # For each channel, calculate variance across sweeps
        # The correct channel should have consistent patterns (lower variance in peak detection)
        # The nonsense channel will have random data (higher variance)
        
        channel_stats = {}
        for ch in channels:
            df_ch = df_mv[df_mv["channel_index"] == ch]
            
            # Group by sweep and calculate signal statistics
            sweep_stats = df_ch.groupby("sweep")["value"].agg(["mean", "std", "min", "max", "count"])
            
            # Calculate coefficient of variation (std / mean) - indicator of signal consistency
            # Nonsense data will have very high CV
            cv_per_sweep = sweep_stats["std"] / (sweep_stats["mean"].abs() + 1e-6)
            avg_cv = cv_per_sweep.mean()
            
            channel_stats[ch] = {
                "avg_cv": avg_cv,
                "mean_std": sweep_stats["std"].mean(),
                "data_points": len(df_ch)
            }
            
            if VERBOSE: print(f"    Channel {ch}: CV={avg_cv:.4f}, Mean Std={sweep_stats['std'].mean():.4f}, Points={len(df_ch)}")
        
        # Select channel with HIGHER CV (more variable = correct channel with real signal)
        # The nonsense channel will have near-zero CV (flat noise or constant value)
        # The correct channel will have natural signal variation (higher CV)
        correct_channel = max(channel_stats.keys(), key=lambda x: channel_stats[x]["avg_cv"])
        
        if VERBOSE: print(f"  ✓ Selected Channel {correct_channel} as correct signal (highest CV)")
        
        # Keep only correct channel
        df_mv_fixed = df_mv[df_mv["channel_index"] == correct_channel].copy()
        
        # Save back
        df_mv_fixed.to_parquet(mv_path, index=False)
        print(f"  ✓ Saved corrected mV data to {mv_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR fixing mV data: {e}")
        return False

def is_current_data_valid(bundle_dir: str, sweep_config: Optional[dict] = None):
    """
    Check if current data exists in the expected stimulus time window.
    
    IMPORTANT: For MIXED PROTOCOL files only:
    sweep_config.json uses RELATIVE times per sweep (0-27s)
    but the parquet files use ABSOLUTE times (all sweeps concatenated, e.g., 278-1856s)
    This function converts relative times to absolute times for mixed protocol files.
    
    For SINGLE PROTOCOL files, times in sweep_config and parquet match directly.
    
    Args:
        bundle_dir: Path to the bundle
        sweep_config: Dict from sweep_config.json with stimulus windows (optional, tries to load if None)

    Returns:
        True if valid current data exists, False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    df_pa = pd.read_parquet(p / man["tables"]["pa"])
    
    # Detect if mixed protocol
    is_mixed = "stimulus" in man["tables"] and "response" in man["tables"]
    
    # Determine time window from sweep_config or use first 10% of data
    if sweep_config is not None:
        try:
            first_valid_sweep_id = None
            t_min_relative = None
            t_max_relative = None
            
            for sweep_id_str, sweep_data in sweep_config.get("sweeps", {}).items():
                if sweep_data.get("valid", False):
                    first_valid_sweep_id = int(sweep_id_str)
                    t_min_relative = sweep_data["windows"].get("stimulus_start_s", 0.1)
                    t_max_relative = sweep_data["windows"].get("stimulus_end_s", 0.75)
                    break
            
            if first_valid_sweep_id is not None:
                # For mixed protocol: sweep_config contains ABSOLUTE times (from NWB file)
                # For single protocol: sweep_config contains RELATIVE times (within each sweep)
                if is_mixed:
                    # Mixed protocol: use absolute times directly from sweep_config
                    t_min = t_min_relative  # Actually absolute times, misnamed variable
                    t_max = t_max_relative  # Actually absolute times, misnamed variable
                else:
                    # Single protocol: use relative times directly
                    t_min = t_min_relative
                    t_max = t_max_relative
            else:
                # No valid sweeps found: use first 10% of data
                t_min = df_pa["t_s"].min()
                t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
        except (KeyError, TypeError):
            # If sweep_config lookup fails, use first 10% of data
            t_min = df_pa["t_s"].min()
            t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
    else:
        # No sweep_config: use first 10% of data for validation
        t_min = df_pa["t_s"].min()
        t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
    
    df_pa_filtered = df_pa[(df_pa["t_s"] >= t_min) & (df_pa["t_s"] <= t_max)]
    return len(df_pa_filtered) > 0


def replace_current_data_with_reference(bundle_dir: str, reference_bundle_dir: str, sweep_config: Optional[dict] = None):
    """
    Replace the VALUES inside the faulty pA parquet file with values from a reference bundle.
    
    Crucially: The reference data sweep numbers are remapped to match the target bundle's sweep numbers,
    since both recordings use the same protocol but may have different sweep numbering.
    The TARGET FILENAME is preserved (e.g., pa_660.parquet stays pa_660.parquet).
    
    Args:
        bundle_dir: Path to the bundle with faulty current data (e.g., pa_660.parquet)
        reference_bundle_dir: Path to the reference bundle with good current data (e.g., pa_668.parquet)
    """
    p = Path(bundle_dir)
    p_ref = Path(reference_bundle_dir)
    
    # Load manifests
    man = json.loads((p / "manifest.json").read_text())
    man_ref = json.loads((p_ref / "manifest.json").read_text())
    
    # Get the pA parquet file paths
    pa_table_name = man["tables"]["pa"]  # e.g., "pa_660.parquet" (target filename to keep)
    pa_ref_table_name = man_ref["tables"]["pa"]  # e.g., "pa_668.parquet" (source)
    
    pa_ref_path = p_ref / pa_ref_table_name
    pa_path = p / pa_table_name  # Target path (keep this filename)
    
    # Load BOTH current datasets
    df_pa_faulty = pd.read_parquet(pa_path)  # Target (faulty) dataset
    df_pa_ref = pd.read_parquet(pa_ref_path)  # Source (reference) dataset
    
    # Get unique sweep numbers from each
    target_sweeps = sorted(df_pa_faulty["sweep"].unique())
    ref_sweeps = sorted(df_pa_ref["sweep"].unique())

    # If the faulty pA file has no sweeps (empty), we'll write the reference sweeps
    # into the target filename and use the reference sweep numbering.
    if len(target_sweeps) == 0:
        if VERBOSE: print("  Note: Faulty pA contains no sweeps. Will write reference sweeps into target file.")
        target_sweeps = list(ref_sweeps)

    if VERBOSE:
        print(f"  Target sweeps: {len(target_sweeps)} sweeps (e.g., {target_sweeps[:5]}...)")
        print(f"  Reference sweeps: {len(ref_sweeps)} sweeps (e.g., {ref_sweeps[:5]}...)")

    # Create mapping by position: map the first N reference sweeps to the first N target sweeps
    # If counts differ, map up to the smaller length and drop any unmapped reference rows.
    n_map = min(len(ref_sweeps), len(target_sweeps))
    if n_map == 0:
        raise ValueError("Reference or target pA has no sweeps to map")

    if len(ref_sweeps) != len(target_sweeps):
        print(f"  WARNING: Sweep count mismatch (ref={len(ref_sweeps)} vs target={len(target_sweeps)}). Mapping first {n_map} sweeps.")

    sweep_mapping = {ref_sweeps[i]: target_sweeps[i] for i in range(n_map)}

    # Remap reference data to target sweep numbers
    df_pa_ref_remapped = df_pa_ref.copy()
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].map(sweep_mapping)

    # Drop rows that could not be mapped (NaN sweep) to avoid NaN sweep ids
    before_rows = len(df_pa_ref_remapped)
    df_pa_ref_remapped = df_pa_ref_remapped.dropna(subset=["sweep"]).copy()
    after_rows = len(df_pa_ref_remapped)
    if after_rows < before_rows:
        if VERBOSE: print(f"  Note: Dropped {before_rows - after_rows} reference rows that could not be mapped to target sweeps.")

    # Ensure sweep is integer type
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].astype(int)
    # Preview summary and ask for confirmation before overwriting target file
    if VERBOSE:
        print("\n--- Preview replacement ---")
        print(f"Target (will keep filename): {pa_table_name} -> {pa_path}")
        print(f"Source (reference): {pa_ref_table_name} from {p_ref}")
        print(f"Remapped rows: {len(df_pa_ref_remapped)} (from {len(df_pa_ref)} source rows)")
        print(f"Target sweeps (post-map) sample: {sorted(df_pa_ref_remapped['sweep'].unique())[:8]}")
        print("First 5 rows of remapped reference data:")
        try:
            print(df_pa_ref_remapped.head().to_string())
        except Exception:
            print(df_pa_ref_remapped.head())

    # Apply baseline offset correction + per-sweep averaging + rounding to 5 pA increments
    try:
        import numpy as _np
        # Step 1: Calculate baseline offset during quiet period (pre-stimulus period, no injection)
        # Use sweep_config if available, otherwise use first 10% of recording
        if sweep_config:
            try:
                # Find first sweep and get its stimulus start time
                for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        t_stim_start = sweep_data["windows"].get("stimulus_start_s", 0.1)
                        break
                baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < t_stim_start]
                if VERBOSE: print(f"Using stimulus start time from sweep_config: {t_stim_start:.6f}s")
            except (KeyError, TypeError, StopIteration):
                # Fallback to first 10% if sweep_config extraction fails
                t_max = df_pa_ref_remapped['t_s'].max()
                baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < (t_max * 0.1)]
                if VERBOSE: print(f"Using fallback: first 10% of recording (up to {t_max * 0.1:.6f}s)")
        else:
            # No sweep_config: use first 10% of recording as baseline
            t_max = df_pa_ref_remapped['t_s'].max()
            baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < (t_max * 0.1)]
            if VERBOSE: print(f"No sweep_config provided: using first 10% of recording (up to {t_max * 0.1:.6f}s) as baseline")
        
        baseline_offset = baseline_window['value'].mean() if len(baseline_window) > 0 else 0.0
        if VERBOSE: print(f"\nBaseline offset (pre-stimulus quiet period): {baseline_offset:.2f} pA")

        # Step 2: Subtract baseline offset from all values
        df_pa_ref_remapped['value'] = df_pa_ref_remapped['value'] - baseline_offset

        # Step 3: Compute mean current in the stimulus window per sweep (after offset correction)
        # Again, use sweep_config if available
        if sweep_config:
            try:
                t_stim_start = None
                t_stim_end = None
                for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        windows = sweep_data["windows"]
                        t_stim_start = windows.get("stimulus_start_s", 0.1)
                        t_stim_end = windows.get("stimulus_end_s", 0.75)
                        break
                if t_stim_start is not None and t_stim_end is not None:
                    df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= t_stim_start) & (df_pa_ref_remapped['t_s'] <= t_stim_end)]
                    if VERBOSE: print(f"Using stimulus window from sweep_config: [{t_stim_start:.6f}, {t_stim_end:.6f}]s")
                else:
                    raise KeyError("Could not extract stimulus window")
            except (KeyError, TypeError):
                # Fallback to 0.1-0.75 if extraction fails
                df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= 0.1) & (df_pa_ref_remapped['t_s'] <= 0.75)]
                if VERBOSE: print("Using fallback stimulus window: [0.1, 0.75]s")
        else:
            # No sweep_config: use middle 50% of recording
            t_min = df_pa_ref_remapped['t_s'].min()
            t_max = df_pa_ref_remapped['t_s'].max()
            t_window_min = t_min + (t_max - t_min) * 0.2
            t_window_max = t_min + (t_max - t_min) * 0.7
            df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= t_window_min) & (df_pa_ref_remapped['t_s'] <= t_window_max)]
            if VERBOSE: print(f"No sweep_config: using middle 50% of recording [{t_window_min:.6f}, {t_window_max:.6f}]s")
        
        avg_pa = df_window.groupby('sweep')['value'].mean().reset_index(name='avg_injected_current_pA')
        # if some sweeps missing in window, fallback to full-sweep mean
        if avg_pa['sweep'].nunique() < df_pa_ref_remapped['sweep'].nunique():
            fallback = df_pa_ref_remapped.groupby('sweep')['value'].mean().reset_index(name='avg_injected_current_pA')
            avg_pa = avg_pa.set_index('sweep').combine_first(fallback.set_index('sweep')).reset_index()

        # Step 4: Round to nearest 5 pA (or 0)
        avg_pa['avg_injected_current_pA_rounded'] = (_np.round(avg_pa['avg_injected_current_pA'] / 5) * 5).astype(float)

        # Step 5: Apply rounded mean to all rows in each sweep
        for _, row in avg_pa.iterrows():
            sw = int(row['sweep'])
            rounded_val = float(row['avg_injected_current_pA_rounded'])
            df_pa_ref_remapped.loc[df_pa_ref_remapped['sweep'] == sw, 'value'] = rounded_val

        if VERBOSE:
            print('\nApplied baseline correction + per-sweep mean and rounded to 5 pA increments (preview):')
            print(avg_pa.head().to_string())
    except Exception as _e:
        print(f"Warning: could not apply per-sweep rounding to remapped data: {_e}")

    # Auto-yes replacement
    if VERBOSE: print("Auto-proceeding with pA replacement...")

    # Save remapped reference data to the TARGET filename, replacing the faulty values
    df_pa_ref_remapped.to_parquet(pa_path, index=False)
    print(f"✓ Replaced VALUES in {pa_table_name} (kept original filename)")
    print(f"  Source: {pa_ref_table_name} from {p_ref}")
    print(f"  Destination: {pa_path}")
    print(f"  Sweep remapping applied for {n_map} sweeps")


from sweep_classifier import process_bundle as classify_bundle_sweeps
from sweep_classifier import process_bundle_abf as classify_bundle_sweeps_abf


def generate_summary_plot(bundle_dir: str):
    """
    Collect all JPEG/PNG plot files from a bundle directory and combine them
    into a single master summary image.
    
    Gathers plots from:
    - AP_Per_Sweep/ (action potential per sweep)
    - Averaged_Peaks_Per_Sweep/ (averaged peaks)
    - SavGol_Plots/ (Savitzky-Golay filter)
    - InputResistance.jpeg
    - RMP_Dist_Post_Filter.png
    - Any combined plots already generated
    """
    try:
        from PIL import Image
    except ImportError:
        print("  WARNING: Pillow not installed. Skipping summary plot.")
        print("  Install with: pip install Pillow")
        return
    
    p = Path(bundle_dir)
    
    # Collect all image files in a structured order
    image_paths = []
    labels = []
    
    # 1. Combined AP plot (if exists) or individual AP plots
    combined_ap = p / "AP_Per_Sweep_combined.png"
    if combined_ap.exists():
        image_paths.append(combined_ap)
        labels.append("Action Potentials (All Sweeps)")
    else:
        ap_dir = p / "AP_Per_Sweep"
        if ap_dir.exists():
            for f in sorted(ap_dir.glob("AP_sweep_*.jpeg")):
                image_paths.append(f)
                sweep_num = f.stem.replace("AP_sweep_", "")
                labels.append(f"AP Sweep {sweep_num}")
    
    # 2. Combined Averaged Peaks plot (if exists) or individual ones
    combined_avg = p / "Averaged_Peaks_Per_Sweep_combined.png"
    if combined_avg.exists():
        image_paths.append(combined_avg)
        labels.append("Averaged Peaks (All Sweeps)")
    else:
        avg_dir = p / "Averaged_Peaks_Per_Sweep"
        if avg_dir.exists():
            for f in sorted(avg_dir.glob("averaged_peaks_for_sweep_*.jpeg")):
                image_paths.append(f)
                sweep_num = f.stem.replace("averaged_peaks_for_sweep_", "")
                labels.append(f"Avg Peaks Sweep {sweep_num}")
    
    # 3. SavGol filter plots
    savgol_dir = p / "SavGol_Plots"
    if savgol_dir.exists():
        for f in sorted(savgol_dir.glob("SavGol_Sweep*.png")):
            image_paths.append(f)
            sweep_id = f.stem.replace("SavGol_Sweep", "").replace("_baseline", "")
            labels.append(f"SavGol Sweep {sweep_id}")
    
    # 4. RMP distribution post-filter
    rmp_plot = p / "RMP_Dist_Post_Filter.png"
    if rmp_plot.exists():
        image_paths.append(rmp_plot)
        labels.append("RMP Distribution")
    
    # 5. Input Resistance
    ir_dir = p / "Input_Resistance"
    ir_plot = None
    if ir_dir.exists():
        ir_candidates = list(ir_dir.glob("InputResistance.jpeg"))
        if ir_candidates:
            ir_plot = ir_candidates[0]
    if ir_plot is None:
        # Check root of bundle
        ir_root = p / "InputResistance.jpeg"
        if ir_root.exists():
            ir_plot = ir_root
    if ir_plot:
        image_paths.append(ir_plot)
        labels.append("Input Resistance")
    
    if not image_paths:
        print("  No plot files found to combine.")
        return
    
    print(f"  Combining {len(image_paths)} plots into summary...")
    
    # Load all images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"  WARNING: Could not open {img_path.name}: {e}")
    
    if not images:
        return
    
    # Create grid layout
    n_plots = len(images)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(label, fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"Analysis Summary: {p.name}", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    summary_path = p / "analysis_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Close PIL images
    for img in images:
        img.close()
    
    print(f"  [OK] Saved master summary plot: {summary_path.name}")


def load_sweep_config(bundle_dir: str):
    """
    Load sweep_config.json if it exists, otherwise run sweep classification
    to detect current injection windows and generate one.
    
    For ABF bundles: Uses consistent window across all sweeps, keeps all sweeps
    For NWB bundles: Uses per-sweep window detection with validation
    
    Args:
        bundle_dir: Path to bundle directory
    
    Returns:
        dict: sweep_config (loaded or generated)
    """
    p = Path(bundle_dir)
    config_path = p / "sweep_config.json"
    
    # Check if this is an ABF bundle (has abf_path in manifest)
    manifest_path = p / "manifest.json"
    is_abf_bundle = False
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        is_abf_bundle = "abf_path" in manifest
    
    if config_path.exists():
        if VERBOSE: print(f"✓ Loading sweep_config.json from {p.name}")
        with open(config_path) as f:
            config = json.load(f)
        
        # Check if this was generated by the old crude auto-generation method
        # If so, re-run proper classification for accurate stimulus detection
        first_sweep = next(iter(config.get("sweeps", {}).values()), None)
        reason = first_sweep.get("reason", "") if first_sweep else ""
        reason = reason or ""  # Handle None value
        if first_sweep and "auto-generated" in reason:
            print(f"⚠ Found outdated auto-generated sweep_config in {p.name}")
            print("  Re-running sweep classifier for accurate stimulus window detection...")
            config_path.unlink()  # Delete old config
            
            # Use ABF-specific classifier for ABF bundles
            if is_abf_bundle:
                sweep_config = classify_bundle_sweeps_abf(bundle_dir, plot_sweeps=True)
            else:
                sweep_config = classify_bundle_sweeps(bundle_dir)
            
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
            return sweep_config
        
        # For ABF bundles, check if we need to regenerate with consistent window
        if is_abf_bundle and not config.get("consistent_window", False):
            print(f"⚠ ABF bundle has per-sweep windows, regenerating with consistent window...")
            config_path.unlink()
            sweep_config = classify_bundle_sweeps_abf(bundle_dir, plot_sweeps=True)
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
            return sweep_config
        
        return config
    else:
        print(f"⚠ No sweep_config.json found in {p.name}")
        print("  Running sweep classifier to detect current injection windows...")
        
        # Use ABF-specific classifier for ABF bundles
        if is_abf_bundle:
            sweep_config = classify_bundle_sweeps_abf(bundle_dir, plot_sweeps=True)
        else:
            sweep_config = classify_bundle_sweeps(bundle_dir)
        
        # Reload from the file that process_bundle wrote
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return sweep_config


def run_for_bundle(bundle_dir: str, reference_bundle_dir: str = None, skip_plots: bool = False):
    p = Path(bundle_dir)
    pA_was_replaced = False  # Track if pA data was replaced
    
    print(f"\n{'='*70}")
    print(f"[Analysis] Starting analysis pipeline for bundle")
    print(f"{'='*70}")
    print(f"Bundle: {p.name}")
    
    # STEP 0: Load sweep_config early so we can use it for data processing
    sweep_config = load_sweep_config(bundle_dir)
    
    # STEP 1: Check for hardware malfunction (empty pA, 2 mV channels)
    if detect_hardware_malfunction(bundle_dir):
        print(f"\n⚠ HARDWARE MALFUNCTION DETECTED in {bundle_dir}")
        print("  Both channels recorded as voltage (empty current data).")
        
        # Step 1a: Fix mV data
        if VERBOSE: print("\n>>> Fixing voltage data: extracting correct mV channel...")
        if fix_hardware_malfunction_mV(bundle_dir):
            print("  ✓ Voltage data fixed")
        else:
            print("  ✗ Failed to fix voltage data")
        
        # Step 1b: Replace pA with reference
        if VERBOSE: print("\n>>> Replacing empty current data with reference recording...")
        
        if reference_bundle_dir is None:
            print("    No reference recording provided - skipping current data replacement")
            reference_bundle_dir = ""  # Auto-skip
        
        if reference_bundle_dir:
            try:
                replace_current_data_with_reference(bundle_dir, reference_bundle_dir, sweep_config)
                print("  ✓ Current data replaced")
                pA_was_replaced = True  # Mark that pA was replaced
            except Exception as e:
                print(f"  ✗ ERROR: Failed to load reference: {e}")
                print("    Proceeding with NaN current values...")
        else:
            print("    Proceeding without current data (input resistance analysis will have NaN)...")
        print()
    
    # STEP 2: Check for invalid current data (non-malfunction case)
    elif not is_current_data_valid(bundle_dir, sweep_config):
        print(f"\n⚠ WARNING: No valid current data found in {bundle_dir}")
        print("  Current data is required for accurate input resistance analysis.")
        
        # If no reference provided, auto-skip
        if reference_bundle_dir is None:
            print("\n>>> No reference recording provided - skipping current data replacement")
            reference_bundle_dir = ""  # Auto-skip
        
        if reference_bundle_dir:
            try:
                if VERBOSE: print(f"\n>>> Replacing faulty current data with reference recording...")
                replace_current_data_with_reference(bundle_dir, reference_bundle_dir, sweep_config)
                print()
            except Exception as e:
                print(f"✗ ERROR: Failed to load reference: {e}")
                print("  Proceeding with NaN current values...")
                print()
        else:
            print("  Proceeding without current data (results will have NaN for current-based metrics)...")
            print()
    else:
        print(f"\n✓ Current data looks valid in {bundle_dir} and no malfunction detected.\n")
    
    print(f"\n[Step 1] Loading data files...")
    man = json.loads((p / "manifest.json").read_text())

    # Ensure meta is a valid dict
    if not man.get("meta"):
        print(f"\n✗ ERROR: manifest.json in {bundle_dir} has no metadata.")
        print(f"  This bundle may need to be re-created. Skipping.")
        return

    # load tables
    print(f"  Loading voltage (mV) data...")
    df_mv = pd.read_parquet(p / man["tables"]["mv"])
    print(f"  ✓ Voltage: {df_mv.shape[0]:,} samples, {df_mv['sweep'].nunique()} sweeps")
    
    print(f"  Loading current (pA) data...")
    df_pa = pd.read_parquet(p / man["tables"]["pa"])
    print(f"  ✓ Current: {df_pa.shape[0]:,} samples, {df_pa['sweep'].nunique()} sweeps")
    
    # STEP 2: Apply low-pass filter (5 kHz) and generate sweep configuration
    print(f"\n[Step 2] Sweep configuration & filtering...")
    print(f"  Applying 5 kHz low-pass filter (pre-processing)...")
    try:
        filter_result = apply_lowpass_filter_to_bundle(bundle_dir, cutoff_hz=5000, inplace=True, verbose=False)
        print(f"  ✓ Low-pass filter applied (5 kHz cutoff)")
        print(f"    - Filtered {filter_result['n_sweeps_mv']} voltage sweeps")
        print(f"    - Filtered {filter_result['n_sweeps_pa']} current sweeps")
        
        # Reload the filtered data
        df_mv = pd.read_parquet(p / man["tables"]["mv"])
        df_pa = pd.read_parquet(p / man["tables"]["pa"])
    except Exception as e:
        print(f"  ⚠ WARNING: Low-pass filter failed: {e}")
        print(f"  Proceeding with unfiltered data...")
    
    print(f"  Generating filter visualizations...")
    visualize_filter_all_sweeps(bundle_dir, skip_plots=skip_plots)
    print(f"  ✓ Sweep configuration & filtering complete")
    
    # Ensure sweep_config is a dict (None when no sweep_config.json exists)
    if sweep_config is None:
        sweep_config = {}
    
    # Auto-skip the pause prompt for automated pipeline (no interactive input)
    # Check if we're running in non-interactive mode
    import sys
    is_interactive = sys.stdin.isatty()
    
    if is_interactive:
        # Pause/resume loop for sweep config inspection (interactive mode only)
        while True:
            response = input("\nContinue to resting membrane potential calculation? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
    else:
        # Auto-proceed in non-interactive mode
        print("\n[Auto] Proceeding with analysis (non-interactive mode)...")
    
    print(f"\n[Step 3] Resting membrane potential calculation...")
    # Filter to only kept sweeps for all analysis
    kept_sweeps = sweep_config.get("kept_sweeps", [])
    
    # If no kept_sweeps defined, use all available sweeps
    if not kept_sweeps:
        kept_sweeps = sorted(df_mv["sweep"].unique().tolist())
        if VERBOSE:
            print(f"\n>>> No sweep filter defined - using all {len(kept_sweeps)} sweeps")
    elif VERBOSE:
        print(f"\n>>> Filtering to kept sweeps: {len(kept_sweeps)} sweeps")
    
    # Filter all dataframes to only include kept sweeps
    df_mv_kept = df_mv[df_mv["sweep"].isin(kept_sweeps)].copy()
    df_pa_kept = df_pa[df_pa["sweep"].isin(kept_sweeps)].copy()
    
    if VERBOSE:
        print(f"    mV data: {len(df_mv_kept)} rows (from {len(df_mv)})")
        print(f"    pA data: {len(df_pa_kept)} rows (from {len(df_pa)})")
    
    # sweep_config was already loaded at the beginning of this function
    print(f"  Calculating resting membrane potential...")
    df_vm_per_sweep = resting_vm_per_sweep(df_mv_kept, sweep_config, bundle_dir)  # one row per sweep, columns like resting_vm_mean_mV
    combined_mean = float(df_vm_per_sweep["resting_vm_mean_mV"].mean())
    print(f"  ✓ Mean resting Vm: {combined_mean:.2f} mV")

    # save analysis outputs
    out_parq = p / "analysis.parquet"
    out_csv  = p / "analysis.csv"
    df_vm_per_sweep.to_parquet(out_parq, index=False)
    df_vm_per_sweep.to_csv(out_csv, index=False)

    # update manifest with analysis pointers (non-destructive)
    man.setdefault("analysis", {})
    man["analysis"]["resting_vm_table"] = out_parq.name
    man["analysis"]["resting_vm_mean"]  = combined_mean
    (p / "manifest.json").write_text(json.dumps(man, indent=2))
    
    print(f"  Saved to: {out_parq.name}")
    
    # Skip interactive pause in non-interactive mode
    if is_interactive:
        # Pause/resume loop for RMP inspection
        while True:
            response = input("\nContinue to spike detection? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
        print(f"\n[Step 4] Spike detection")
    else:
        print(f"\n[Step 4] Spike detection")

    # spike detection
    print(f"  ⚡ Detecting action potentials...")
    # CRITICAL: Reload pA from disk to pick up replaced data (if malfunction was fixed above)
    df_pa_kept = pd.read_parquet(p / man["tables"]["pa"])
    df_pa_kept = df_pa_kept[df_pa_kept["sweep"].isin(kept_sweeps)].copy()
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    fs = man["meta"]["sampleRate_Hz"]
    
    run_spike_detection(df_mv_kept, df_pa_kept, df_analysis, fs, bundle_dir, 
                       pA_was_replaced=pA_was_replaced, sweep_config=sweep_config,
                       skip_plots=skip_plots)
    if VERBOSE: print("Spike detection was successful")
    #After running above line, analysis.parquet and analysis.csv and manifest.json will be updated
    
    print(f"  ✓ Spike detection complete")
    
    if is_interactive:
        # Pause/resume loop for spike detection inspection
        while True:
            response = input("\nContinue to Savitzky-Goyal filtering? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
        print(f"\n[Step 5] Savitzky-Golay filtering")
    else:
        print(f"\n[Step 5] Savitzky-Golay filtering")

    #low pass filter
    print(f"  🔄 Applying Savitzky-Golay low-pass filter...")
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    run_sav_gol(df_mv_kept, df_analysis, fs, bundle_dir, sweep_config=sweep_config, skip_plots=skip_plots)
    if VERBOSE: print("Running Sav Gol was successful")
    
    print(f"  ✓ Savitzky-Golay filtering complete")
    
    if is_interactive:
        # Pause/resume loop for SavGol inspection
        while True:
            response = input("\nContinue to input resistance calculation? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
        print(f"\n[Step 6] Input resistance calculation")
    else:
        print(f"\n[Step 6] Input resistance calculation")
    
    #input resistance
    print(f"  ⚡ Computing input resistance...")
    # Reuse df_pa_kept from spike detection (pA data unchanged between steps)
    get_input_resistance(df_mv_kept, df_pa_kept, bundle_dir, sweep_config=sweep_config, skip_plots=skip_plots)
    if VERBOSE: print("Getting input resistance was successful")
    #After running above line, manifest.json will be updated

    print(f"  ✓ Input resistance calculation complete")
    
    if is_interactive:
        # Pause/resume loop for input resistance inspection
        while True:
            response = input("\nContinue to sag current analysis? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")

    # STEP 6: Sag current analysis (HCN channel characterization)
    print(f"\n[Step 6] Sag current analysis (HCN channels)...")
    print(f"  📊 Computing sag current from hyperpolarizing sweeps...")
    
    sag_results = calculate_sag_for_bundle(bundle_dir, verbose=True)
    
    if sag_results:
        # Add sag measurements to analysis.parquet
        df_analysis = pd.read_parquet(p / "analysis.parquet")
        
        # Initialize columns with NaN
        df_analysis['sag_voltage_mV'] = np.nan
        df_analysis['sag_ratio'] = np.nan
        df_analysis['sag_percent'] = np.nan
        
        # Fill in values for hyperpolarizing sweeps
        for sweep, measurements in sag_results['sag_results'].items():
            mask = df_analysis['sweep'] == sweep
            df_analysis.loc[mask, 'sag_voltage_mV'] = measurements['sag_voltage_mV']
            df_analysis.loc[mask, 'sag_ratio'] = measurements['sag_ratio']
            df_analysis.loc[mask, 'sag_percent'] = measurements['sag_percent']
        
        # Save updated analysis.parquet
        df_analysis.to_parquet(p / "analysis.parquet", index=False)
        
        print(f"  ✓ Sag current analysis complete")
        
        # STEP 7: After sag calculation
        print(f"\n{'='*70}")
        print("✓ STEP 7: SAG CURRENT ANALYSIS COMPLETE")
        print("="*70)
        if sag_results['summary']:
            print(f"Hyperpolarizing sweeps analyzed: {len(sag_results['hyper_sweeps'])}")
            print(f"Mean sag ratio: {sag_results['summary']['mean_sag_ratio']:.3f} ± {sag_results['summary']['std_sag_ratio']:.3f}")
            print(f"Sag columns added to analysis.parquet:")
            print(f"  - sag_voltage_mV: Absolute sag magnitude (mV)")
            print(f"  - sag_ratio: Sag as fraction of hyperpolarization (≈1.0 = complete recovery)")
            print(f"  - sag_percent: Sag as percentage")
        print()
    else:
        print(f"  ⚠ No hyperpolarizing sweeps found - skipping sag analysis")
    
    if is_interactive:
        # Pause/resume loop for sag inspection
        while True:
            response = input("\nContinue to finalize results? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")

    print(f"\n{'='*70}")
    print("📊 Finalizing results...")
    print("="*70)

    #attach manifest details to analysis results
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    attach_manifest_to_analysis(bundle_dir, df_analysis)
    if VERBOSE:
        print("Adding to analysis was successful")
        print(f"All updates completed and successful for {bundle_dir}.")

    # Generate master summary plot combining all figures
    if not skip_plots:
        print("🖼️  Generating summary plots...")
        generate_summary_plot(bundle_dir)
    
    # FINAL CHECKPOINT: Pipeline complete
    print(f"\n{'='*70}")
    print("✅ ANALYSIS PIPELINE COMPLETE!")
    print("="*70)
    print(f"All analysis steps completed successfully for: {p.name}")
    print(f"Results saved to: {bundle_dir}")
    print(f"\n📁 Output files:")
    print(f"  - analysis.parquet: Complete results table")
    print(f"  - analysis.csv: Exported results (CSV format)")
    print(f"  - sweep_config.json: Sweep metadata and timing windows")
    print(f"  - Individual PNG/PDF plots: In {bundle_dir}")
    print()

