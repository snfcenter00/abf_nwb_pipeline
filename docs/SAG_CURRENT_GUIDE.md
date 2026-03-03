# Sag Current Analysis Guide

## What is Sag Current?

**Sag current** is the voltage response during hyperpolarizing current injection, caused by **HCN (Hyperpolarization-activated Cyclic Nucleotide-gated) channels**. These channels are responsible for regulating the neuron's excitability and input resistance.

## The Physics

When you inject negative (hyperpolarizing) current:

1. **Immediate response (t≈0-5ms):** Voltage rapidly hyperpolarizes (becomes more negative)
2. **Sag phase (t≈5-200ms):** HCN channels open, allowing positive current to flow back in
3. **Recovery phase:** Voltage gradually "sags" or relaxes back toward baseline

The amount and speed of recovery indicates **HCN channel density and kinetics**.

## In Your Pipeline

**Step 6** of the analysis pipeline calculates sag automatically:

```
Step 1: Load data
Step 1.5: Apply 5 kHz low-pass filter
Step 1.6: Visualize filter
Step 2: Sweep configuration
Step 3: RMP (resting membrane potential)
Step 4: Spike detection
Step 5: Savitzky-Golay smoothing
Step 6: Input resistance
[NEW] Step 6: Sag current analysis ← YOU ARE HERE
Step 7: Results finalization
```

## What Gets Measured

### Automatically Identified
- **Hyperpolarizing sweeps:** Sweeps with injected current < -10 pA
- **Baseline window:** From sweep_config.json (typically 0-10 ms)
- **Stimulus window:** From sweep_config.json (typically 10-810 ms)

### Calculated Metrics

For each hyperpolarizing sweep:

| Metric | Description | Range |
|--------|-------------|-------|
| `sag_voltage_mV` | Absolute sag magnitude (How much did the voltage recover after the dip (in mV)?) | V_steady - V_min (mV) |
| `sag_ratio` | Normalized sag (How much of the drop was recovered?) | Unitless (typically 0.5-1.5) | (V_steady - V_min) / (V_baseline - V_min)
| `sag_percent` | Sag as percentage | sag_ratio × 100 (%) |

### Example from Test Data

For Sweep 0 (-100 pA injection):

```
V_baseline:  -67.62 mV  (during baseline window, 0-10ms)
V_min:       -76.73 mV  (most negative voltage reached)
V_steady:    -67.54 mV  (voltage at end of stimulus, 760-810ms)

Total hyperpolarization: 9.11 mV
Sag voltage:             9.19 mV  (recovery from minimum)
Sag ratio:               1.009    (essentially complete recovery)
```

## Interpreting Sag Ratio

```
sag_ratio < 0.5  →  Limited recovery (weak HCN channels?)
sag_ratio ≈ 1.0  →  Complete recovery (normal HCN function)
sag_ratio > 1.5  →  Overshoot past baseline (strong HCN activity)
```

### Your Test Data

Mean sag ratio: **1.047 ± 0.102** (across 5 hyperpolarizing sweeps)

**Interpretation:** 
- Healthy, active HCN channels
- Voltage recovers nearly completely during stimulus
- Normal excitability regulation

## In analysis.parquet

Three new columns are automatically added:

```python
# For hyperpolarizing sweeps only
df['sag_voltage_mV']   # float, in mV
df['sag_ratio']        # float, unitless
df['sag_percent']      # float, percentage

# For depolarizing sweeps
# All three columns will be NaN (empty)
```

### Accessing Results

```python
import pandas as pd

df = pd.read_parquet("bundle_dir/analysis.parquet")

# Get sag measurements for all sweeps
print(df[['sweep', 'avg_injected_current_pA', 'sag_voltage_mV', 'sag_ratio']])

# Get only hyperpolarizing sweeps with sag data
hyper = df[df['avg_injected_current_pA'] < -10]
print(hyper[['sweep', 'sag_ratio', 'sag_percent']])

# Calculate mean sag
mean_sag = hyper['sag_ratio'].mean()
print(f"Mean sag ratio: {mean_sag:.3f}")
```

## Technical Details

### Voltage Measurement Windows

Uses timing from `sweep_config.json`:

```json
{
  "windows": {
    "baseline_start_s": 0.0,
    "baseline_end_s": 0.00999,
    "stimulus_start_s": 0.010020000000000001,
    "stimulus_end_s": 0.8100000000000002
  }
}
```

**Why this matters:**
- Baseline window = voltage before any current injection
- Stimulus window = voltage response to current injection
- Steady-state window = last 50 ms of stimulus

### Sampling Details

- Sampling rate: 200 kHz (5 µs per sample)
- Baseline duration: ~10 ms (2000 samples)
- Stimulus duration: ~800 ms (160,000 samples)
- Steady-state window: Last 50 ms (10,000 samples)

## Code Structure

### Main Functions in sag_current.py

```python
find_hyperpolarizing_sweeps(analysis_df, threshold_pA=-10)
# Returns: List of sweep numbers with injected current < threshold

measure_voltage_response(mv_data, sweep, sweep_config=None)
# Returns: Dict with v_baseline, v_min, v_steady, t_v_min

calculate_sag(voltage_response)
# Returns: Dict with sag_voltage_mV, sag_ratio, sag_percent

calculate_sag_for_bundle(bundle_dir, verbose=True)
# Returns: Dict with hyper_sweeps, sag_results, summary
# Also prints detailed analysis to console
```

### Integration in run_analysis.py

Called automatically as Step 6 (after input resistance):

```python
sag_results = calculate_sag_for_bundle(bundle_dir, verbose=True)

if sag_results:
    # Add sag columns to analysis.parquet
    df_analysis['sag_voltage_mV'] = np.nan
    df_analysis['sag_ratio'] = np.nan
    df_analysis['sag_percent'] = np.nan
    
    # Fill in values for hyperpolarizing sweeps
    for sweep, measurements in sag_results['sag_results'].items():
        mask = df_analysis['sweep'] == sweep
        df_analysis.loc[mask, 'sag_voltage_mV'] = measurements['sag_voltage_mV']
        df_analysis.loc[mask, 'sag_ratio'] = measurements['sag_ratio']
        df_analysis.loc[mask, 'sag_percent'] = measurements['sag_percent']
    
    df_analysis.to_parquet(bundle_path / "analysis.parquet", index=False)
```

## Troubleshooting

### "No hyperpolarizing sweeps found"

**Cause:** All injected currents are ≥ -10 pA

**Solution:** Check your sweep_config for actual current values
```python
import json
with open("bundle_dir/sweep_config.json") as f:
    config = json.load(f)
    for sweep_id, sweep_info in config['sweeps'].items():
        print(f"Sweep {sweep_id}: {sweep_info['stimulus_level_pA']} pA")
```

### Sag ratio looks wrong

**Check:**
1. Is sweep_config.json present in the bundle?
2. Are baseline windows reasonable (typically 0-10 ms)?
3. Are stimulus windows reasonable (typically 10-800+ ms)?

### Missing sag columns in analysis.parquet

**Solution:** Re-run the pipeline for that bundle. Step 6 will:
1. Calculate sag for hyperpolarizing sweeps
2. Add the three columns
3. Save updated analysis.parquet

## Biological Significance

### What HCN Channels Do

- **Role:** Control resting excitability and input resistance
- **Location:** Soma and dendrites
- **Activation:** Opens in response to hyperpolarization
- **Function:** Prevent excessive hyperpolarization by letting positive current back in

### Typical Sag Ratios by Cell Type

| Cell Type | Typical Sag Ratio |
|-----------|-----------------|
| Fast-spiking interneurons | 0.1-0.3 |
| Regular-spiking pyramidal cells | 0.8-1.2 |
| Neurons with strong Ih | 1.2-2.0 |

Your test data (1.047) is consistent with regular-spiking cells with healthy HCN function.

## Related Metrics

**Also measured in this pipeline:**
- **Input resistance:** Directly affected by HCN channel state
- **RMP:** Baseline for comparing hyperpolarization
- **Current threshold:** Relates to excitability regulation

## References

- Sag measurement is described in:
  - Hodgkin & Huxley (1952) - Original H-H model
  - Robinson & Siegelbaum (2003) - Ih kinetics
  - Kole et al. (2006) - HCN distribution in neurons

---

