"""
Kink detection module for identifying pre-upstroke peaks in spike upstrokes.

Improved version:
- Anchors to main upstroke (max dV/dt)
- Only considers peaks BEFORE main peak
- Uses stronger prominence threshold
- Filters by kink-to-main ratio
- Applies temporal constraint (kink must be close to upstroke)
"""

import numpy as np
from scipy.signal import find_peaks


# -----------------------------
# Configuration
# -----------------------------
KINK_DETECTION_PROMINENCE_PERCENT = 0.1   # 10% of max dV/dt
KINK_DETECTION_MIN_DISTANCE_SAMPLES = 5   # Increase separation
KINK_RATIO_THRESHOLD = 0.2                # Kink must be ≥20% of main peak
KINK_MAX_TIME_BEFORE_PEAK_MS = 1.0        # Kink must occur within 1 ms before peak


# -----------------------------
# Peak detection in dV/dt
# -----------------------------
def find_peaks_in_dvdt(dvdt_array, prominence_percent=KINK_DETECTION_PROMINENCE_PERCENT):
    if len(dvdt_array) < 3:
        return []

    max_dvdt = np.max(dvdt_array)
    if max_dvdt <= 0:
        return []

    min_prominence = prominence_percent * max_dvdt

    peaks, properties = find_peaks(
        dvdt_array,
        prominence=min_prominence,
        distance=KINK_DETECTION_MIN_DISTANCE_SAMPLES
    )

    return peaks, properties


# -----------------------------
# Kink metric computation
# -----------------------------
def measure_kink_metrics(dvdt_array, times_array):
    """
    Measure kink metrics using improved detection logic.
    
    Kink = secondary dV/dt peak BEFORE max upstroke (max dV/dt).
    """

    result = {
        'num_peaks_in_upstroke': 0,
        'kink_interval_ms': np.nan,
        'kink_ratio': np.nan,
        'has_kink': False
    }

    if len(dvdt_array) < 3:
        return result

    # --- Step 1: Identify max upstroke (TRUE anchor) ---
    max_upstroke_idx = np.argmax(dvdt_array)
    max_upstroke_height = dvdt_array[max_upstroke_idx]

    if max_upstroke_height <= 0:
        return result

    # --- Step 2: Find candidate peaks ---
    peaks, properties = find_peaks_in_dvdt(dvdt_array)

    if len(peaks) == 0:
        return result

    result['num_peaks_in_upstroke'] = len(peaks)

    # --- Step 3: Only consider peaks BEFORE max upstroke ---
    pre_peaks = [p for p in peaks if p < max_upstroke_idx]

    if len(pre_peaks) == 0:
        return result

    # --- Step 4: Select strongest pre-upstroke peak ---
    kink_idx = max(pre_peaks, key=lambda p: dvdt_array[p])
    kink_height = dvdt_array[kink_idx]

    # --- Step 5: Ratio filter ---
    kink_ratio = kink_height / max_upstroke_height
    if kink_ratio < KINK_RATIO_THRESHOLD:
        return result

    # --- Step 6: Temporal constraint (relative to upstroke) ---
    kink_time = times_array[kink_idx]
    upstroke_time = times_array[max_upstroke_idx]

    time_diff_ms = (kink_time - upstroke_time) * 1000  # should be negative

    if time_diff_ms < -KINK_MAX_TIME_BEFORE_PEAK_MS:
        return result

    # --- Step 7: Compute metrics ---
    kink_interval_ms = abs(time_diff_ms)

    result.update({
        'kink_interval_ms': kink_interval_ms,
        'kink_ratio': kink_ratio,
        'has_kink': True
    })

    return result


# -----------------------------
# Wrapper for full spike
# -----------------------------
def measure_kink_for_spike(voltages, times, peak_idx, pre_threshold_samples):
    """
    Measure kink metrics for a single spike.
    """

    w1_start_idx = max(0, peak_idx - pre_threshold_samples)
    w1_end_idx = peak_idx + 1

    if w1_start_idx >= w1_end_idx:
        return {
            'num_peaks_in_upstroke': 0,
            'kink_interval_ms': np.nan,
            'kink_ratio': np.nan,
            'has_kink': False
        }

    v_up = voltages[w1_start_idx:w1_end_idx]
    t_up = times[w1_start_idx:w1_end_idx]

    if len(t_up) < 2:
        return {
            'num_peaks_in_upstroke': 0,
            'kink_interval_ms': np.nan,
            'kink_ratio': np.nan,
            'has_kink': False
        }

    # Compute dV/dt (no smoothing per request)
    dvdt_up = np.gradient(v_up, t_up) * 1000  # mV/ms

    t_peak = times[peak_idx]

    return measure_kink_metrics(dvdt_up, t_up)