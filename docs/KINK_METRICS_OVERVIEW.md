## 🔬 Kink Detection Metrics (dV/dt Analysis)

This module detects and characterizes **“kinks”** in the spike upstroke using the derivative of voltage (**dV/dt**).

A **kink** is defined as a **secondary peak in dV/dt that occurs before the main upstroke peak**.

---

## 🧠 How Kink Detection Works (Per Spike)

For each detected spike:

### 1. Compute dV/dt
- Calculate the derivative of voltage over the upstroke window.

---

### 2. Find all peaks in dV/dt
- Use peak detection to identify **all local maxima** in the dV/dt signal.
- These include:
  - The **main upstroke peak** (largest peak)
  - Any **smaller bumps** (potential kinks or noise)

---

### 3. Identify the main upstroke
- The **maximum dV/dt value** is treated as the true upstroke.

---

### 4. Look only at peaks before the upstroke
- Kinks must occur **before** the main upstroke.
- All peaks after the upstroke are ignored.

---

### 5. Select ONE candidate kink
- Among all pre-upstroke peaks, we select:
  
  👉 **the largest (strongest) peak**

- Only this peak is tested as a potential kink.

> ⚠️ Important:  
> We do **NOT** test all peaks — only the strongest pre-upstroke peak.

---

### 6. Apply kink validation criteria

The selected peak must pass two checks:

#### ✅ Size (ratio threshold)
\[
\text{kink ratio} = \frac{\text{kink peak height}}{\text{main upstroke height}}
\]

- Must exceed a minimum threshold (e.g., 0.2)
- Filters out small noise fluctuations

---

#### ✅ Timing constraint
- Kink must occur within a small window before the upstroke (e.g., ≤ 1 ms)
- Prevents unrelated earlier bumps from being counted

---

### 7. Final classification

If both checks pass:

```text
has_kink = True