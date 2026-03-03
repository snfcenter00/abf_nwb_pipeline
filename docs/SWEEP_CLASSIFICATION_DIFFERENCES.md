## 📊 Sweep Classifier Differences: ABF vs NWB

This section describes how sweep classification differs between **ABF** and **NWB** data formats.  
The classifier adapts its strictness based on the expected data quality of each format.

---

### 🧪 ABF Files

ABF recordings are typically **noisier**, so classification is intentionally more **lenient**.

#### Behavior

- **Lenient filtering**
  - Designed to tolerate experimental noise

- **Square wave stimulus check**
  - Uses a relaxed threshold (~70%)

- **No low-amplitude filtering**
  - Sweeps are accepted even at very low stimulus levels (e.g., 0–5 pA)

- **No flatness check**
  - Baseline flatness is not enforced due to noise

- **Voltage artifact detection**
  - Removes sweeps with clear non-physiological artifacts:
    - Sharp corners in voltage trace   
    - “Right-angle” shaped signals  

---

### 🧬 NWB Files

NWB recordings are generally **cleaner and standardized**, so stricter checks are applied.

#### Behavior

- **Square wave stimulus check**
  - Ensures proper stimulus structure

- **Flatness check (~90%)**
  - Requires stable baseline before stimulus onset

- **Voltage artifact detection**
  - Same as ABF:
    - Sharp corners  

- **No low-amplitude filtering**
  - Small-amplitude stimuli are still accepted

---

### ⚖️ Summary

| Feature                   | ABF Files        | NWB Files        |
|--------------------------|-----------------|-----------------|
| Overall strictness       | Lenient         | Stricter        |
| Square wave check        | ~70% threshold  | Standard        |
| Flatness check           | ❌ Not applied   | ✅ ~90% required |
| Low-amplitude filtering  | ❌ Not applied   | ❌ Not applied   |
| Artifact detection       | ✅ Yes           | ✅ Yes           |

---

### 💡 Rationale

- **ABF** → Noisy experimental data → more tolerant classification  
- **NWB** → Clean, structured data → stricter validation  

---
