---
title: GlucoSense AI
emoji: 🩸
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.35.0"
app_file: demo/app.py
pinned: false
license: mit
short_description: Predictive glucose monitoring — LSTM & TCN on CGM data
---

# diabetes-vision-monitor

> Predicting blood glucose 30 and 60 minutes ahead from continuous CGM sensor data, using LSTM and Temporal CNN architectures evaluated with clinical-grade metrics.

---

## Overview

Continuous Glucose Monitors (CGMs) measure blood glucose every 5 minutes, giving a real-time view of glycemic state. But a reading tells you where you *are* — not where you're *going*. For a Type 1 diabetic, a glucose of 90 mg/dL falling at 2 mg/dL/min at 2 AM is a medical emergency. A glucose of 90 mg/dL and rising is fine.

This project builds and compares machine learning models that predict future glucose values at 30-minute and 60-minute horizons from recent CGM history. It is the backend inference engine for the **GlucoSense AI** concept — a predictive glucose monitoring system designed for wearables.

Models are evaluated using the **Clarke Error Grid** (EGA), the standard clinical metric used to assess whether a glucose prediction would lead to a safe, benign, or dangerous treatment decision. Numerical metrics alone (RMSE, MAE) miss this: a prediction that is 25 mg/dL off in the safe zone is clinically irrelevant; the same error during hypoglycemia could cause a dangerous overcorrection.

---

## Demo

**Live demo:** [glucosense-ai on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_HF_USERNAME/glucosense-ai) ← replace after deploy

**Run locally:**
```bash
# 1. Generate synthetic CGM data
python data/sample/generate_synthetic.py

# 2. Launch the Streamlit dashboard
streamlit run demo/app.py
```

The demo includes:
- Live glucose dashboard with 30/60-min predictions and trend arrow
- 24-hour trace with meal event markers and projections
- Clarke Error Grid visualization
- Nocturnal hypoglycemia alert (Smart Alert screen)

---

## Architecture

```
Input: CGM window (24 readings × 5 min = 2-hour context)
         ↓
┌─────────────────────────────────────────┐
│  Preprocessing                          │
│  • Median filter (sensor noise)         │
│  • Linear interpolation (short gaps)    │
│  • MinMax normalization (per-patient)   │
└───────────────┬─────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│  Model (LSTM or Temporal CNN)             │
│                                           │
│  LSTM:  2-layer bidirectional LSTM        │
│         → last hidden state → FC(2)       │
│                                           │
│  TCN:   3 dilated causal residual blocks  │
│         (dilation 1, 2, 4)                │
│         → last timestep → FC(2)           │
└───────────────┬───────────────────────────┘
                ↓
Output: [glucose_30min, glucose_60min] (mg/dL)
```

The TCN uses dilated causal convolutions so that:
1. No future timesteps leak into past predictions (causality)
2. Receptive field grows exponentially with depth, capturing long-range trends cheaply

---

## Dataset

**OhioT1DM** — Marling & Bunescu, 2018  
8 Type 1 diabetic patients, CGM readings at 5-minute intervals over 8+ weeks.  
Includes: CGM glucose, meal events, bolus/basal insulin, exercise, sleep.

Access: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html (free for academic use)

For development and demo, a synthetic dataset is included that simulates realistic CGM patterns including postprandial spikes, nocturnal hypoglycemia, and exercise-induced drops.

| Split | % | Notes |
|-------|---|-------|
| Train | 70% | Used for scaler fitting and model training |
| Validation | 15% | Early stopping and learning rate scheduling |
| Test | 15% | Held out — evaluated once per model |

---

## Results

Trained on **real OhioT1DM data** — 6 patients, 12 XML files (training + testing splits per patient), 5-min CGM readings.  
Patient-level 70/15/15 train/val/test split. Shared scaler fit on training data only. Evaluated on held-out test windows.

| Model | MARD 30-min | RMSE 30-min | MAE 30-min | MARD 60-min | RMSE 60-min | Zone A (30-min) | Zone B (30-min) |
|-------|-------------|-------------|------------|-------------|-------------|-----------------|-----------------|
| **Linear baseline** | **11.0%** | **23.2 mg/dL** | **16.5 mg/dL** | **18.2%** | **35.1 mg/dL** | **85.7%** | **13.8%** |
| Temporal CNN | 17.0% | 35.3 mg/dL | 27.0 mg/dL | 33.1% | 66.3 mg/dL | 72.6% | 26.7% |
| LSTM | 24.7% | 47.6 mg/dL | 39.0 mg/dL | 26.3% | 50.6 mg/dL | 45.1% | 45.4% |

*Train: 22,202 windows · Val: 4,425 · Test: 4,440. Device: Apple MPS.*

**Why does the linear baseline win here?**  
CGM data has very strong short-range autocorrelation — glucose rarely changes faster than ~2 mg/dL/min, so a learned linear extrapolation already captures most of the signal. With only ~8 weeks of data per patient (small by deep learning standards), the neural networks overfit before learning the underlying physiology. The proper fix is per-patient fine-tuning, more patients, or auxiliary features (insulin, meals, activity), none of which are in this version. This is a known finding in the glucose prediction literature — see Oviedo et al. (2017) for a survey.

**Clarke Error Grid zones:**
- **Zone A** (clinically accurate): target >95%
- **Zone B** (benign errors): acceptable
- **Zone C/D/E**: clinically dangerous — minimise

---

## How to Run

```bash
pip install -r requirements.txt

# Generate synthetic data (skip if using OhioT1DM)
python data/sample/generate_synthetic.py --days 30

# Train a single model
python train.py --config configs/training_config.yaml --model lstm

# Run ablation (all models)
python train.py --config configs/training_config.yaml --model all

# Launch demo
streamlit run demo/app.py
```

---

## Key Implementation Notes

- **Clarke Error Grid** is implemented from scratch following Clarke et al. (1987) — not imported from a library. This makes the zone logic auditable and transparent.
- **Causal convolutions**: the TCN pads only the left side of each convolution, guaranteeing that no future information leaks into the prediction. This is the correct implementation for streaming inference on a wearable device.
- **Per-patient normalization**: the scaler is fit only on training data and applied to val/test, preventing data leakage. In deployment, it would be fit on the first N days of a new patient's CGM data.
- **MARD over MAPE**: MARD is the glucose-specific analogue of MAPE, standard in the CGM accuracy literature (ISO 15197). Reporting both allows comparison with sensor accuracy benchmarks.
- **Experiment tracking**: set `use_wandb: true` in `configs/training_config.yaml` to log all runs to Weights & Biases.

---

## Limitations & Future Work

**Current limitations:**
- Trained on synthetic or limited data — performance on real OhioT1DM patients will differ
- No meal/insulin/activity features used — adding these as auxiliary inputs would significantly improve postprandial prediction
- Single-horizon architecture; a seq2seq decoder would give full glucose curves rather than point predictions at fixed horizons

**What I'd do next:**
- Add meal event flags as auxiliary input (OhioT1DM includes meal timestamps)
- Implement uncertainty quantification (MC Dropout or conformal prediction intervals) — critical for clinical use, where the model should express doubt rather than be confidently wrong
- Test on real OhioT1DM subjects across different glycemic control profiles
- Investigate whether personalization (fine-tuning a population model on individual data) closes the gap for outlier patients

---

## References

- Clarke, W. et al. (1987). Evaluating clinical accuracy of systems for self-monitoring of blood glucose. *Diabetes Care*, 10(5):622–628.
- Marling, C. & Bunescu, R. (2018). The OhioT1DM Dataset for Blood Glucose Level Prediction. *KHD Workshop, IJCAI*.
- Bai, S. et al. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
- Oviedo, S. et al. (2017). A review of personalized blood glucose prediction strategies for T1DM patients. *IJBC*.
