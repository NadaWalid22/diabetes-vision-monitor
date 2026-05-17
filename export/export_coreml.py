"""
Export the trained GlucoSense LSTM to Core ML for on-device inference on Apple Watch.

Pipeline:
    PyTorch (.pth) → TorchScript → Core ML (.mlpackage)

Requirements:
    pip install coremltools onnx

Usage:
    python export/export_coreml.py --model lstm --checkpoint models/checkpoints/lstm_best.pth
    python export/export_coreml.py --model temporal_cnn --checkpoint models/checkpoints/temporal_cnn_best.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import GlucoseLSTM, GlucoseTemporalCNN


def export(model_type: str, checkpoint: str, output_dir: str = "export/") -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load trained model ────────────────────────────────────────────
    if model_type == "lstm":
        model = GlucoseLSTM(hidden_size=64, num_layers=2, num_horizons=2)
    elif model_type == "temporal_cnn":
        model = GlucoseTemporalCNN(num_channels=[32, 64, 64], num_horizons=2)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    # ── 2. Trace to TorchScript ──────────────────────────────────────────
    # Input: (1, 24, 1) — batch=1, seq_len=24 readings, features=1
    example_input = torch.randn(1, 24, 1)
    traced = torch.jit.trace(model, example_input)
    print(f"TorchScript trace OK — output shape: {traced(example_input).shape}")

    # ── 3. Convert to Core ML ────────────────────────────────────────────
    try:
        import coremltools as ct
    except ImportError:
        print("\ncoremltools not installed. Run: pip install coremltools")
        print("Saving TorchScript model instead...")
        ts_path = out / f"GlucoSense_{model_type}.pt"
        traced.save(str(ts_path))
        print(f"Saved TorchScript model → {ts_path}")
        return str(ts_path)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="cgm_window",
            shape=(1, 24, 1),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="glucose_predictions")],
        minimum_deployment_target=ct.target.watchOS8,
        compute_units=ct.ComputeUnit.CPU_AND_NE,   # Neural Engine on Apple Watch S7+
    )

    # ── 4. Add metadata ──────────────────────────────────────────────────
    mlmodel.short_description = "GlucoSense AI — 30-min and 60-min glucose prediction"
    mlmodel.input_description["cgm_window"] = (
        "24 CGM readings at 5-min intervals (2-hour window), normalised 0–1. "
        "Shape: (1, 24, 1)"
    )
    mlmodel.output_description["glucose_predictions"] = (
        "Predicted glucose at [30-min, 60-min] ahead, normalised 0–1. "
        "Apply inverse MinMaxScaler (min=40, max=400) to convert to mg/dL."
    )

    out_path = out / f"GlucoSense_{model_type}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"\nCore ML model saved → {out_path}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024:.1f} KB")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "temporal_cnn"])
    parser.add_argument("--checkpoint", default="models/checkpoints/lstm_best.pth")
    parser.add_argument("--output_dir", default="export/")
    args = parser.parse_args()
    export(args.model, args.checkpoint, args.output_dir)
