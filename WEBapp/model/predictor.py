import numpy as np
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        tflite = None


class ASLPredictor:
    """
    Predictor for WLASL-100 word-level sign language recognition.
    Input:  (1, 30, 147) or (30, 147) landmark sequence.
    Output: (predicted_word: str, confidence: float)
    """

    MAX_FRAMES   = 30
    FEATURE_SIZE = 147

    def __init__(
        self,
        model_path=None,
        model_architecture=None,
        label_map=None,        # list of word strings, index == class idx
        label_map_path=None,   # path to JSON {idx: word} or {word: idx}
    ):
        self.model_path         = model_path or os.path.join(
            os.path.dirname(__file__), "converted_model.tflite"
        )
        self.model_architecture = model_architecture
        self.is_tflite          = self.model_path.endswith(".tflite")
        self.interpreter        = None
        self.input_details      = None
        self.output_details     = None
        self.model              = None
        self.device             = None

        # Build label list
        if label_map is not None:
            self.LABELS = list(label_map)
        elif label_map_path and os.path.exists(label_map_path):
            import json
            with open(label_map_path) as f:
                raw = json.load(f)
            if all(k.isdigit() for k in raw):
                self.LABELS = [raw[str(i)] for i in range(len(raw))]
            else:
                inv = {int(v): k for k, v in raw.items()}
                self.LABELS = [inv[i] for i in range(len(inv))]
        else:
            self.LABELS = [str(i) for i in range(100)]

        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        if self.is_tflite:
            try:
                if tflite is None:
                    raise ImportError("Neither tflite_runtime nor tensorflow installed.")
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.input_details  = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"[ASLPredictor] TFLite model loaded from {self.model_path}")
            except Exception as e:
                print(f"[ASLPredictor] TFLite load error: {e}")
                self._stub()
            return

        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
            elif isinstance(checkpoint, dict) and self.model_architecture is not None \
                    and "model_state_dict" not in checkpoint:
                self.model_architecture.load_state_dict(checkpoint)
                self.model = self.model_architecture
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                if self.model_architecture is None:
                    raise ValueError("Provide model_architecture for state_dict checkpoint.")
                self.model_architecture.load_state_dict(checkpoint["model_state_dict"])
                self.model = self.model_architecture
            else:
                raise ValueError("Unrecognized checkpoint format.")

            self.model.to(self.device).eval()
            print(f"[ASLPredictor] PyTorch model loaded on {self.device}")

        except ImportError:
            print("[ASLPredictor] PyTorch not installed.")
            self._stub()
        except Exception as e:
            print(f"[ASLPredictor] Load error: {e}")
            self._stub()

    def _stub(self):
        print("[ASLPredictor] Running in STUB mode — predictions are random.")
        self.model       = None
        self.interpreter = None

    # ------------------------------------------------------------------
    def predict(self, sequence: np.ndarray):
        seq = np.array(sequence, dtype=np.float32)
        if seq.ndim == 2:
            seq = seq[np.newaxis, ...]   # (30,147) → (1,30,147)

        # ── TFLite ──────────────────────────────────────────────────────
        if self.is_tflite and self.interpreter is not None:
            inp = np.transpose(seq, (0, 2, 1))   # (1,147,30)
            self.interpreter.set_tensor(self.input_details[0]['index'], inp)
            self.interpreter.invoke()
            logits = self.interpreter.get_tensor(self.output_details[0]['index'])
            e_x   = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = (e_x / e_x.sum(axis=-1, keepdims=True))[0] # Shape: (num_classes,)
            
            # Extract Top 10 Indices Sorted Descending
            top_10_indices = np.argsort(probs)[::-1][:10]
            
            print("\n" + "="*45 + "\n   TOP 10 CONFIDENCE SCORES (TFLite)\n" + "-"*45, flush=True)
            for rank, idx in enumerate(top_10_indices, 1):
                word = self.LABELS[idx] if idx < len(self.LABELS) else str(idx)
                print(f" {rank:2d}. [Index {idx:3d}] {word:<15} -> Prob: {probs[idx]:.4f}", flush=True)
            print("="*45 + "\n", flush=True)

            idx   = int(top_10_indices[0])
            conf  = float(probs[idx])
            word  = self.LABELS[idx] if idx < len(self.LABELS) else str(idx)
            return word, round(conf, 4)

        # ── PyTorch ─────────────────────────────────────────────────────
        if self.model is not None:
            import torch
            tensor = torch.tensor(seq, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)[0] # Shape: (num_classes,)
                
                # Extract Top 10 Indices Sorted Descending
                top_values, top_indices = torch.topk(probs, 10)
                
            print("\n" + "="*45 + "\n   TOP 10 CONFIDENCE SCORES (PyTorch)\n" + "-"*45, flush=True)
            for rank in range(10):
                idx = top_indices[rank].item()
                val = top_values[rank].item()
                word = self.LABELS[idx] if idx < len(self.LABELS) else str(idx)
                print(f" {rank+1:2d}. [Index {idx:3d}] {word:<15} -> Prob: {val:.4f}", flush=True)
            print("="*45 + "\n", flush=True)

            idx  = top_indices[0].item()
            conf = top_values[0].item()
            word = self.LABELS[idx] if idx < len(self.LABELS) else str(idx)
            return word, round(conf, 4)

        # ── Stub ─────────────────────────────────────────────────────────
        idx  = np.random.randint(0, len(self.LABELS))
        conf = round(float(np.random.uniform(0.55, 0.95)), 4)
        return self.LABELS[idx], conf

    # ------------------------------------------------------------------
    def get_model_info(self):
        loaded = (self.model is not None) or (self.interpreter is not None)
        return {
            "loaded":      loaded,
            "model_path":  self.model_path,
            "num_classes": len(self.LABELS),
            "labels":      self.LABELS,
            "mode":        "inference" if loaded else "stub",
            "device":      "CPU (TFLite)" if self.is_tflite else (str(self.device) if self.device else "N/A"),
            "input_shape": f"(1, {self.FEATURE_SIZE}, {self.MAX_FRAMES})" if self.is_tflite
                           else f"(1, {self.MAX_FRAMES}, {self.FEATURE_SIZE})",
        }