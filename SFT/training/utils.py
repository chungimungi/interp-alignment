import gc
import time
from pathlib import Path


def configure_pytorch_cuda_for_training() -> None:
    """Throughput-friendly CUDA defaults; does not enable AMP/autocast."""
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def release_cuda_memory() -> None:
    """Drop Python cycles and return cached blocks to the CUDA pool."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def latest_checkpoint_path(output_dir: str) -> str | None:
    out_path = Path(output_dir)
    if not out_path.exists():
        return None
    checkpoints = [p for p in out_path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


class ProgressLogger:
    """Callback-compatible timer that logs step progress, elapsed, and ETA."""

    def __init__(self, total_steps: int):
        from transformers import TrainerCallback

        self.total_steps = total_steps
        self._start: float | None = None

        class _Callback(TrainerCallback):
            def on_train_begin(cb_self, args, state, control, **kwargs):
                self._start = time.time()
                print(f"[timer] Training started — {self.total_steps} steps total")

            def on_step_end(cb_self, args, state, control, **kwargs):
                if self._start is None:
                    return
                step = state.global_step
                elapsed = time.time() - self._start
                pct = step / self.total_steps * 100
                avg_per_step = elapsed / step if step > 0 else 0
                eta = avg_per_step * (self.total_steps - step)

                gpu_info = ""
                if step % 50 == 0 or self.total_steps <= 10:
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        used_gb = mem.used / 1024**3
                        total_gb = mem.total / 1024**3
                        gpu_info = (
                            f" — GPU mem {used_gb:.1f}/{total_gb:.1f}GB"
                            f" — GPU util {util.gpu}%"
                        )
                    except Exception:
                        pass

                print(
                    f"[timer] step {step}/{self.total_steps} "
                    f"({pct:.1f}%) — "
                    f"elapsed {elapsed:.0f}s — "
                    f"ETA {eta:.0f}s"
                    f"{gpu_info}"
                )

            def on_train_end(cb_self, args, state, control, **kwargs):
                if self._start is None:
                    return
                total = time.time() - self._start
                print(f"[timer] Training done in {total:.0f}s ({total/60:.1f} min)")

        self.callback = _Callback()
