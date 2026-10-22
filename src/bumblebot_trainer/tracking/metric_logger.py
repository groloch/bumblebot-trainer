from collections import deque, defaultdict
import torch
import mlflow

class AccumulationBuffer:
    """Buffer containing a key-value data structure, with value being
    stored on device for fast access without synchronizing function call.
    Used mainly for storing metrics to average throughout gradient accumulation steps.
    """
    def __init__(self, accumulation_steps, device, dtype=torch.bfloat16):
        self.accumulation_steps = accumulation_steps
        self.device = device
        self.dtype = dtype
        self.buffers = {}

    def update(self, name, value, partial_step):
        if name not in self.buffers:
            self.buffers[name] = torch.zeros(self.accumulation_steps, device=self.device, dtype=self.dtype)
        self.buffers[name][partial_step % self.accumulation_steps] = value

    def get_mean(self, name, ignore_zeros=False):
        if name not in self.buffers:
            return 0.0
        buf: torch.Tensor = self.buffers[name]
        if ignore_zeros:
            valid: torch.Tensor = buf[buf != 0]
            return valid.mean().item() if valid.numel() > 0 else 0.0
        return buf.mean().item()

    def reset(self):
        buf: torch.Tensor
        for buf in self.buffers.values():
            buf.zero_()

class Meter:
    """Object containing a sliding window of value for averaging
    metrics over last steps.
    """
    def __init__(self, fmt='{avg:.4e}', window_size=100):
        self.fmt = fmt
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)

    def get_value(self):
        return sum(self.values) / len(self.values) if self.values else 0

    def __str__(self):
        avg = sum(self.values) / len(self.values) if self.values else 0
        return self.fmt.format(avg=avg)
        

class MetricLogger:
    """Utility object for streamlined metrics tracking, averaging and logging.
    Currently only support mlflow for tracking.
    """
    def __init__(self, use_mlflow):
        self.metrics = defaultdict(Meter)
        self.use_mlflow = use_mlflow

    def log(self, step, exclude_if_contains=None):
        if exclude_if_contains is None:
            exclude_if_contains = []
        names = self.metrics.keys()
        res = ''

        if len(names) == 0:
            return res

        if self.use_mlflow:
            # Batch all metrics into a single write instead of one round-trip per metric.
            mlflow.log_metrics(
                {name: self.metrics[name].get_value() for name in names},
                step=step,
            )

        for name in names:
            if name.endswith('_avg') and not any(ex in name for ex in exclude_if_contains):
                res += f'{name[:-4]}: {self.metrics[name]}  '

        return res[:-2]
        
    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = Meter(window_size=1)
        self.metrics[name].update(value)
        self.metrics[f'{name}_avg'].update(value)
