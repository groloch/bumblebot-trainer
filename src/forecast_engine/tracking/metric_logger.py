from collections import deque, defaultdict
import torch
import mlflow

class AccumulationBuffer:
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
        buf = self.buffers[name]
        if ignore_zeros:
            valid = buf[buf != 0]
            return valid.mean().item() if valid.numel() > 0 else 0.0
        return buf.mean().item()

    def reset(self):
        for buf in self.buffers.values():
            buf.zero_()

class Meter:
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

        for name in names:
            if self.use_mlflow:
                mlflow.log_metric(name, self.metrics[name].get_value(), step=step)
            if name.endswith('_avg') and not any(ex in name for ex in exclude_if_contains):
                res += f'{name[:-4]}: {self.metrics[name]}  '

        return res[:-2]
        
    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = Meter(window_size=1)
        self.metrics[name].update(value)
        self.metrics[f'{name}_avg'].update(value)
