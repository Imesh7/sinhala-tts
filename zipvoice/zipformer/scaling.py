
from typing import Union

from torch import nn


class ScheduledFloat(nn.Module):
    def __init__(self, schedule=None, default=0.0):
        super().__init__()
        self.schedule = schedule
        self.batch_size = None
        self.default = default

    def __float__(self):
        if not self.training or self.schedule is None:
            return float(self.default)
        else:
            for step, value in self.schedule:
                if step > self._get_current_step():
                    return float(value)
            return float(self.schedule[-1][1])

    def _get_current_step(self):
        # Implement logic to get the current training step
        # This is a placeholder and should be replaced with actual logic
        return self.batch_size if self.batch_size is not None else 0
    
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
        
FloatLike = Union[float, ScheduledFloat]