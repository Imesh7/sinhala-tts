
from typing import Union


class ScheduledFloat:
    def __init__(self, schedule=None):
        self.schedule = schedule
        self.batch_size = None

    def __float__(self):
        for step, value in self.schedule:
            if step > self._get_current_step():
                return value
        return self.schedule[-1][1]

    def _get_current_step(self):
        # Implement logic to get the current training step
        # This is a placeholder and should be replaced with actual logic
        return self.batch_size if self.batch_size is not None else 0
    
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
        
FloatLike = Union[float, ScheduledFloat]