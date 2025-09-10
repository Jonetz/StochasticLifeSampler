from abc import abstractmethod
from typing import Optional
import torch
import math

class BaseTemperatureScheduler():
    """
    Abstract base class for temperature schedulers.

    Subclasses must implement `get_temperature`. Handles batch inference
    from previous temperatures if provided.
    """
    def __init__(self, start_temp: float, end_temp: float, steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.steps = steps

    @abstractmethod
    def get_temperature(self, step: int, prev_temps: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the temperature(s) at a given step.

        Args:
            step: current step index (int)
            prev_temps: optional tensor of previous temperatures, shape (batch_size,)
                        used to infer batch size or to maintain stateful schedules.

        Returns:
            torch.Tensor: temperatures for each element in the batch, shape (batch_size,)
        """
        pass

    def _infer_batch_size(self, prev_temps: torch.Tensor = None) -> int:
        """Infer batch size from previous temperatures or default to 1."""
        if prev_temps is not None:
            return prev_temps.shape[0]
        return 1

class ExponentialScheduler(BaseTemperatureScheduler):
    def get_temperature(self, step: int, prev_temps: torch.Tensor = None) -> torch.Tensor:
        batch_size = self._infer_batch_size(prev_temps)
        alpha = step / self.steps
        temp = self.start_temp * (self.end_temp / self.start_temp) ** alpha
        return torch.full((batch_size,), temp, dtype=torch.float32, device=prev_temps.device)

class PlateauExponentialScheduler(BaseTemperatureScheduler):
    def __init__(self, start_temp: float, end_temp: float, steps: int, plateau_frac: float = 0.3):
        super().__init__(start_temp, end_temp, steps)
        self.plateau_steps = int(steps * plateau_frac)

    def get_temperature(self, step: int, prev_temps: torch.Tensor = None) -> torch.Tensor:
        batch_size = self._infer_batch_size(prev_temps)
        step = min(max(0, step), self.steps)
        if step < self.plateau_steps:
            temp = self.start_temp
        else:
            alpha = (step - self.plateau_steps) / (self.steps - self.plateau_steps)
            temp = self.start_temp * (self.end_temp / self.start_temp) ** alpha
        return torch.full((batch_size,), temp, dtype=torch.float32, device=prev_temps.device)

class OscillatingScheduler(BaseTemperatureScheduler):
    def __init__(self, start_temp: float, end_temp: float, steps: int, freq: int = 100):
        super().__init__(start_temp, end_temp, steps)
        self.freq = freq

    def get_temperature(self, step: int, prev_temps: torch.Tensor = None) -> torch.Tensor:
        batch_size = self._infer_batch_size(prev_temps)
        step = min(max(0, step), self.steps)
        base_temp = self.start_temp * (self.end_temp / self.start_temp) ** (step / self.steps)
        oscillation = (self.start_temp - self.end_temp) * 0.1 * math.sin(2 * math.pi * step / self.freq)
        temp = max(base_temp + oscillation, self.end_temp)
        return torch.full((batch_size,), temp, dtype=torch.float32, device=prev_temps.device)
    
class AdaptiveScheduler(BaseTemperatureScheduler):
    """
    Adaptive temperature scheduler for MCMC.

    Features:
    - Slow exponential annealing from start_temp to end_temp.
    - Oscillations to periodically re-explore.
    - Adaptive reheating when acceptance stays too low.

    Args:
        start_temp: initial temperature (>0).
        end_temp: final temperature (>0, <= start_temp).
        steps: number of total steps for annealing.
        freq: oscillation frequency (steps per cycle).
        amp_frac: fraction of (start - end) used for oscillation amplitude.
        min_accept: acceptance rate threshold for reheating trigger.
        reheat_factor: multiplicative bump applied when reheating.
    """
    def __init__(
        self,
        start_temp: float,
        end_temp: float,
        steps: int,
        freq: int = 500,
        amp_frac: float = 0.1,
        min_accept: float = 0.05,
        reheat_factor: float = 1.2,
    ):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.steps = steps
        self.freq = freq
        self.amp_frac = amp_frac
        self.min_accept = min_accept
        self.reheat_factor = reheat_factor

        # state
        self.last_reheat_step = -1
        self.current_multiplier = 1.0

    def get_temperature(
        self,
        step: int,
        prev_temps: Optional[torch.Tensor] = None,
        accept_rate: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute temperature(s) at given step.

        Args:
            step: current step (0 <= step <= steps).
            prev_temps: optional tensor of previous temperatures (for batch inference).
            accept_rate: optional acceptance rate for adaptive reheating.

        Returns:
            Tensor of temperatures (shape inferred from prev_temps if given).
        """
        step = min(max(0, step), self.steps)

        # --- Base exponential annealing ---
        alpha = step / self.steps
        base_temp = self.start_temp * (self.end_temp / self.start_temp) ** alpha

        # --- Oscillation ---
        amplitude = (self.start_temp - self.end_temp) * self.amp_frac
        oscillation = amplitude * math.sin(2 * math.pi * step / self.freq)

        temp = base_temp + oscillation

        # --- Adaptive reheating ---
        if accept_rate is not None and accept_rate < self.min_accept:
            if step > self.last_reheat_step + self.freq:  # avoid spamming reheats
                self.current_multiplier *= self.reheat_factor
                self.last_reheat_step = step

        temp *= self.current_multiplier
        temp = max(temp, self.end_temp)

        # --- Batchify ---
        if prev_temps is not None:
            return torch.full_like(prev_temps, float(temp))
        else:
            return torch.tensor(temp, dtype=torch.float32)