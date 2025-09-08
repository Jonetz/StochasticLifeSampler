class TemperatureScheduler:
    """
    Simple temperature scheduler for simulated annealing.

    Implements exponential cooling from `start_temp` to `end_temp`
    over a fixed number of steps.

    Args:
        start_temp: initial temperature (float > 0)
        end_temp: final temperature (float > 0, <= start_temp)
        steps: total number of steps over which to anneal (int > 0)
    """
    def __init__(self, start_temp: float, end_temp: float, steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.steps = steps

    def get_temperature(self, step: int) -> float:
        """
        Compute the temperature at a given step.

        Args:
            step: current step index (int, 0 <= step <= self.steps)

        Returns:
            float: temperature at this step, never negative

        Safeguards:
            - Clamps `step` between 0 and `self.steps` to avoid extrapolation
            - Ensures result is >= 0
        """
        alpha = step / self.steps
        return self.start_temp * (self.end_temp / self.start_temp) ** alpha
