class TemperatureScheduler:
    """
    Example: exponential cooling for simulated annealing.
    """
    def __init__(self, start_temp: float, end_temp: float, steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.steps = steps

    def get_temperature(self, step: int) -> float:
        alpha = step / self.steps
        return self.start_temp * (self.end_temp / self.start_temp) ** alpha
