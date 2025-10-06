from abc import ABC, abstractmethod
from tvm import te

class OptimizationStrategy(ABC):
    """
    Abstract Base Class for an optimization strategy.
    Each strategy will implement the `apply` method to define a specific
    TVM schedule optimization.
    """
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        """
        Applies a specific optimization to the given TVM schedule.

        Args:
            schedule (te.Schedule): The TVM schedule to be modified.
            operator (te.Tensor): The output tensor of the operation to be scheduled.
        """
        pass

    def __str__(self):
        return self.__class__.__name__.replace("Strategy", "")
