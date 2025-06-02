import cProfile
import functools
import pstats
import time
from typing import Any, Callable, Optional

from diabetes_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function using cProfile.

    Args:
        func: Function to profile

    Returns:
        Wrapped function that profiles execution
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        profiler = cProfile.Profile()
        try:
            result = profiler.runcall(func, *args, **kwargs)
            # Create stats object and print results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            logger.info(f"\nProfiling results for {func.__name__}:")
            stats.print_stats(20)  # Show top 20 functions
            return result
        except Exception as e:
            logger.error(f"Error during profiling: {str(e)}")
            return func(*args, **kwargs)
    return wrapper

class PerformanceTracker:
    """Class to track performance metrics of model training and prediction."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None
        self.metrics: dict = {}

    def start(self) -> None:
        """Start timing an operation."""
        self.start_time = time.time()

    def end(self) -> float:
        """End timing an operation and return duration in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        duration = time.time() - self.start_time
        self.metrics['duration'] = duration
        logger.info(f"{self.name} took {duration:.2f} seconds")
        return duration

    def add_metric(self, name: str, value: float) -> None:
        """Add a custom metric.

        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.metrics[name] = value
        logger.info(f"{self.name} - {name}: {value}")

    def get_metrics(self) -> dict:
        """Get all tracked metrics.

        Returns:
            Dictionary of all metrics
        """
        return self.metrics
