"""This module implements time related utility functions."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import datetime
import functools
import time
from typing import Optional


def now() -> datetime.datetime:
    """Return the current UTC creation_date and time up to a millisecond accuracy.

    This function is preferable over the Python datetime.datetime.now() function
    because it uses the same accuracy (milliseconds) as MongoDB rather than microsecond accuracy.

    Returns:
        Date and time up to a millisecond precision.
    """
    date = datetime.datetime.now(datetime.timezone.utc)
    return date.replace(microsecond=(date.microsecond // 1000) * 1000)


# Debug tools
def timeit(func):
    """This function can be used as a decorator as @timeit.

    It will print out how long the function took to execute.

    Args:
        func: The decorated function

    Returns:
        The wrapped function
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"function [{func.__name__}] finished in {float(elapsed_time * 1000)} ms")
        return res

    return new_func


class TimeEstimator:
    """The time estimator.

    Estimate the remaining time given the progress, and the progress changes. The estimator starts estimation at a
    starting progress that is not necessarily 0. This choice is motivated by the fact that the first percent of
    progress often takes a much longer time than the following percents.

    Args:
        smoothing_factor (float): Smoothing factor for the exponentially
            weighted moving average. There's a great explanation at
            https://www.wallstreetmojo.com/ewma/
        inflation_factor (float): The factor by which the initial total time
            estimation is inflated to ensure decreasing
        update_window (float): Last update happened at progress1, next update
            will happen at (progress1 + update window)
        starting_progress (float): The progress at which the time_remaining
            estimation starts time estimation
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        smoothing_factor: float = 0.02,
        inflation_factor: float = 1.1,
        update_window: float = 1.0,
        starting_progress: float = 1.0,
    ):
        self.estimated_total_time: Optional[float] = None
        self.estimated_end_time: Optional[float] = None
        self.first_update_progress = starting_progress
        self.first_update_time: Optional[float] = None
        self.last_update_progress: Optional[float] = None
        self.estimated_remaining_time: Optional[float] = None
        self.starting_progress = starting_progress
        self.smoothing_factor = smoothing_factor
        self.inflation_factor = inflation_factor
        self.update_window = update_window

    def time_remaining_from_progress(self, progress: float) -> float:
        """Updates the current progress, and returns the estimated remaining time in seconds (float).

        Args:
            progress (float): The new progress (floating point percentage, 0.0 - 100.0)

        Returns:
            The expected remaining time in seconds (float)
        """
        estimation = -1.0
        if progress is not None and progress > 0:
            self.update(progress)
            estimation = self.get_time_remaining()
        return estimation

    def get_time_remaining(self):
        """If the new estimation is higher than the previous one by up to 2 seconds, return old estimation.

        Returns:
            Estimated remaining time in seconds (float)
        """
        new_estimation = self.estimated_end_time - time.time() if self.estimated_end_time is not None else -1.0
        if self.estimated_remaining_time is None or not 0.0 < new_estimation - self.estimated_remaining_time < 2.0:
            self.estimated_remaining_time = new_estimation
        return self.estimated_remaining_time

    def update(self, progress: float):
        """Update the estimator with a new progress (floating point percentage, between 0.0 - 100.0).

        Args:
            progress (float): Progress of the process

        Returns:
            None
        """
        if progress >= self.first_update_progress and self.last_update_progress is None:
            self.first_update_progress = progress
            self.last_update_progress = progress
            self.first_update_time = time.time()

        if self.last_update_progress is not None and progress - self.last_update_progress >= self.update_window:

            if self.first_update_time is None or self.first_update_progress is None:
                raise AssertionError(
                    "first_update_time and first_update_progress both can not be None when calling "
                    "TimeEstimator.update()."
                )

            self.last_update_progress = progress
            # normalized progress since starting point for estimation
            normalized_progress = (progress - self.first_update_progress) / (100 - self.first_update_progress)
            time_elapsed = time.time() - self.first_update_time
            estimated_total = time_elapsed / normalized_progress
            if self.estimated_total_time is None:
                # inflating the initial estimation to ensure decreasing time remaining
                self.estimated_total_time = estimated_total * self.inflation_factor
            else:
                self.estimated_total_time = (
                    self.smoothing_factor * self.estimated_total_time + (1 - self.smoothing_factor) * estimated_total
                )
            self.estimated_end_time = self.first_update_time + self.estimated_total_time
