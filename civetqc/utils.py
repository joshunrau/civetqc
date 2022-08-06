import time

from typing import Callable

def use_timer(func: Callable):
  def wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    func(*args, **kwargs)
    print(f"Completed in {time.perf_counter() - start_time:.3f}s")
  return wrapper