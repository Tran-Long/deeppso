import time
import logging

# file_logger = logging.getLogger(__name__)
# file_logger.setLevel(logging.INFO)
# file_handler = logging.FileHandler("execution_times.log")
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter("%(message)s")
# file_handler.setFormatter(formatter)
# file_logger.addHandler(file_handler)


def timeit(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"⌛ {func.__name__}: {end_time - start_time:.4f} seconds")
        # file_logger.info(f"{func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper