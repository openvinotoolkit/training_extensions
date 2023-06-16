import numpy as np


def sanitize_coordinates(bbox: np.ndarray, height: int, width: int, padding=1) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.int)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return np.array([x1, y1, x2, y2])
