import numpy as np

def cosine_similarity(A, B):
    """
    Calculate cosine similarity between 1D array and 2D array.
    A: 1D array
    B: 2D array
    """
    return np.array([np.dot(A, b) / (np.linalg.norm(A) * np.linalg.norm(b)) for b in B])

def process_texts(objects):
    return [f"a photo of a {obj}" for obj in objects]

def softmax_with_temperature(x, temperature=1.0):
    """
    Apply softmax to an array of scores with temperature scaling.
    Lower temperatures make the highest values more pronounced.
    """
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()