import numpy as np

class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # Compute pairwise distances using broadcasting
    dist_matrix = np.sqrt(np.sum((des1[:, np.newaxis] - des2) ** 2, axis=2))

    # Find best matches in both directions
    best_for_des1 = np.argmin(dist_matrix, axis=1)
    best_for_des2 = np.argmin(dist_matrix, axis=0)

    # Create indices arrays for cross-check
    query_indices = np.arange(des1.shape[0])
    train_indices = np.arange(des2.shape[0])

    # Find mutual matches using boolean indexing
    mask = best_for_des2[best_for_des1] == query_indices
    mutual_matches = query_indices[mask]

    # Create DummyMatch objects
    matches = [
        DummyMatch(
            i,
            best_for_des1[i],
            dist_matrix[i, best_for_des1[i]]
        )
        for i in mutual_matches
    ]

    return sorted(matches, key=lambda x: x.distance)