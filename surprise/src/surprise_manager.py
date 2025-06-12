# surprise_manager.py

import io
import numpy as np
from PIL import Image
from typing import List, Tuple

class SurpriseManager:
    def __init__(self):
        # A wider beam gives more accuracy at a slight performance cost.
        # 10 is a great balance.
        self.BEAM_WIDTH = 10

    def _calculate_dissimilarity_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculates dissimilarity using Normalized Cross-Correlation (NCC).
        NCC is robust to brightness and contrast variations.
        A perfect match gives NCC = 1. We convert this to a cost (dissimilarity)
        by returning 1.0 - NCC. A lower cost is better.
        """
        # We only care about the edge pixels
        edge1 = img1[:, -1].flatten().astype(np.float32)
        edge2 = img2[:, 0].flatten().astype(np.float32)

        # Subtract the mean to make it zero-centered
        edge1_mean = np.mean(edge1)
        edge2_mean = np.mean(edge2)
        edge1 = edge1 - edge1_mean
        edge2 = edge2 - edge2_mean

        # Calculate the normalization terms (standard deviations)
        norm1 = np.sqrt(np.sum(edge1**2))
        norm2 = np.sqrt(np.sum(edge2**2))

        # Avoid division by zero for blank edges
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 1.0  # Max cost for blank edges

        # Calculate the cross-correlation
        ncc = np.dot(edge1, edge2) / (norm1 * norm2)

        # NCC gives a similarity score from -1 to 1. We want a cost (dissimilarity).
        # We use 1.0 - ncc, so that a perfect match (ncc=1) has a cost of 0.
        return 1.0 - ncc

    def surprise(self, slices: list[bytes]) -> list[int]:
        """
        Reconstructs a document using a high-performance Beam Search guided by a
        robust Normalized Cross-Correlation (NCC) metric. This provides a superior
        balance of speed and accuracy.
        """
        num_slices = len(slices)
        if num_slices <= 1:
            return list(range(num_slices))

        # 1. Decode images. We use grayscale ('L') as NCC handles brightness for us.
        images = [np.array(Image.open(io.BytesIO(s)).convert('L')) for s in slices]

        # 2. Build the cost matrix using our superior NCC metric
        cost_matrix = np.full((num_slices, num_slices), 2.0) # Max cost is 2.0
        for i in range(num_slices):
            for j in range(num_slices):
                if i != j:
                    cost_matrix[i, j] = self._calculate_dissimilarity_ncc(images[i], images[j])

        # 3. Run the refined Beam Search algorithm
        
        # a. Initialize the beam with the best `k` starting pairs
        beam: List[Tuple[float, List[int]]] = []
        flat_indices = np.argsort(cost_matrix.flatten())[:self.BEAM_WIDTH]
        for idx in flat_indices:
            i, j = np.unravel_index(idx, cost_matrix.shape)
            beam.append((cost_matrix[i, j], [i, j]))

        # b. Iteratively expand the best candidates
        for _ in range(num_slices - 2):
            candidates = []
            for avg_score, chain in beam:
                total_score = avg_score * (len(chain) - 1)
                used = set(chain)
                
                # Try extending to the right
                for k in range(num_slices):
                    if k not in used:
                        new_score = (total_score + cost_matrix[chain[-1], k]) / len(chain)
                        candidates.append((new_score, chain + [k]))
                
                # Try extending to the left
                for k in range(num_slices):
                    if k not in used:
                        new_score = (total_score + cost_matrix[k, chain[0]]) / len(chain)
                        candidates.append((new_score, [k] + chain))
            
            # c. Prune the beam, keeping only the top `k` candidates
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:self.BEAM_WIDTH]

        # 4. Return the best chain found
        return beam[0][1] if beam else list(range(num_slices))