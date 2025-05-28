from typing import Optional, Tuple

import numpy as np


def seeding(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    seed_seq = np.random.SeedSequence(seed)
    seed = seed_seq.entropy
    bit_generator = np.random.PCG64(seed)
    return np.random.Generator(bit_generator), seed
