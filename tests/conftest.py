import numpy as np
import random
import os

def pytest_configure(config):
    # Make tests deterministic(ish)
    seed = int(os.environ.get("TEST_SEED", "12345"))
    np.random.seed(seed)
    random.seed(seed)
