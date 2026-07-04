import numpy as np, pytest

@pytest.fixture
def tiny_rgb():
    """256x256 RGB: dark matrix with a couple of bright blobs (sulfide) and a grey blob."""
    rng = np.random.default_rng(0)
    img = (rng.integers(8, 28, (256, 256, 3))).astype(np.uint8)  # dark matrix
    img[40:110, 40:110] = 220   # bright sulfide blob
    img[150:210, 150:210] = 120  # mid-grey magnetite blob
    return img
