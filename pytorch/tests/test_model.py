import torch
from model import SimpleCNN

def test_forward_shape():
    m = SimpleCNN()
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    assert y.shape == (2, 10)
