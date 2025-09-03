import tensorflow as tf
from model import build_model

def test_forward_shape():
    m = build_model()
    x = tf.random.normal([2, 28, 28, 1])
    y = m(x)
    assert y.shape == (2, 10)
