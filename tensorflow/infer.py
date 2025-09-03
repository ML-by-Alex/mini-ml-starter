import argparse
import tensorflow as tf
from data import get_datasets

def evaluate(args):
    _, test_ds = get_datasets(args.batch_size)
    model = tf.keras.models.load_model(args.checkpoint)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test acc: {acc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="./outputs/tensorflow/best.keras")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    evaluate(args)
