import argparse
import os
import tensorflow as tf
from data import get_datasets
from model import build_model

def main(args):
    train_ds, val_ds = get_datasets(args.batch_size)
    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best.keras")
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1)

    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=[ckpt])
    print("Best model saved to", ckpt_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="./outputs/tensorflow")
    args = p.parse_args()
    main(args)
