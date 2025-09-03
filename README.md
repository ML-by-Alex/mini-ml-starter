# mini-ml-starter (PyTorch & TensorFlow)

Tiny, no-nonsense starter for image classification on **Fashion-MNIST** with **two side-by-side implementations**: **PyTorch** and **TensorFlow/Keras**. A clean baseline you can run, read, and extend — no extra tooling required.

## Highlights
- ✅ Same simple CNN in both frameworks (`pytorch/` and `tensorflow/`)
- ✅ Ready-to-run training & inference scripts
- ✅ Lightweight tests with `pytest` (fast shape checks)
- ✅ Makefile & Docker for repeatable runs
- ✅ GitHub Actions CI out of the box
- ✅ No extra deps beyond the frameworks

---

## Quickstart

### PyTorch
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts ctivate
pip install -r requirements-pytorch.txt
python pytorch/train.py --epochs 3
python pytorch/infer.py --checkpoint outputs/pytorch/best.pt
```

### TensorFlow
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts ctivate
pip install -r requirements-tf.txt
python tensorflow/train.py --epochs 3
python tensorflow/infer.py --checkpoint outputs/tensorflow/best.keras
```

### Makefile shortcuts
```bash
make setup-pytorch && make train-pytorch && make test-pytorch
make setup-tf && make train-tf && make test-tf
```

### Docker (CPU)
```bash
# PyTorch
docker build -t mini-ml-pytorch --build-arg FRAMEWORK=pytorch .
docker run --rm -it -v $PWD:/app mini-ml-pytorch bash -lc "python pytorch/train.py"

# TensorFlow
docker build -t mini-ml-tf --build-arg FRAMEWORK=tensorflow .
docker run --rm -it -v $PWD:/app mini-ml-tf bash -lc "python tensorflow/train.py"
```

---

## Project layout
```
mini-ml-starter/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ Makefile
├─ requirements-pytorch.txt
├─ requirements-tf.txt
├─ Dockerfile
├─ .github/workflows/ci.yml
├─ pytorch/
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ infer.py
│  └─ tests/test_model.py
└─ tensorflow/
   ├─ data.py
   ├─ model.py
   ├─ train.py
   ├─ infer.py
   └─ tests/test_model.py
```

## Tests
Run the tiny sanity test that checks output shapes:
```bash
cd pytorch && pytest -q    # or: cd tensorflow && pytest -q
```

## Notes
- Datasets download automatically to `./data` on first run.
- Checkpoints are saved to `./outputs/<framework>/`.
- GPU is optional; CPU works fine for this tutorial.

## Troubleshooting
- CI fails on TensorFlow install? Replace `tensorflow>=2.15` with `tensorflow-cpu>=2.15` in `requirements-tf.txt` and commit again.

## Extend this starter
- Add augmentations (e.g., random crop/flip)
- Introduce a proper train/val/test split
- Swap Fashion-MNIST for your own images dataset
- Add logging (TensorBoard or W&B) and config via YAML

## License
MIT — see `LICENSE`.

