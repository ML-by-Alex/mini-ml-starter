.PHONY: setup-pytorch setup-tf train-pytorch train-tf test-pytorch test-tf

    setup-pytorch:
	pip install -r requirements-pytorch.txt

    setup-tf:
	pip install -r requirements-tf.txt

    train-pytorch:
	cd pytorch && python train.py --epochs 3

    train-tf:
	cd tensorflow && python train.py --epochs 3

    test-pytorch:
	cd pytorch && pytest -q

    test-tf:
	cd tensorflow && pytest -q
