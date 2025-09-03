FROM python:3.11-slim
ARG FRAMEWORK=pytorch
WORKDIR /app

COPY requirements-pytorch.txt requirements-tf.txt ./
RUN if [ "$FRAMEWORK" = "pytorch" ]; then \
  pip install --no-cache-dir -r requirements-pytorch.txt; \
else \
  pip install --no-cache-dir -r requirements-tf.txt; \
fi

COPY . .
CMD ["bash", "-lc", "echo 'Set FRAMEWORK=pytorch|tensorflow and run: python pytorch/train.py or tensorflow/train.py'"]
