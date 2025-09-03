# mini-ml-starter (PyTorch & TensorFlow)

Небольшой учебный проект: одна задача (классификация одежды из Fashion‑MNIST), две реализации — PyTorch и TensorFlow/Keras.
Цель — дать аккуратную базу, с которой можно быстро стартовать и сравнить фреймворки без лишней магии.

## Особенности
- Две одинаковые по идее реализации (простая CNN): `pytorch/` и `tensorflow/`.
- Готовые скрипты обучения и инференса, минимальные тесты на форму выхода.
- Makefile, Dockerfile и GitHub Actions (CI) уже настроены.
- Код без внешних зависимостей, кроме стандартных библиотек фреймворков.

## Быстрый старт

### PyTorch
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-pytorch.txt
python pytorch/train.py --epochs 3
python pytorch/infer.py --checkpoint outputs/pytorch/best.pt
```

### TensorFlow
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-tf.txt
python tensorflow/train.py --epochs 3
python tensorflow/infer.py --checkpoint outputs/tensorflow/best.keras
```

### Удобные цели Makefile
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

## Структура
```
mini-ml-starter/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ Makefile
├─ requirements-pytorch.txt
├─ requirements-tf.txt
├─ Dockerfile
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ pytorch/
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ infer.py
│  └─ tests/
│     └─ test_model.py
└─ tensorflow/
   ├─ data.py
   ├─ model.py
   ├─ train.py
   ├─ infer.py
   └─ tests/
      └─ test_model.py
```

## Репродуцируемость
В PyTorch фиксируется seed; в Keras поведение по умолчанию достаточно стабильное для учебной задачи.

## Тесты
В обеих реализациях есть быстрый тест на форму выхода сети — это помогает ловить очевидные ошибки без долгих прогона и скачивания датасетов.

## Частые вопросы
**Почему Fashion‑MNIST?** Небольшой датасет, скачивается автоматически, позволяет быстро увидеть прогресс.
**Можно ли подключить свой датасет?** Да: самый простой путь — подготовить папку с изображениями и написать свой загрузчик.
**Нужна ли GPU?** Нет. На CPU проект обучается, пусть и медленнее.

## Лицензия
MIT. См. файл `LICENSE`.
