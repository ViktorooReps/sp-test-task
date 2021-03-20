```bash
git clone https://github.com/ViktorooReps/sp-test-task
cd sp-test-task
```

## 1. Установка зависимостей:
```bash
pip install -r requirements.txt
```

## 2. Загрузка эмбеддингов, препроцессирование данных:
```bash
./init.sh
```

## 3. Обучение модели:
```bash
python train.py
```

## 4. Оценка модели:
```bash
python evaluate.py
```
