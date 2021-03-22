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
На всем датасете:
```bash
python train.py full
```
На маленькой части:
```bash
python train.py full --mini
```
Также вместо `full` можно поставить `LSTM`, чтобы обучить bLSTMCRF модель или `CNN`, чтобы обучить CNNCRF.

## 4. Оценка модели:
```bash
python evaluate.py
```
