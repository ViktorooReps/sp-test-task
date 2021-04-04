```bash
git clone https://github.com/ViktorooReps/sp-test-task
cd sp-test-task
```

## 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

## 2. Загрузка эмбеддингов, препроцессирование данных
```bash
./init.sh
```

## 3. Обучение модели

# 3.1 Без использоавания активного обучения
На всем датасете:
```bash
python train.py
```
На маленькой части:
```bash
python train.py --mini
```
C определением оптимального количества эпох с помощью early stopping:
```bash
python train.py --stopper
```

# 3.2 С использованием активного обучения
Со стратегией сэмплирования новых данных N-best Sequence Entropy:
```bash
python train.py --active
```
Со рандомным сэмплированием новых данных:
```bash
python train.py --active --randsampling
```

## 4. Оценка модели
Полученной без активного обучения:
```bash
python evaluate.py
```
Полученной после активного обучения:
```bash
python evaluate.py --active
```
