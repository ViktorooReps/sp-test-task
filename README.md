# N-Best Sequence Entropy and random sampling strategies comparison for Active Learning of Named Entity Recognition and Classification model 

![Alt text](plots/i100s100_entropy.gif?raw=true "Title")


## Results reproduction

```bash
git clone https://github.com/ViktorooReps/sp-test-task
cd sp-test-task
```

### 1. Install dependencies 
```bash
pip install -r requirements.txt
```

### 2. Download embeddings, preprocess data
```bash
./init.sh
```

### 3. Train model

#### 3.1 Without active learning

On the whole dataset:
```bash
python train.py
```
On small part of dataset:
```bash
python train.py --mini
```
With early stopping to determine optimal epoch count:
```bash
python train.py --stopper
```

#### 3.2 With active learning

With N-best Sequence Entropy sampling strategy:
```bash
python train.py --active
```
With random sampling strategy:
```bash
python train.py --active --randsampling
```

### 4. Evaluate model

Trained normally:
```bash
python evaluate.py
```
Trained with active learning:
```bash
python evaluate.py --active
```
