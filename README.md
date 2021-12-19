# N-Best Sequence Entropy and random sampling strategies comparison for Active Learning of Named Entity Recognition and Classification model 

![Alt text](plots/i100s100_entropy.gif?raw=true "Title")

## Results

To illustrate the effectiveness of N-Best Sequence Entropy (NSE) sampling the CNN-LSTM-CRF model was implemented following [Xuezhe Ma and Eduard Hovy](https://arxiv.org/pdf/1603.01354.pdf).

| Model                     | F1 on dev | F1 on test
| ------------------------- | --------- | ----------
| Our implementation        | 0.9474    | 0.9092
| Xuezhe Ma and Eduard Hovy | 0.9474    | 0.9121

NSE and random sampling strategies comparison for different initial training sets sizes:

1. Initial set of 100 sentences

![Alt text](plots/active_comp_i100_s100_f1_seqs.png?raw=true "Title")

2. Initial set of 500 sentences

![Alt text](plots/active_comp_i500_s100_f1_seqs.png?raw=true "Title")

Entropy-based sampling gives a stable increase in F1 score for any (tested) size of intial size of training set. But it also increases number of new tokens for tagging. This is because on average longer sequences have larger entropy. In our experiments average length of sampled sentences increased up to 35% compared to sentences sampled with random strategy. 

When tagging complexity (number of new tokens with non-O tags) is taken into account the F1 score increase becomes a lot less dramatic:

1. Initial set of 100 sentences

![Alt text](plots/active_comp_i100_s100_f1_tags.png?raw=true "Title")

2. Initial set of 500 sentences

![Alt text](plots/active_comp_i500_s100_f1_tags.png?raw=true "Title")


## Conclusion

Though entropy-based sampling does increase tagging complexity of new sentences, choosing on average sentences with larger number of named entities, it still gives a noticeable perfomance increase that allows to achieve adequate model perfomance on much smaller datasets. 

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
