from model.hyperparams import *
from model.nerc import Data, collate_fn
from utils.memory_management import load_obj, save_obj
from utils.plotter import plot_last_run

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def evaluate_model(model, dataloader):
    model.eval()

    predicted_labels = []
    labels = []
    total_batches = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (chars, toks, lbls) in enumerate(dataloader):
            chars = chars.to(device)
            toks = toks.to(device)
            lbls = lbls.to(device)

            emissions = model(chars, toks).to(device)
            loss = model.loss(emissions, lbls.reshape(1, batch_size)).item()

            total_loss += loss
            total_batches += 1

            try:
                predicted_labels += model.decode(emissions)[0]
            except Exception:
                predicted_labels += model.decode(emissions).tolist()

            labels += lbls.detach().tolist()

        final_loss = total_loss / total_batches
        f1 = f1_score(labels, predicted_labels, average="micro")

    return (final_loss, f1)

if __name__ == '__main__':
    char_to_idx = load_obj("char_to_idx")
    tag_to_idx = load_obj("tag_to_idx")
    tok_to_idx = load_obj("tok_to_idx")

    model = load_obj("model")

    test_tokens = load_obj("test_tokens")
    test_labels = load_obj("test_labels")

    test_data = Data(test_tokens, test_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len)

    test_dataloader = DataLoader(
        test_data, batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    train_tokens = load_obj("train_tokens")
    train_labels = load_obj("train_labels")

    train_data = Data(train_tokens, train_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len)

    train_dataloader = DataLoader(
        train_data, batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    val_tokens = load_obj("val_tokens")
    val_labels = load_obj("val_labels")

    val_data = Data(val_tokens, val_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len)

    val_dataloader = DataLoader(
        val_data, batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    print("\nEvaluating on test set")
    test_loss, test_f1 = evaluate_model(model, test_dataloader)
    print("Evaluating on train set")
    train_loss, train_f1 = evaluate_model(model, train_dataloader)
    print("Evaluating on valid set")
    val_loss, val_f1 = evaluate_model(model, val_dataloader)

    print("\n[test]  loss: " + str(test_loss) + " F1: " + str(test_f1))
    print("[train] loss: " + str(train_loss) + " F1: " + str(train_f1))
    print("[valid] loss: " + str(val_loss) + " F1: " + str(val_f1))

    plot_last_run()