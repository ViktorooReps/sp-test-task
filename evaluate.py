from model.hyperparams import *
from model.nerc import Data, collate_fn
from utils.memory_management import load_obj, save_obj
from utils.plotter import plot_last_run

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from pprint import pprint 

import matplotlib.pyplot as plt
import numpy as np

def score(labels, preds, idx_to_tag, excluded_tags={"O"}):
    tag_to_oh_preds = dict()
    tag_to_oh_lbls = dict()

    for val in idx_to_tag.values():
        tag_to_oh_preds[val] = []
        tag_to_oh_lbls[val] = []

    for pred in preds:
        for val in idx_to_tag.values():
            tag_to_oh_preds[val].append(1 if val == idx_to_tag[pred] else 0)

    for lbl in labels:
        for val in idx_to_tag.values():
            tag_to_oh_lbls[val].append(1 if val == idx_to_tag[lbl] else 0)

    tag_to_oh_preds = { val : torch.tensor(lst, requires_grad=False) 
        for val, lst in tag_to_oh_preds.items() }

    tag_to_oh_lbls = { val : torch.tensor(lst, requires_grad=False) 
        for val, lst in tag_to_oh_lbls.items() }

    TP = lambda lbl, prds: torch.sum(lbl * prds).item()
    FP = lambda lbl, prds: torch.sum((lbl == 0) * prds).item()
    FN = lambda lbl, prds: torch.sum(lbl * (prds == 0)).item()

    tag_to_cnts = {
        val : {
            "TP" : TP(tag_to_oh_lbls[val], tag_to_oh_preds[val]),
            "FP" : FP(tag_to_oh_lbls[val], tag_to_oh_preds[val]),
            "FN" : FN(tag_to_oh_lbls[val], tag_to_oh_preds[val])
        } for val in idx_to_tag.values()
    }

    with np.errstate(divide='ignore', invalid='ignore'):
        tag_to_score = {
            val : {
                "precision" : np.true_divide(tag_to_cnts[val]["TP"],
                    (tag_to_cnts[val]["TP"] + tag_to_cnts[val]["FP"])),
                "recall"    : np.true_divide(tag_to_cnts[val]["TP"],
                    (tag_to_cnts[val]["TP"] + tag_to_cnts[val]["FN"])),
                "f1"        : np.true_divide(tag_to_cnts[val]["TP"], 
                    (tag_to_cnts[val]["TP"] + (tag_to_cnts[val]["FP"] + tag_to_cnts[val]["FN"]) / 2))
            } for val in idx_to_tag.values()
        }

    f1 = sum(tag_to_score[val]["f1"] if val not in excluded_tags else 0 
        for val in idx_to_tag.values()) / (len(idx_to_tag) - 1)

    return f1, tag_to_score

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
        #f1 = f1_score(labels, predicted_labels, average="micro")

        idx_to_tag = load_obj("idx_to_tag")
        f1, tag_to_score = score(labels, predicted_labels, idx_to_tag)
        pprint(tag_to_score)

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