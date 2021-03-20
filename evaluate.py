from model.hyperparams import device, batch_size

from sklearn.metrics import f1_score

def evaluate_model(model, dataloader):
    model.eval()

    predicted_labels = []
    labels = []
    total_batches = 0
    total_loss = 0
    for batch_idx, (chars, toks, lbls) in enumerate(dataloader):
        chars = chars.to(device)
        toks = toks.to(device)
        lbls = lbls.to(device)

        emissions = model(chars, toks).to(device)
        loss = model.loss(emissions, lbls.reshape(1, batch_size)).item()

        total_loss += loss
        total_batches += 1

        predicted_labels += model.decode(emissions)[0]
        labels += lbls.detach().tolist()

    final_loss = total_loss / total_batches
    f1 = f1_score(labels, predicted_labels, average="micro")

    return (final_loss, f1)