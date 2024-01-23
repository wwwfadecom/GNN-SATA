import time
import torch
import torch.nn.functional as F
import argparse
from openTSNE import TSNE
from hps4 import get_hyper_param
from model.history import SATA
from matplotlib import pyplot as plt
import utils1
from util import load_dataset, root, get_mask, get_accuracy, set_seed
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,default='cora')
args = parser.parse_args()
set_seed(0xC0FFEE)
epochs = 1000
patience = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = root + "/checkpoint"
feat, label, n, nfeat, nclass, adj = load_dataset(args.dataset, norm=True, device=device)
hp = get_hyper_param(args.dataset)

def train(model, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    result= model(feat=feat, adj=adj)
    loss = F.nll_loss(result[train_mask], label[train_mask])
    loss.backward()
    optimizer.step()
    return get_accuracy(result[train_mask], label[train_mask]), loss.item()


def test(model, test_mask):
    model.eval()
    with torch.no_grad():
        result= model(feat=feat, adj=adj)
        loss = F.nll_loss(result[test_mask], label[test_mask].to(device))
        return get_accuracy(result[test_mask], label[test_mask]), loss.item()


def validate(model, val_mask) -> float:
    model.eval()
    with torch.no_grad():
        result= model(feat=feat, adj=adj)
        return get_accuracy(result[val_mask], label[val_mask])

tsne = TSNE(
        perplexity=100,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
    )

def plot(x, y, **kwargs):
        utils1.plot(
            x,
            y,
        )

def run():
    train_mask, test_mask, val_mask = get_mask(label, 0.6, 0.2, device=device)
    model = SATA(
        n=n,
        nclass=nclass,
        nfeat=nfeat,
        nlayer=hp["layer"],
        lambda_1=hp["lambda_1"],
        lambda_2=hp["lambda_2"],
        dropout=hp["dropout"],
        adj = adj,
        blancealpha = hp["alpha"]
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {'params': model.params1, 'weight_decay': hp["wd1"]},
            {'params': model.params2, 'weight_decay': hp["wd2"]},
            {'params': model.params12, 'weight_decay': hp["wd1"]}
        ],
        lr=hp["lr"]
    )
    checkpoint_file = "{}/{}-{}.pt".format(checkpoint_path, model.__class__.__name__, args.dataset)
    tolerate = 0
    best_loss = 100
    max = 0
    maxadj =adj
    for epoch in range(epochs):
        if tolerate >= patience:
            break
        train_acc, train_loss= train(model, optimizer, train_mask)
        if train_loss < best_loss:
            tolerate = 0
            best_loss = train_loss
        else:
            tolerate += 1
        message = "Epoch={:<4} | Tolerate={:<3} | Train_acc={:.4f} | Train_loss={:.4f}".format(
            epoch,
            tolerate,
            train_acc,
            train_loss,
        )
        val_acc = validate(model, val_mask)
        if val_acc>max:
            max = val_acc
            print(max)
            # maxadj = add
        print(message)


    print(args.dataset)
    print('layer',hp["layer"])
    print('lambda_1',hp["lambda_1"])
    print('lambda_2',hp["lambda_2"])
    print('dropout',hp["dropout"])
    print('lr',hp["lr"])
    print('max:',max)
    torch.save(model.state_dict(), checkpoint_file)
    return max


if __name__ == '__main__':
    run()
