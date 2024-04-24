import time
import torch
import torch.nn.functional as F
import argparse
from openTSNE import TSNE
from hps import get_hyper_param
from model.history1 import SATA
from matplotlib import pyplot as plt
import utils1
from util import load_dataset, root, get_mask, get_accuracy, set_seed
torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,default='chameleon')
args = parser.parse_args()
set_seed(0xC0FFEE)
epochs = 1000
patience = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = root + "/checkpoint"
feat, label, n, nfeat, nclass, adj = load_dataset(args.dataset, norm=True, device=device)
hp = get_hyper_param(args.dataset)

def train(model, optimizer, train_mask,value1,value2):
    model.train()
    optimizer.zero_grad()
    result,EP,EA,add= model(feat=feat, adj=adj,value1 = value1,value2=value2)
    loss = F.nll_loss(result[train_mask], label[train_mask])
    loss.backward()
    optimizer.step()
    return get_accuracy(result[train_mask], label[train_mask]), loss.item(), EP, EA,result,add


def test(model, test_mask):
    model.eval()
    with torch.no_grad():
        result,EP,EA,add= model(feat=feat, adj=adj)
        loss = F.nll_loss(result[test_mask], label[test_mask].to(device))
        return get_accuracy(result[test_mask], label[test_mask]), loss.item()


def validate(model, val_mask) -> float:
    model.eval()
    with torch.no_grad():
        result,EP,EA,add= model(feat=feat, adj=adj)
        return get_accuracy(result[val_mask], label[val_mask])

tsne = TSNE(
        perplexity=100,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
    )

def g(t, T,lambda_val):
    result = max(0, lambda_val + (1 - lambda_val) * t/T)
    # result = min(1,np.sqrt(lambda_val*lambda_val+(1-lambda_val*lambda_val)*t/T))
    # result = min(1, 2**(np.log2(lambda_val)-np.log2(lambda_val*t/T)))
    return result
def g1(t, T,lambda_val):
    result = min(1, lambda_val + (1 - lambda_val) * t/T)
    # result = min(1,np.sqrt(lambda_val*lambda_val+(1-lambda_val*lambda_val)*t/T))
    # result = min(1, 2**(np.log2(lambda_val)-np.log2(lambda_val*t/T)))
    return 1-result
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
    accuracy_values = []
    for epoch in range(epochs):
        if tolerate >= patience:
            break
        if epoch ==0:
            train_acc, train_loss,EP, EA, p,add= train(model, optimizer, train_mask,None,None)
            count = (EP < 0.1).sum().item()
            lambda_val1 = count / EP.shape[0] / EP.shape[0]
            count = (EA > 0.9).sum().item()
            lambda_val2 = count / EP.shape[0] / EP.shape[0]
        else:
            train_acc, train_loss, EP, EA, p,add = train(model, optimizer, train_mask,g1(epoch,200,lambda_val1),g1(epoch,200,lambda_val2))
            # print('g(epoch,200,lambda_val1)',g1(epoch,200,lambda_val1))
            # print('g(epoch,200,lambda_val2)',g1(epoch,200,lambda_val2))
        test_acc, test_loss = test(model, test_mask)
        if train_loss < best_loss:
            tolerate = 0
            best_loss = train_loss
        else:
            tolerate += 1
        message = "Epoch={:<4} | Tolerate={:<3} | Train_acc={:.4f} | Train_loss={:.4f} | Test_acc={:.4f} | Test_loss={:.4f}".format(
            epoch,
            tolerate,
            train_acc,
            train_loss,
            test_acc,
            test_loss
        )
        val_acc = validate(model, val_mask)
        # accuracy_values.append(val_acc+0.02)
        from sklearn.metrics import silhouette_score, pairwise_distances
        import numpy as np
        # 假设已经有了聚类结果labels


        if val_acc > max:
            max = val_acc
            print(max)
            maxadj = add
            if epoch > 40:

                distances = pairwise_distances(p.cpu().detach().numpy(), metric='euclidean')

                silhouette_avg = silhouette_score(distances, np.argmax(p.cpu().detach().numpy(), axis=1),
                                                  metric='precomputed')
                print("轮廓系数: ", silhouette_avg)
                embedding = tsne.fit(p.cpu().detach().numpy())
                plot(embedding, label.cpu().numpy(), colors=utils1.MACOSKO_COLORS)
                plt.savefig('./img/chameleon-.pdf')
        print(message)
    # sum1= 0
    # sum2 = 0
    # sum3= 0
    # sum4 = 0
    # for i in range(n):
    #     for j in range(n - i):
    #         if maxadj[i][j] != 0:
    #             sum3 = sum3+1
    #             if label[i]==label[j]:
    #                 sum1 = sum1+1
    # print(sum1)
    # print(sum2)
    # print(sum3)
    # print("相连同类边:", sum1)
    # print("相连不同类:", sum3-sum1)
    # print("相连边:", sum3)

    print(args.dataset)
    print('layer',hp["layer"])
    print('alpha',hp["alpha"])
    print('lambda_1',hp["lambda_1"])
    print('lambda_2',hp["lambda_2"])
    print('dropout',hp["dropout"])
    print('wd1',hp["wd1"])
    print('wd2',hp["wd2"])
    print('lr',hp["lr"])
    print('max:',max)
    torch.save(model.state_dict(), checkpoint_file)


    # 假设你有一个存储每个epoch准确率的列表 accuracy_values
    # 创建一个新的图
    # plt.figure()

    # 绘制准确率曲线图

    # 添加标题和标签
    # plt.title('Accuracy over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, marker='.', markersize=5)
    #
    # # 添加标题和标签
    # plt.title('Accuracy over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylabel('Accuracy')
    # plt.yticks([0.3, 0.4, 0.5, 0.6,0.7])
    # plt.tight_layout()  # 自动调整子图间距
    # plt.savefig('accuracy_curvechameleon.pdf')
    return max



if __name__ == '__main__':
    run()
