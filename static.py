import numpy as np
import torch
from copy import deepcopy
from typing import List
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def confusion_matrix(label: torch.Tensor, predict: torch.Tensor, num_class: int) -> np.ndarray:
    cm = np.zeros((num_class, num_class))
    for i in range(num_class):
        for j in range(num_class):
            cm[i][j] = ((predict == j) & (label == i)).sum() # row is label, col is pred
    return cm.astype(np.int32)

def tp(cm: np.ndarray, target_class: int) -> int:
    return cm[target_class, target_class]

def fp(cm: np.ndarray, target_class: int) -> int:
    cm_temp = deepcopy(cm)
    cm_temp[target_class, target_class] = 0
    return np.sum(cm_temp[:, target_class])

def tn(cm: np.ndarray, target_class: int) -> int:
    cm_temp = deepcopy(cm)
    cm_temp[target_class, :] = 0
    cm_temp[:, target_class] = 0
    return np.sum(cm_temp)

def fn(cm: np.ndarray, target_class: int) -> int:
    cm_temp = deepcopy(cm)
    cm_temp[target_class, target_class] = 0
    return np.sum(cm_temp[target_class, :])

def tpr(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fn = fn(cm, target_class)
    denominator = (_tp + _fn)
    if denominator != 0:
        return "{}".format(_tp / denominator), "{}/{}".format(_tp, denominator)
    else:
        return "0", "0/0"

def fpr(cm: np.ndarray, target_class: int) -> list:
    _tn = tn(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_fp + _tn)
    if denominator != 0:
        return "{}".format(_fp / denominator), "{}/{}".format(_fp, denominator)
    else:
        return "0", "0/0"

# def tnr(cm: np.ndarray, target_class: int) -> float:
#     return 1 - fpr(cm, target_class)

# def fnr(cm: np.ndarray, target_class: int) -> float:
#     return -1.0

def acc(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fn = fn(cm, target_class)
    _tn = tn(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_tp + _fn + _tn + _fp)
    if denominator != 0:
        return "{}".format((_tp + _tn) / denominator), "{}/{}".format(_tp + _tn, denominator)
    else:
        return "0", "0/0"

def recall(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fn = fn(cm, target_class)
    denominator = (_tp + _fn)
    if denominator != 0:
        return "{}".format(_tp / denominator), "{}/{}".format(_tp, denominator)
    else:
        return "0", "0/0"

def specificity(cm:np.ndarray,target_class:int) -> list:#specificity = tn/(tn+fp)
    _tn = tn(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_tn + _fp)
    if denominator != 0:
        return "{}".format(_tn / denominator), "{}/{}".format(_tn, denominator)
    else:
        return "0", "0/0"

def f1_score(cm:np.ndarray) -> float:
    tpList = [tp(cm,i) for i in range(len(cm))]
    fpList = [fp(cm,i) for i in range(len(cm))]
    fnList = [fn(cm,i) for i in range(len(cm))]
    microP = np.mean(tpList)/(np.mean(tpList)+np.mean(fpList))
    microR = np.mean(tpList)/(np.mean(tpList)+np.mean(fnList))
    micro_F1 = 2*microP*microR/(microP+microR)
    return micro_F1


def precision(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_tp + _fp)
    if denominator != 0:
        return "{}".format(_tp / denominator), "{}/{}".format(_tp, denominator)
    else:
        return "0", "0/0"

def npv(cm: np.ndarray, target_class: int) -> list:
    _tn = tn(cm, target_class)
    _fn = fn(cm, target_class)
    denominator = (_tn + _fn)
    if denominator != 0:
        return "{}".format(_tn / denominator), "{}/{}".format(_tn, denominator)
    else:
        return "0", "0/0"
    
def youden_index(cm: np.ndarray, target_class: int) -> list:
    _tpr, _ = tpr(cm, target_class)
    _fpr, _ = fpr(cm, target_class)
    return float(_tpr) - float(_fpr)

def ci95v2(predict: torch.Tensor):
    # assert predict shape is (N, num_class)
    max_acc, _ = torch.max(predict, dim=1)
    max_acc = max_acc.cpu().numpy()
    ci95_interval = stats.t.interval(0.95, df=len(max_acc) - 1, loc=np.mean(max_acc), scale=stats.sem(max_acc))
    return ci95_interval

def ci95(max_acc: np.ndarray):
    ci95_interval = stats.t.interval(0.95, df=len(max_acc) - 1, loc=np.mean(max_acc), scale=stats.sem(max_acc) + 1e-16)
    return ci95_interval

def roc_and_aucv2(label_set: list, predict_set: list, tag_set: list, n_class: int, ylabel: str="TPR", xlabel: str="FPR", head="", filename=""):
    # [label1, label2]
    plt.cla()
    plt.figure(dpi=300)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    color_set = ["orangered", "cyan", "fuchsia", "darkorange", "mediumseagreen", "cornflowerblue", "darkorchid"]
    for index in range(n_class):
        one_hot_set = label_binarize(label_set, classes=[i for i in range(n_class)])
        if n_class == 2:
            one_hot_set = np.hstack((1 - one_hot_set, one_hot_set))
        fpr, tpr, threshold = roc_curve(one_hot_set[:, index], predict_set[:, index])
        auc_value = auc(fpr, tpr)
        auc_95_list = []
        for auc_95_index in range(4):
            auc_95_fpr, auc_95_tpr, _ = roc_curve(one_hot_set[auc_95_index * 30:(auc_95_index+1) * 30, index], predict_set[auc_95_index * 30:(auc_95_index+1) * 30, index])
            auc_95_list.append(auc(auc_95_fpr, auc_95_tpr))
        auc_95_value = ci95(np.array(auc_95_list))
        auc_95_value = [min(max(item, 0), 1) for item in auc_95_value]
        print("类别:{} | AUC为{} | 95%CI为{}".format(tag_set[index], auc_value, auc_95_value))
        plt.plot(fpr, tpr, color=color_set[index], label="{}, AUC={:.4f}".format(tag_set[index], auc_value))
    plt.legend(loc=0)
    plt.savefig(filename)

def cm(c_matrix, class_list, figure_name="cm_smallZuwai.png"):
    plt.cla()
    c_matrix = c_matrix.astype(np.float32)
    assert len(class_list) == c_matrix.shape[0] == c_matrix.shape[1]
    cmap = plt.cm.get_cmap('Blues')
    plt.figsize=(10, 10)
    plt.imshow(c_matrix, cmap=cmap)
    plt.colorbar()
    xlocations = np.array(range(len(class_list)))
    plt.xticks(xlocations, class_list, rotation=60)
    plt.yticks(xlocations, class_list)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix')
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            plt.text(x=j, y=i, s=int(c_matrix[i, j]), va='center', ha='center', color='black', fontsize=10)
    plt.savefig(figure_name,dpi=500)