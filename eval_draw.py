import os
from pickletools import optimize
import time
import torch
import numpy as np
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from model import *
from loss import *
import static


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ROIdataset = '...'
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.ColorJitter(brightness=(1.0,1.7)),
    torchvision.transforms.ToTensor()
])
test_set = ImageFolder(os.path.join(ROIdataset,'test'), transform=test_transform)
test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=0)


def test_epoch(model,splits, loader):
    model.eval()
    end = time.time()
    cm = None
    model_pred = []
    model_label = []
    with torch.no_grad():
        for batch_idx, (images,labels) in enumerate(loader):

            images, labels = images.to(device),labels.to(device)
            images = Variable(images)
            labels = Variable(labels)

            # compute output
            outputs = model(images)
            model_pred.append(outputs.cpu().numpy()[0])
            model_label.append(labels.cpu().numpy()[0])
            
            pred = outputs.max(1, keepdim=True)[1]
            end = time.time()
            if cm is None:
                cm = static.confusion_matrix(labels, pred.view(-1), 2)
            else:
                cm += static.confusion_matrix(labels, pred.view(-1), 2)
        for class_idx in range(2):
            print("############# CID {} TEST #############".format(class_idx))
            print("Class idx {}".format(class_idx))
            print("CM: \n{}".format(cm))
            print("ACC: {}".format(static.acc(cm, class_idx)))
            print("RECALL :{}".format(static.recall(cm, class_idx)))
            print("PPV: {}".format(static.precision(cm, class_idx)))
            print("NPV: {}".format(static.npv(cm, class_idx)))
        static.roc_and_aucv2(model_label, np.array(model_pred), ["class 0", "class 1"], 2, filename="auc.png")
        static.cm(cm, ["Class 0", "Class 1"], "cm.png")


if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load("..."))
    model.to(device)
    test_epoch(model, "valid", test_set)