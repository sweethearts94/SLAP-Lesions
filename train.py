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
# device = 'cpu'
best_acc = 0
n_epochs = 100
batch_size = 64
learning_rate = 1e-5
weight = [1,1.6]
dataset = '...'
ROIdataset = '...'
save = '...'

# Data transforms
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=(1.0,1.7)),
    # torchvision.transforms.RandomRotation((-15,+15)),
    torchvision.transforms.ToTensor(),
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.ColorJitter(brightness=(1.0,1.7)),
    torchvision.transforms.ToTensor()
])

ROItrain_set = ImageFolder(os.path.join(ROIdataset,'train'), transform=train_transform)
ROIvalid_set = ImageFolder(os.path.join(ROIdataset, "valid"), transform=test_transform)
train_set = ImageFolder(os.path.join(dataset,'train'), transform=train_transform)
valid_set = ImageFolder(os.path.join(dataset,'valid'), transform=test_transform)
test_set = ImageFolder(os.path.join(dataset,'test'), transform=test_transform)

ROItrain_set = torch.utils.data.DataLoader(dataset=ROItrain_set, batch_size=batch_size, shuffle=True, num_workers=0)
ROIvalid_set = torch.utils.data.DataLoader(dataset=ROIvalid_set, batch_size=batch_size, shuffle=True, num_workers=0)
train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
valid_set = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)

loss_fn = focal_loss(alpha=0.1, gamma=2, num_classes=2)

# criterion = F.cross_entropy(weight=torch.Tensor(weight))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        args(val(value),n=1(number))
        val = val,sum += val*n,count+=n,avg = sum/count
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=100):
    """
    train stage
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    cm = None
    # Model on train mode
    model.train()
    TP,TN,FP,FN =0,0,0,0
    end = time.time()
    for batch_idx, (images,labels) in enumerate(loader):
        images,labels = images.to(device),labels.to(device)
        
        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        # loss = torch.nn.functional.cross_entropy(outputs, labels)
        # loss = loss_fn(outputs, labels)
        loss = F.cross_entropy(weight=torch.Tensor(weight).to(device),input=outputs,target=labels).to(device)
        pred = outputs.max(1, keepdim=True)[1]

        # measure accuracy and record loss
        batch_size = labels.size(0)
        acc.update(val=pred.eq(labels.view_as(pred)).sum().item() / batch_size, n=batch_size)
        losses.update(val=loss.item() / batch_size, n=batch_size)
        tn,tp,fn,fp = _label_sum(pred, labels)
        TN += tn
        TP += tp
        FN += fn
        FP += fp
        if cm is None:
            cm = static.confusion_matrix(labels, pred.view(-1), 2)
        else:
            cm += static.confusion_matrix(labels, pred.view(-1), 2)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    for class_idx in range(2):
        print("############# EPOCH {} CID {} TRAIN #############".format(epoch, class_idx))
        print("Class idx {}".format(class_idx))
        print("CM: \n{}".format(cm))
        print("ACC: {}".format(static.acc(cm, class_idx)))
        print("RECALL :{}".format(static.recall(cm, class_idx)))
        print("PPV: {}".format(static.precision(cm, class_idx)))
        print("NPV: {}".format(static.npv(cm, class_idx)))
    with open(os.path.join(save, 'train_matrix.csv'), 'a') as f:
        f.write('%d,%d,%d,%d\n' % (
            TP, FP, FN, TN))
    return batch_time.avg, losses.avg, acc.avg

def _label_sum(pred, target):
    """
    count positive and negetive samples how many of them have been correctly predticted,
    and the number of correctly predicted samples in all samples
    """
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    TList = []
    FList = []
    for i in range(len(target)):
        TList.append(pred[i] + target[i])
    for i in range(len(target)):
        if pred[i] == 0 and target[i]==1:
            FList.append(0)
        elif pred[i] == 1 and target[i] ==0:
            FList.append(1)
    return TList.count(0), TList.count(2), FList.count(0),FList.count(1)

def test_epoch(model,splits, loader, epoch, is_test=True):
    """
    test stage
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    
    TP,TN,FP,FN =0,0,0,0
    TPR,TNR,FPR,FNR = 0,0,0,0

    # Model on eval mode
    model.eval()

    end = time.time()
    cm = None
    with torch.no_grad():
        for batch_idx, (images,labels) in enumerate(loader):

            images, labels = images.to(device),labels.to(device)
            images = Variable(images)
            labels = Variable(labels)

            # compute output
            outputs = model(images)
            
            pred = outputs.max(1, keepdim=True)[1]
            # loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss = F.cross_entropy(weight=torch.Tensor(weight).to(device),input=outputs,target=labels).to(device)

            # measure accuracy and record loss
            batch_size = labels.size(0)
            acc.update(pred.eq(labels.view_as(pred)).sum().item() / batch_size, batch_size)
            losses.update(loss.item() / batch_size, batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            tn,tp,fn,fp = _label_sum(pred, labels)
            TN += tn
            TP += tp
            FN += fn
            FP += fp
            if cm is None:
                cm = static.confusion_matrix(labels, pred.view(-1), 2)
            else:
                cm += static.confusion_matrix(labels, pred.view(-1), 2)
        for class_idx in range(2):
            print("############# EPOCH {} CID {} TEST #############".format(epoch, class_idx))
            print("Class idx {}".format(class_idx))
            print("CM: \n{}".format(cm))
            print("ACC: {}".format(static.acc(cm, class_idx)))
            print("RECALL :{}".format(static.recall(cm, class_idx)))
            print("PPV: {}".format(static.precision(cm, class_idx)))
            print("NPV: {}".format(static.npv(cm, class_idx)))
        datasetPath = os.path.join(dataset,splits)
        H = len(os.listdir(os.path.join(datasetPath,'1')))
        U = len(os.listdir(os.path.join(datasetPath,'0')))
        TPR = TP / H
        FPR = FP / U
        FNR = FN / H
        TNR = TN / U
        

        res = '\t'.join([
            'Test:' if is_test else 'Valid:',
            'Time %.3f s' % batch_time.sum,
            'Loss %.4f ' % losses.avg,
            'Acc %.4f ' % acc.avg,
            'TP %d' % TP,
            'FP %d' % FP,
            'FN %d' % FN,
            'TN %d' % TN
        ])
        with open(os.path.join(save, 'test_matrix.csv'), 'a') as f:
            f.write('%d,%d,%d,%d\n' % (
                TP, FP, FN, TN))

    return batch_time.avg, losses.avg, acc.avg

def train(model, train_set, valid_set, test_set, save, n_epochs=200,
          lr=learning_rate, wd=0.0001, momentum=0.95, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model = model.to(device)
    
    # InitWeight
    def initWeights(module):
        if type(module) == nn.Conv2d and module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    model.apply(initWeights)

    model_wrapper = model
    num_params = sum(p.numel() for p in model.parameters())
    record_full_file = open(os.path.join(save, 'acc_best.txt'), 'a')
    print("Total parameters: {},model:{}".format(num_params,model),file=record_full_file)
    record_full_file.close()
    # optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, weight_decay=wd) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, 1e-6)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_acc,valid_loss,valid_acc,test_acc,lr\n')
    with open(os.path.join(save, 'train_matrix.csv'), 'w') as f:
        f.write('TP,FP,FN,TN\n')
    with open(os.path.join(save, 'test_matrix.csv'), 'w') as f:
            f.write('TP,FP,FN,TN\n')

    # Train model
    best_acc = 0
    for epoch in range(n_epochs):
        if epoch < n_epochs:
            _, train_loss, train_acc = train_epoch(
                model=model_wrapper,
                loader=ROItrain_set,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
            )
        else:
            _, train_loss, train_acc = train_epoch(
                model=model_wrapper,
                loader=train_set,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
            )
        torch.save(model.state_dict(), os.path.join(save, 'model_{}.pkl'.format(epoch)))
        scheduler.step()
        _, valid_loss, valid_acc = test_epoch(
            model=model_wrapper,
            splits='valid',
            epoch=epoch,
            loader=ROIvalid_set if ROIvalid_set else test_set,
            # loader = ROItrain_set,
            is_test=(not valid_set)
        )
        record_full_file = open(os.path.join(save, 'acc_best.txt'), 'a')
        print('{}, {:.5f}'.format(epoch+1,valid_acc),file=record_full_file)
        record_full_file.close()
        # Determine if model is the best
        if valid_set:
            if valid_acc > best_acc:
                best_acc = valid_acc
                print('New best acc: %.4f' % best_acc)
                record_file = open(os.path.join(save, 'acc_best.txt'), 'a')
                print('{}, {:.5f}'.format(epoch+1,best_acc),file=record_file)
                record_file.close()

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,%f\n' % (
                (epoch + 1), train_loss, train_acc, valid_loss, valid_acc,optimizer.state_dict()['param_groups'][0]['lr']
            ))

    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(save, 'model.pkl')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model1 = torch.nn.DataParallel(model).to(device)
    test_results = test_epoch(
        model=model,
        splits='test',
        loader=test_set,
        is_test=True
    )
    _, _, test_acc = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % test_acc)
    print('Final test acc: %.4f' % test_acc)

def main(seed=None):
    model = LeNet()
    cudnn.benchmark = True 

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save,
          n_epochs=n_epochs, seed=seed)
    print('Done!')

if __name__ == '__main__':
    main()