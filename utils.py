import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import shutil
import os
import fnmatch
from PIL import Image

def countImg(fileroot):
    count = 0
    for file in os.listdir(fileroot):
        filelist = fnmatch.filter(os.listdir(os.path.join(fileroot,file)), '*.jpg')
        count += len(filelist)
        return count

def countSonFileImg(fileroot):
    data = pd.read_excel(r"...")
    data = np.array(data).tolist()
    count = 0
    for row in data:
        file_path = os.path.join(fileroot,str(row[8]))
        file_list = fnmatch.filter(os.listdir(file_path), '*.jpg')
        count += len(file_list)
    return count

def labelChange(x):
    if x == 0 or x == 1:
        return 0
    elif x == 2:
        return 1

def changeAllLabel(fileroot):
    data = pd.read_excel(fileroot)
    data['关节镜结果（正常0，磨损1，分离/撕裂2）'] = data['关节镜结果（正常0，磨损1，分离/撕裂2）'].apply(lambda x:labelChange(x))
    save_data = data
    file_name= 'D:\SLAP\slapchangeLabel.xlsx' 
    save_data.to_excel(file_name, index = False)

def count(fileroot):
    return len(os.listdir(fileroot))

def countEXCEL(fileroot):
    data= pd.read_excel(fileroot)
    print(data['关节镜结果（正常0，磨损1，分离/撕裂2）'].value_counts())

def sequnce(fileroot):
    data = pd.read_excel(fileroot)
    data = data.sort_values(axis=0,ascending=True,by='关节镜结果（正常0，磨损1，分离/撕裂2）')
    save_data = data
    file_name= 'D:\SLAP\slapSequence(2).xlsx' 
    save_data.to_excel(file_name, index = False)

def cutXlsx(fileroot):
    data = pd.read_excel(fileroot)
    row_num, column_num = data.shape
    save_positive_data = data.iloc[0:396]
    file_name= '...'
    save_positive_data.to_excel(file_name, index = False)
    save_negetive_data = data.iloc[396:636]
    file_name= '...'
    save_negetive_data.to_excel(file_name, index = False)

def splitXlsx():
    data_pos = pd.read_excel(r'...')
    data_neg = pd.read_excel(r'...')

    data_pos_train = data_pos.iloc[0:int(396*0.8)]
    data_pos_valid = data_pos.iloc[int(396*0.8):int(396*0.8)]
    data_pos_test = data_pos.iloc[int(396*0.9)::]
    data_neg_train = data_neg.iloc[0:int(240*0.8)]
    data_neg_valid = data_neg.iloc[int(240*0.8):int(240*0.9)]
    data_neg_test = data_neg.iloc[int(240*0.9)::]

    data_train = [data_pos_train,data_neg_train]
    data_train = pd.concat(data_train)
    data_train = data_train.iloc[:,[8,4]]

    data_valid = [data_pos_valid,data_neg_valid]
    data_valid = pd.concat(data_valid)
    data_valid = data_valid.iloc[:,[8,4]]

    data_test = [data_pos_test,data_neg_test]
    data_test = pd.concat(data_test)
    data_test = data_test.iloc[:,[8,4]]

    file_name= r'...'
    data_train.to_excel(file_name, index = False)
    file_name= r'...'
    data_valid.to_excel(file_name, index = False)
    file_name= r'...'
    data_test.to_excel(file_name, index = False)
def preDivideDataset(fileroot):
    """
    put all the same type in one folder
    """
    positiveData_folder = os.makedirs(os.path.join(fileroot,"AllData","0"),exist_ok=True)
    negetiveData_folder = os.makedirs(os.path.join(fileroot,"AllData","1"),exist_ok=True)

    all_data = {}
    positive_data = pd.read_excel(r"...")
    positive_data = np.array(positive_data).tolist()
    all_data['positive'] = positive_data
    negetive_data = pd.read_excel(r"...")
    negetive_data = np.array(negetive_data).tolist()
    all_data['negetive'] = negetive_data
    dst_pos = os.path.join(fileroot,"AllData","0")
    dst_neg = os.path.join(fileroot,"AllData","1")
    for datas in all_data.keys():
        for info in all_data[datas]:
            imgPath = os.path.join(fileroot,info[8])
            imgList = fnmatch.filter(os.listdir(imgPath), '*.jpg')
            for img in imgList:
                src = os.path.join(imgPath,img)
                if info[4]==0:
                    shutil.copy(src,dst_pos)
                else:
                    shutil.copy(src,dst_neg)

def divideDataset(pos_Path,neg_Path):
    # create folders
    train_positive_folder = os.path.join(fileroot,'train','0')
    os.makedirs(train_positive_folder,exist_ok=True)

    train_negetive_folder = os.path.join(fileroot,'train','1')
    os.makedirs(train_negetive_folder,exist_ok=True)

    valid_positive_folder = os.path.join(fileroot,'valid','0')
    os.makedirs(valid_positive_folder,exist_ok=True)

    valid_negetive_folder = os.path.join(fileroot,'valid','1')
    os.makedirs(valid_negetive_folder,exist_ok=True)

    test_positive_folder = os.path.join(fileroot,'test','0')
    os.makedirs(test_positive_folder,exist_ok=True)

    test_negetive_folder = os.path.join(fileroot,'test','1')
    os.makedirs(test_negetive_folder,exist_ok=True)

    percentage = {'train':0.8,"valid":0.9,"test":1}

    for img in pos_Path:
        img_list = os.listdir(pos_Path)
        for i in range(0,int(percentage['train']*len(img_list))):
            src = os.path.join(pos_Path,img_list[i])
            dst = train_positive_folder
            shutil.copy(src,dst)
        for i in range(int(percentage['train']*len(img_list)),int(percentage['valid']*len(img_list))):
            src = os.path.join(pos_Path,img_list[i])
            dst = valid_positive_folder
            shutil.copy(src,dst)
        for i in range(int(percentage['valid']*len(img_list)),len(img_list)):
            src = os.path.join(pos_Path,img_list[i])
            dst = test_positive_folder
            shutil.copy(src,dst)
    for img in neg_Path:
        img_list = os.listdir(neg_Path)
        for i in range(0,int(percentage['train']*len(img_list))):
            src = os.path.join(neg_Path,img_list[i])
            dst = train_negetive_folder
            shutil.copy(src,dst)
        for i in range(int(percentage['train']*len(img_list)),int(percentage['valid']*len(img_list))):
            src = os.path.join(neg_Path,img_list[i])
            dst = valid_negetive_folder
            shutil.copy(src,dst)
        for i in range(int(percentage['valid']*len(img_list)),len(img_list)):
            src = os.path.join(neg_Path,img_list[i])
            dst = test_negetive_folder
            shutil.copy(src,dst)
    
def compare():
    root1 = r"..."
    root2 = r"..."
    list1 = os.listdir(root1)
    list2 = os.listdir(root2)
    print(os.listdir(root1)==os.listdir(root2))
def findMin(fileroot):
    img_name = os.listdir(fileroot)
    imgs = []
    for name in img_name:
        img_path = os.path.join(fileroot,name)
        img = Image.open(img_path)
        imgSize = img.size  #大小/尺寸
        imgs.append(imgSize)
        
    return(min(imgs))

fileroot = r"..."

