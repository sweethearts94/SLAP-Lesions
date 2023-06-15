import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import shutil
import os
import re
import PIL
from PIL import Image
import PIL.ImageOps   
def labelChange(x):
    if x == 0 or x == 1:
        return 0
    elif x == 2:
        return 1

def changeAllLabel(fileroot):
    positive,negetive = 0,0
    data = pd.read_excel(fileroot)
    # data = np.array(data).tolist()
    # name = data[0]
    # for i in range(len(data)):
    #     if  data[i][4]== 1:
    #         data[i][4] = 0
    #         positive += 1
    #     elif   data[i][4] == 2:
    #         data[i][4] = 1
    #         negetive += 1
    #     elif  data[i][4] == 0:
    #         positive += 1
    #         continue
    
    # data = pd.DataFrame(data)
    data['关节镜结果（正常0，磨损1，分离/撕裂2）'] = data['关节镜结果（正常0，磨损1，分离/撕裂2）'].apply(lambda x:labelChange(x))
    save_data = data
    file_name= 'D:\SLAP\slapchangeLabel.xlsx' 
    save_data.to_excel(file_name, index = False)

def count(fileroot):
    data= pd.read_excel(fileroot)
    print(data['关节镜结果（正常0，磨损1，分离/撕裂2）'].value_counts())

def sequnce(fileroot):
    data = pd.read_excel(fileroot)
    data = data.sort_values(axis=0,ascending=True,by='关节镜结果（正常0，磨损1，分离/撕裂2）')
    save_data = data
    file_name= '...' 
    save_data.to_excel(file_name, index = False)

def cutXlsx(fileroot):
    data = pd.read_excel(fileroot)
    row_num, column_num = data.shape    #数据共有多少行，多少列
    save_positive_data = data.iloc[0:396] #每隔1万循环一次
    file_name= '...'
    save_positive_data.to_excel(file_name, index = False)
    save_negetive_data = data.iloc[396:636] #每隔1万循环一次
    file_name= '...'
    save_negetive_data.to_excel(file_name, index = False)

def splitXlsx(fileroot):
    data_pos = pd.read_excel(r'...')
    data_neg = pd.read_excel(r'...')
    os.makedirs(os.path.join(fileroot,'splits'),exist_ok=True)
    data_pos_train = data_pos.iloc[0:int(396*0.8)]
    data_pos_valid = data_pos.iloc[int(396*0.8):int(396*0.9)]
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

def divideDataset(fileroot):
    # # create folders
    train_positive_folder = os.makedirs(os.path.join(fileroot,'slapnotROIDataset','train','0'),exist_ok=True)
    train_negetive_folder = os.makedirs(os.path.join(fileroot,'slapnotROIDataset','train','1'),exist_ok=True)
    valid_positive_folder = os.makedirs(os.path.join(fileroot,'slapnotROIDataset','valid','0'),exist_ok=True)
    valid_negetive_folder = os.makedirs(os.path.join(fileroot,'slapnotROIDataset','valid','1'),exist_ok=True)
    test_positive_folder = os.makedirs(os.path.join(fileroot,'slapnotROIDataset','test','0'),exist_ok=True)
    test_negetive_folder = os.makedirs(os.path.join(fileroot,'slapnotROIDataset','test','1'),exist_ok=True)
    # read the splits files
    all_data = {}
    train_data = pd.read_excel(os.path.join(fileroot,'splits','train.xlsx'))
    train_data = np.array(train_data).tolist()
    all_data['train'] = train_data
    valid_data = pd.read_excel(os.path.join(fileroot,'splits','valid.xlsx'))
    valid_data = np.array(valid_data).tolist()
    all_data['valid'] = valid_data
    test_data = pd.read_excel(os.path.join(fileroot,'splits','test.xlsx'))
    test_data = np.array(test_data).tolist()
    all_data['test'] = test_data
    for key in all_data.keys():
    # copy files from all folders
        for data in all_data[key]:
            TargetRoot = os.path.join(fileroot,'slap','notROI',str(data[0]))
            if not os.path.exists(TargetRoot): 
                TargetRoot = os.path.join(fileroot,'slap','notROI','0'+str(data[0]))
            destinateRoot= os.path.join(fileroot,'slapnotROIDataset',key,str(data[1]))
            filelist = os.listdir(TargetRoot)
            for item in filelist:    #遍历
                src = os.path.join(os.path.join(TargetRoot), item)
                dst = os.path.join(os.path.join(destinateRoot),item)
                shutil.copy(src,dst) #将src复制到dst

def findROI():
    stepRoot = r'...'
    slapFolder = os.listdir(stepRoot)
    for file in slapFolder:
        filePath = os.path.join(stepRoot,file)
        filePathFolder = os.listdir(filePath)
        for img in filePathFolder:
            if img[:-4]+'.xml' in filePathFolder:
                src = os.path.join(stepRoot,filePath,img[:-4]+'.jpg')
                os.makedirs(os.path.join(r'...',file),exist_ok=True)  
                dst = os.path.join(r'...',file,img[:-4]+'.jpg')
                shutil.copy(src,dst)
            else:
                src = os.path.join(stepRoot,filePath,img[:-4]+'.jpg')
                os.makedirs(os.path.join(r'...',file),exist_ok=True)  
                dst = os.path.join(r'...',file,img[:-4]+'.jpg')
                shutil.copy(src,dst) #将src复制到dst

def reverse(fileroot):
    stages = os.listdir(fileroot)
    for stage in stages:
        folders = os.listdir(os.path.join(fileroot,stage))
        for folder in folders:
            imgs = os.listdir(os.path.join(fileroot,stage,folder))
            for img in imgs:
                imgPath = os.path.join(fileroot,stage,folder,img)
                reverseImg = Image.open(imgPath)
                reverseImg = PIL.ImageOps.invert(reverseImg)
                reverseImg.save(os.path.join(fileroot,stage,folder,img[:-4]+'reverse.jpg'))

fileroot = r"..."
# changeAllLabel(fileroot)
# count(fileroot)
# sequnce(fileroot)
# cutXlsx(fileroot)
# splitXlsx(r"D:\SLAP")
# divideDataset(r"D:\SLAP")
# findROI()
reverse("...")