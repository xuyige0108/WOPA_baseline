import os
import re
import pandas as pd
from common_utils import *

root='../output'
dataset='vlcs'
project='NS80_lr5_arcface_E50_5_0.5_resnet50_clip'
pacs=['art_painting','cartoon','photo','sketch']


'''
=> result
* total: 3,928
* correct: 3,174
* accuracy: 80.8%
* error: 19.2%
* macro_f1: 82.8%
'''

def txt_process(txt_path):
    file = open(txt_path, 'r')
    content = file.read()
    pattern=r"\* accuracy: (\d+\.\d+)"

    acc_epoch_list=[]
    matches_test_acc = re.findall(pattern , content)
    matches_test_acc = [float(element) for element in matches_test_acc] 
    epochs=len(matches_test_acc)//4
    for j in range(epochs):
        acc=0
        #print(matches_test_acc[j*4],matches_test_acc[j*4+1],matches_test_acc[j*4+2],matches_test_acc[j*4+3])
        acc+=matches_test_acc[j*4]+matches_test_acc[j*4+1]+matches_test_acc[j*4+2]+matches_test_acc[j*4+3]
        acc/=4
        acc_epoch_list.append(acc)
    
    final_acc=acc_epoch_list[-1]
    final_epoch_list=matches_test_acc[-4:]

    acc_list_with_indices = sorted(enumerate(acc_epoch_list), key=lambda x: x[1], reverse=True)
    max_epoch,max_acc=acc_list_with_indices[0]
    max_epoch_list=[matches_test_acc[max_epoch*4],matches_test_acc[max_epoch*4+1],matches_test_acc[max_epoch*4+2],matches_test_acc[max_epoch*4+3]]
    return max_epoch,max_acc,max_epoch_list,final_acc,final_epoch_list


if __name__=='__main__':
    project_dir=os.path.join(root,dataset,project)
    #class_project_dir=[os.path.join(project_dir,i) for i in pacs ]

    pacs_acc_dict=dict()
    floder=os.listdir(project_dir)
    floder=['seed1','seed2','seed3']

    for i in range(3):
        average_acc=0
        seed_dir_path=os.path.join(project_dir,floder[i])  
        print(seed_dir_path)
        txt_files_path=list_with_suffix(seed_dir_path, suffix='.txt', with_dir_path=True)
        max_epoch,max_acc,max_epoch_list,final_acc,final_epoch_list=txt_process(txt_files_path[0])
        print("max_epoch: ",max_epoch+1)
        print("max_acc: ", max_acc)
        print("max_acc_list")
        print(max_epoch_list)

        print("final_acc: ",final_acc )
        print("final_acc_list")
        print(final_epoch_list)


        
        
        



