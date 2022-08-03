import argparse
import glob
from collections import defaultdict
import os
import numpy as np
import random
from tqdm import tqdm
from collections import Counter
import pickle
import math
import torchvision.transforms as transforms
from torch.nn import Parameter
from PIL import Image
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torchvision.models as models
import time
import logging
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
import json
from torch.utils.tensorboard import SummaryWriter
from math import sqrt
import copy
def get_logger(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(fmt="%(message)s")
    
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    
    if not os.path.exists(args.WORK_DIR):
        os.makedirs(args.WORK_DIR)
    
    if not os.path.exists(args.WORK_DIR+'/log'):
        os.makedirs(args.WORK_DIR+'/log')
    
    fHandler = logging.FileHandler(args.WORK_DIR +'log/'+ file_name+'.log', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    return logger


class AidDataSet:
    def __init__(self, transform=None, npy_file="/Public/YongkunLiu/DataSet/aid_ml.npy",img_path="None",inp_name=None, adj=None):
        self.target_transform = target_transform
        self.npy_file=npy_file
        self.img_path=img_path

def default_loader(path):
    return Image.open(path).convert('RGB')

class DataGeneratorML:

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train',inp_name=""):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        
        self.head_nums=7
        
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
            self.inp =torch.tensor(self.inp, dtype=torch.float32)
            print(self.inp)

        self.CreateIdx2fileDict()
    
    def CreateIdx2fileDict(self):
        random.seed(42)

        if self.dataset == 'AID_multilabel':
            data = np.load(AID_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'UCMerced':
            data = np.load(UCM_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'DFC15_multilabel':
            data = np.load(DFC_ML_npy, allow_pickle=True).item()

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        if self.dataset != 'DFC15_multilabel':
            for _, scenePth in enumerate(self.sceneList):

                subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
                # random.seed(1)
                
                random.shuffle(subdirImgPth)

                train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
                val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
                test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

                self.train_numImgs += len(train_subdirImgPth)
                self.test_numImgs += len(test_subdirImgPth)
                self.val_numImgs += len(val_subdirImgPth)

                for imgPth in train_subdirImgPth:
                    
                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.train_idx2fileDict[train_count] = (imgPth, multi_hot)
                    train_count += 1
                
                for imgPth in test_subdirImgPth:

                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.test_idx2fileDict[test_count] = (imgPth, multi_hot)
                    test_count += 1

                for imgPth in val_subdirImgPth:

                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.val_idx2fileDict[val_count] = (imgPth, multi_hot)
                    val_count += 1
        else:

            imgPths = sorted(glob.glob(os.path.join(self.datadir, '*.'+self.imgExt)))
            # random.seed(1)
            random.shuffle(imgPths)

            train_subdirImgPth = imgPths[:int(0.7*len(imgPths))]
            val_subdirImgPth = imgPths[int(0.7*len(imgPths)):int(0.8*len(imgPths))]
            test_subdirImgPth = imgPths[int(0.8*len(imgPths)):]

            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                    
                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.train_idx2fileDict[train_count] = (imgPth, multi_hot)
                train_count += 1
            
            for imgPth in test_subdirImgPth:

                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.test_idx2fileDict[test_count] = (imgPth, multi_hot)
                test_count += 1

            for imgPth in val_subdirImgPth:

                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.val_idx2fileDict[val_count] = (imgPth, multi_hot)
                val_count += 1

        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)
    
    def __data_generation(self, idx):

        if self.phase == 'train':
            imgPth, multi_hot = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, multi_hot = self.val_idx2fileDict[idx]
        else:
            imgPth, multi_hot = self.test_idx2fileDict[idx]
        
        img = default_loader(imgPth)
        
        bin2int = int(''.join(list(map(str, multi_hot.tolist()))), 2)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        return (img,self.inp),multi_hot.astype(np.float32),imgPth

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)
        
        

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = _adj[_adj >= t]
#     print(_adj)
    #为什么要乘0.25
    # _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    # _adj = _adj + np.identity(num_classes, int)
    return _adj

# def gen_adj(A):
#     D = torch.pow(A.sum(1).float(), -0.5)
#     D = torch.diag(D)
#     adj = torch.matmul(torch.matmul(A, D).t(), D)
#     return adj

# def GNNattention(query: Tensor, 
#               key: Tensor, 
#               value: Tensor, 
#               mask: Optional[Tensor] = None, 
#               dropout: float = 0.1):
#     """
#     Define how to calculate attention score
#     Args:
#         query: shape (batch_size, num_heads, seq_len, k_dim)
#         key: shape(batch_size, num_heads, seq_len, k_dim)
#         value: shape(batch_size, num_heads, seq_len, v_dim)
#         mask: shape (batch_size, num_heads, seq_len, seq_len). Since our assumption, here the shape is
#               (1, 1, seq_len, seq_len)
#     Return:
#         out: shape (batch_size, v_dim). Output of an attention head.
#         attention_score: shape (seq_len, seq_len).

#     """
#     k_dim = query.size(-1)

#     # shape (seq_len ,seq_len)，row: token，col: that token's attention score
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_dim)
        
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e10)

#     attention_score = F.softmax(scores, dim = -1)

#     if dropout is not None:
#         attention_score = dropout(attention_score)
        
#     out = torch.matmul(attention_score, value)
    
#     return out, attention_score # shape: (seq_len, v_dim), (seq_len, seq_lem)
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None,graph_type="GAT",res="True"):
        super(GCNResnet, self).__init__()
        if args.MODEL=="vgg16":#懒得改，先这样
            self.model=model
            # self.model.classifier=self.model.classifier[:-1]
            model.classifier[-1]=nn.Linear(4096,num_classes)
        else:
            self.model=model
            # self.features = nn.Sequential(
            #     model.conv1,
            #     model.bn1,
            #     model.relu,
            #     model.maxpool,
            #     model.layer1,
            #     model.layer2,
            #     model.layer3,
            #     model.layer4,
            # )
# layer1:torch.Size([2, 256, 56, 56])
# layer2:torch.Size([2, 512, 28, 28])
# layer3:torch.Size([2, 1024, 14, 14])
# layer4:torch.Size([2, 2048, 7, 7])
        self.num_classes = num_classes
        self.pooling = nn.AdaptiveAvgPool2d([1,1])#nn.MaxPool2d(8, 8)
        self.res=res
        
        self.GRAPH=graph_type
        #kun结构
        self.kun1={
          'conv11':nn.AdaptiveAvgPool2d([1,1]),
          'Q':clones(nn.Linear(1,num_classes),args.HEADS_NUM),
          'V': clones(nn.Linear(1,1),args.HEADS_NUM),
        }
        
        self.kun2={
          'conv11': nn.AdaptiveAvgPool2d([1,1]),
          'Q':clones(nn.Linear(1,num_classes),args.HEADS_NUM),
          'V': clones(nn.Linear(1,1),args.HEADS_NUM),
        }
        self.kun3={
          'conv11':nn.AdaptiveAvgPool2d([1,1]),
          'Q':clones(nn.Linear(1,num_classes),args.HEADS_NUM),
          'V': clones(nn.Linear(1,1),args.HEADS_NUM),
        }
        self.kun4={
          'conv11': nn.AdaptiveAvgPool2d([1,1]),
          'Q':clones(nn.Linear(1,num_classes),args.HEADS_NUM),
          'V': clones(nn.Linear(1,1),args.HEADS_NUM),
            
          'Q2':clones(nn.Linear(1,num_classes),args.HEADS_NUM),
          'K2':clones(nn.Linear(1,num_classes),args.HEADS_NUM),
          'V2': clones(nn.Linear(1,1),args.HEADS_NUM)
        }
        
        
        # self.conv11=nn.AdaptiveAvgPool2d([1,1])
        # Q=nn.Linear(1,num_classes)#1,17
        # V=nn.Linear(1,1)
        
        
        
        #self attention
        input_dim,dim_k,dim_v=1,args.AT_SIZE,1
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        if args.MODEL=='resnet50' or args.MODEL=="resnet101" or args.MODEL=="wideresnet50":
            if args.LAYER=='1':
                self.fc_last=nn.Linear(args.HEADS_NUM*256,256)
            if args.LAYER=='2':
                self.fc_last=nn.Linear(args.HEADS_NUM*512,512)
            if args.LAYER=='3':
                self.fc_last=nn.Linear(args.HEADS_NUM*1024,1024)
            if args.LAYER=='4':
                self.fc_last=nn.Linear(args.HEADS_NUM*2048,2048)
        if args.MODEL=="resnet34" or args.MODEL=='resnet18':
            if args.LAYER=='1':
                self.fc_last=nn.Linear(args.HEADS_NUM*64,64)
            if args.LAYER=='2':
                self.fc_last=nn.Linear(args.HEADS_NUM*128,128)
            if args.LAYER=='3':
                self.fc_last=nn.Linear(args.HEADS_NUM*256,256)
            if args.LAYER=='4':
                self.fc_last=nn.Linear(args.HEADS_NUM*512,512)
        if args.MODEL=="vgg16":
            self.fc_last=nn.Linear(args.HEADS_NUM*512,512)

        
        if graph_type=="GCN":
            if args.MODEL=="resnet50" or args.MODEL=="resnet101" or args.MODEL=="wideresnet50":
                self.gc1 = clones(GCNConv(in_channel, 256),args.HEADS_NUM)
                self.gc2 = clones(GCNConv(in_channel, 512),args.HEADS_NUM)
                self.gc3={
                    'gcn1':clones(GCNConv(in_channel, 512),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GCNConv(512, 1024),args.HEADS_NUM),
                }

                self.gc4={
                    'gcn1':clones(GCNConv(in_channel, 1024),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GCNConv(1024, 2048),args.HEADS_NUM),
                }
# torch.Size([16, 64, 64, 64]) resnet18,34
# torch.Size([16, 128, 32, 32])
# torch.Size([16, 256, 16, 16])
# torch.Size([16, 512, 8, 8])

            elif args.MODEL=="resnet34" or args.MODEL=='resnet18' or args.MODEL=="vgg16":
                self.gc1 = clones(GCNConv(in_channel, 64),args.HEADS_NUM)
                self.gc2 = clones(GCNConv(in_channel, 128),args.HEADS_NUM)
                self.gc3={
                    'gcn1':clones(GCNConv(in_channel, 128),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GCNConv(128, 256),args.HEADS_NUM),
                }

                self.gc4={
                    'gcn1':clones(GCNConv(in_channel, 256),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GCNConv(256, 512),args.HEADS_NUM),
                }
                
        elif graph_type=="SAGE":
            if args.MODEL=="resnet50" or args.MODEL=="resnet101" or args.MODEL=="wideresnet50":
                self.gc1 = clones(SAGEConv(in_channel, 256),args.HEADS_NUM)
                self.gc2 = clones(SAGEConv(in_channel, 512),args.HEADS_NUM)
                self.gc3={
                    'gcn1':clones(SAGEConv(in_channel, 512),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(SAGEConv(512, 1024),args.HEADS_NUM),
                }

                self.gc4={
                    'gcn1':clones(SAGEConv(in_channel, 1024),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(SAGEConv(1024, 2048),args.HEADS_NUM),
                }
# torch.Size([16, 64, 64, 64]) resnet18,34
# torch.Size([16, 128, 32, 32])
# torch.Size([16, 256, 16, 16])
# torch.Size([16, 512, 8, 8])

            elif args.MODEL=="resnet34" or args.MODEL=='resnet18' or args.MODEL=="vgg16":
                self.gc1 = clones(SAGEConv(in_channel, 64),args.HEADS_NUM)
                self.gc2 = clones(SAGEConv(in_channel, 128),args.HEADS_NUM)
                self.gc3={
                    'gcn1':clones(SAGEConv(in_channel, 128),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(SAGEConv(128, 256),args.HEADS_NUM),
                }

                self.gc4={
                    'gcn1':clones(SAGEConv(in_channel, 256),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(SAGEConv(256, 512),args.HEADS_NUM),
                }
            
        elif graph_type=="GAT":
            if args.MODEL=="resnet50" or args.MODEL=="resnet101" or args.MODEL=="wideresnet50":
                self.gc1 = clones(GATConv(in_channel, 256),args.HEADS_NUM)
                self.gc2 = clones(GATConv(in_channel, 512),args.HEADS_NUM)
                self.gc3={
                    'gcn1':clones(GATConv(in_channel, 512),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GATConv(512, 1024),args.HEADS_NUM),
                }

                self.gc4={
                    'gcn1':clones(GATConv(in_channel, 1024),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GATConv(1024, 2048),args.HEADS_NUM),
                }
            elif args.MODEL=="resnet34" or args.MODEL=='resnet18' or args.MODEL=="vgg16":
                self.gc1 = clones(GATConv(in_channel, 64),args.HEADS_NUM)
                self.gc2 = clones(GATConv(in_channel, 128),args.HEADS_NUM)
                self.gc3={
                    'gcn1':clones(GATConv(in_channel, 128),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GATConv(128, 256),args.HEADS_NUM),
                }

                self.gc4={
                    'gcn1':clones(GATConv(in_channel, 256),args.HEADS_NUM),
                    'relu':clones(nn.ReLU(),args.HEADS_NUM),
                    'gcn2':clones(GATConv(256, 512),args.HEADS_NUM),
                }
   
        self.relu1 = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.relu3 = nn.LeakyReLU(0.2)
        self.relu4 = nn.LeakyReLU(0.2)
        if args.MODEL=='resnet101' or args.MODEL=='resnet50' or args.MODEL=="wideresnet50":
            self.fc=nn.Linear(2048, num_classes)
        if args.MODEL=='resnet34' or args.MODEL=='resnet18':
            self.fc=nn.Linear(512, num_classes)
        if args.MODEL=="vgg16":
            self.fc=nn.Linear(4096, num_classes)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
#         print(_adj)
#         print(self.A)

        self.index=[[],[]]
        self.weight=[]
        for idx_i,i in enumerate(_adj):
            for idx_j,j in enumerate(i):
                if(j!=0):
                    self.index[0].append(idx_i)
                    self.index[1].append(idx_j)
                    self.weight.append(j)
        self.index=torch.tensor(self.index, dtype=torch.long)
        self.weight=torch.tensor(self.weight, dtype=torch.float32)
        
        self.index=self.index.to(ctx)
        self.weight=self.weight.to(ctx)
        
        # self.softmax
        
        
        
        
#         print(self.index.type())
#         print(self.weight)
                
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        if args.MODEL=="vgg16":#vgg16只能插入最后一层进行作用
            for i in self.kun1:
                self.kun1[i]=self.kun1[i].to(ctx)
            for i in self.kun2:
                self.kun2[i]=self.kun2[i].to(ctx)
            for i in self.kun3:
                self.kun3[i]=self.kun3[i].to(ctx)
            for i in self.kun4:
                self.kun4[i]=self.kun4[i].to(ctx)
                
            for i in self.gc3:
                self.gc3[i]=self.gc3[i].to(ctx)
            for i in self.gc4:
                self.gc4[i]=self.gc4[i].to(ctx)
                
            feature=feature.to(ctx)
            
            feature = self.model.features(feature)#torch.Size([16, 512, 8, 8])
            if self.GRAPH=="GAT":
                inp=inp[0]
                inp=inp.to(ctx)
                # print("inp:"+str(inp.size()))
                inp=inp.repeat(args.HEADS_NUM,1,1)
                # print(self.index.size())#
                index=self.index.repeat(args.HEADS_NUM,1,1)
                # print("inp:"+str(inp.size()))
            else:
                inp=inp.to(ctx)
                # print("inp:"+str(inp.size()))
                inp=inp.repeat(args.HEADS_NUM,1,1,1)
                # print(self.index.size())#
                index=self.index.repeat(args.HEADS_NUM,1,1)
                # print("inp:"+str(inp.size()))
            # print(inp.device,index.device,model.device)
            k=[model_t(inp_t,index_t) for inp_t,index_t,model_t in zip(inp,index,self.gc4['gcn1'])]
            k=torch.stack(k,0)
            # print("kkkk:"+str(k.size()))

            k=[model_t(input_t) for input_t,model_t in zip(k,self.gc4['relu'])]
            k=torch.stack(k,0)

            k=[model_t(input_t,self.index) for input_t,model_t in zip(k,self.gc4['gcn2'])]
            k=torch.stack(k,0)#:torch.Size([2, 16, 17, 2048])
            if self.GRAPH=="GAT":
                dim_k=k.size()
                k=k.repeat([args.BATCH_SIZE,1,1])
                k=k.view([dim_k[0],args.BATCH_SIZE,dim_k[1],dim_k[2]])
            #kun结构

            feature1=self.kun4['conv11'](feature)#torch.Size([16, 2048, 1, 1])

            feature1=feature1.view(feature1.size()[0],feature1.size()[1],1)#torch.Size([16, 2048, 1])
            feature2=feature1.repeat(args.HEADS_NUM,1,1,1)##################################
            feature1=feature1.repeat(args.HEADS_NUM,1,1,1)

            q=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun4['Q'])]
            q=torch.stack(q,0)
            # print("qqqq:"+str(q.size()))

            v=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun4['V'])]
            v=torch.stack(v,0)

            feature1=[torch.bmm(q_t,k_t) for q_t,k_t in zip(q,k)]
            feature1=torch.stack(feature1,0)
            feature1=feature1*math.sqrt(args.AT_SIZE)
            feature1=F.softmax(feature1,dim=-1)
            # print("feature1:"+str(feature1.size()))
            #torch.Size([5, 16, 2048, 2048])

            feature1=[torch.bmm(f_t,v_t) for f_t,v_t in zip(feature1,v)]
            feature1=torch.stack(feature1,0)
            # print("hahaha:"+str(feature1.size()))
            #torch.Size([5, 16, 2048, 1])

            feature1=feature1.view(feature1.size()[0],feature1.size()[1],feature1.size()[2],1,1)##[4, 16, 2048, 1, 1]
            feature1=feature1.permute(1,0,2,3,4)
            s=feature1.size()
            feature1=feature1.reshape([s[0],s[1]*s[2]])
            
            # print(feature1.size(),"feature1")
            feature1=self.fc_last(feature1)#[16, 512]
            feature1=feature1.reshape([feature1.size()[0],feature1.size()[1],1,1])#[16, 512, 1, 1]
            feature1=feature1*feature
            feature=feature+feature1

            feature=self.model.avgpool(feature)

            feature = torch.flatten(feature, 1)
            feature=self.model.classifier(feature)
            
            # feature=F.softmax(feature,dim=1)

            return feature


        else:
            #     model.conv1,
            #     model.bn1,
            #     model.relu,
            #     model.maxpool,
            #     model.layer1,
            #     model.layer2,
            #     model.layer3,
            #     model.layer4,
            for i in self.kun1:
                self.kun1[i]=self.kun1[i].to(ctx)
            for i in self.kun2:
                self.kun2[i]=self.kun2[i].to(ctx)
            for i in self.kun3:
                self.kun3[i]=self.kun3[i].to(ctx)
            for i in self.kun4:
                self.kun4[i]=self.kun4[i].to(ctx)
                
            for i in self.gc3:
                self.gc3[i]=self.gc3[i].to(ctx)
            for i in self.gc4:
                self.gc4[i]=self.gc4[i].to(ctx)
                
            feature=feature.to(ctx)
            feature=self.model.conv1(feature)
            feature=self.model.bn1(feature)
            feature=self.model.relu(feature)
            feature=self.model.maxpool(feature)
            
            feature=self.model.layer1(feature)#256,56,56 resnet18:torch.Size([16, 64, 64, 64])
            # print("feature:"+str(feature.size()))
            # feature=self.model.layer2(feature)#256,56,56 resnet18:torch.Size([16, 64, 64, 64])
            # print("feature:"+str(feature.size()))
            # feature=self.model.layer3(feature)#256,56,56 resnet18:torch.Size([16, 64, 64, 64])
            # print("feature:"+str(feature.size()))
            # feature=self.model.layer4(feature)#256,56,56 resnet18:torch.Size([16, 64, 64, 64])
            # print("feature:"+str(feature.size()))
            
            
            if args.LAYER=="1":
                # adj = gen_adj(self.A).detach()
                # inp=inp[0]
                if self.GRAPH=="GAT":
                    inp=inp[0]
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
                else:
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
        
                k=[model_t(inp_t,index_t) for inp_t,index_t,model_t in zip(inp,index,self.gc1)]
                k=torch.stack(k,0)
                if self.GRAPH=="GAT":
                    dim_k=k.size()
                    k=k.repeat([args.BATCH_SIZE,1,1])
                    k=k.view([dim_k[0],args.BATCH_SIZE,dim_k[1],dim_k[2]])

                #kun结构
                
                feature1=self.kun4['conv11'](feature)#torch.Size([16, 2048, 1, 1])
                feature1=feature1.view(feature1.size()[0],feature1.size()[1],1)#torch.Size([16, 2048, 1])
                feature1=feature1.repeat(args.HEADS_NUM,1,1,1)
                
                q=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun1['Q'])]
                q=torch.stack(q,0)
                # print("qqqq:"+str(q.size()))
                
                v=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun1['V'])]
                v=torch.stack(v,0)
                
                
                feature1=[torch.bmm(q_t,k_t) for q_t,k_t in zip(q,k)]
                feature1=torch.stack(feature1,0)
                feature1=feature1*math.sqrt(args.AT_SIZE)
                feature1=F.softmax(feature1,dim=-1)
                # print("feature1:"+str(feature1.size()))   / math.sqrt(k_dim)
                #torch.Size([5, 16, 2048, 2048])
                
                feature1=[torch.bmm(f_t,v_t) for f_t,v_t in zip(feature1,v)]
                feature1=torch.stack(feature1,0)

                feature1=feature1.view(feature1.size()[0],feature1.size()[1],feature1.size()[2],1,1)#[4, 16, 256, 1, 1]
                feature1=feature1.permute(1,0,2,3,4)
                s=feature1.size()
                feature1=feature1.reshape([s[0],s[1]*s[2]])
                feature1=self.fc_last(feature1)#[16, 512]
                feature1=feature1.reshape([feature1.size()[0],feature1.size()[1],1,1])#[16, 512, 1, 1]
                feature1=feature1*feature
                feature=feature+feature1

#             #layer2
            feature=self.model.layer2(feature)#256,56,56
            if args.LAYER=="2":
                if self.GRAPH=="GAT":
                    inp=inp[0]
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
                else:
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
                k=[model_t(inp_t,index_t) for inp_t,index_t,model_t in zip(inp,index,self.gc2)]
                k=torch.stack(k,0)
                if self.GRAPH=="GAT":
                    dim_k=k.size()
                    k=k.repeat([args.BATCH_SIZE,1,1])
                    k=k.view([dim_k[0],args.BATCH_SIZE,dim_k[1],dim_k[2]])
                

                #kun结构
                
                feature1=self.kun4['conv11'](feature)#torch.Size([16, 2048, 1, 1])
                feature1=feature1.view(feature1.size()[0],feature1.size()[1],1)#torch.Size([16, 2048, 1])
                feature1=feature1.repeat(args.HEADS_NUM,1,1,1)
                
                q=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun2['Q'])]
                q=torch.stack(q,0)
                # print("qqqq:"+str(q.size()))
                
                v=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun2['V'])]
                v=torch.stack(v,0)

                
                # print("size:",q.size(),k.size(),v.size())
                #q:[4, 16, 512, 17],k:[4, 17, 512],v:[4, 16, 512, 1]
                
                feature1=[torch.bmm(q_t,k_t) for q_t,k_t in zip(q,k)]
                feature1=torch.stack(feature1,0)
                feature1=feature1*math.sqrt(args.AT_SIZE)
                feature1=F.softmax(feature1,dim=-1)
                # print("feature1:"+str(feature1.size()))
                #torch.Size([5, 16, 2048, 2048])
                
                feature1=[torch.bmm(f_t,v_t) for f_t,v_t in zip(feature1,v)]
                feature1=torch.stack(feature1,0)#[4, 16, 512, 1]
                
                feature1=feature1.view(feature1.size()[0],feature1.size()[1],feature1.size()[2],1,1)
                feature1=feature1.permute(1,0,2,3,4)
                s=feature1.size()
                feature1=feature1.reshape([s[0],s[1]*s[2]])
                feature1=self.fc_last(feature1)#[16, 512]
                feature1=feature1.reshape([feature1.size()[0],feature1.size()[1],1,1])#[16, 512, 1, 1]
                feature1=feature1*feature
                feature=feature+feature1


            #layer3
            feature=self.model.layer3(feature)#256,56,56
            if args.LAYER=="3":
                if self.GRAPH=="GAT":
                    inp=inp[0]
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
                else:
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
        
                k=[model_t(inp_t,index_t) for inp_t,index_t,model_t in zip(inp,index,self.gc3['gcn1'])]
                k=torch.stack(k,0)
                # print("kkkk:"+str(k.size()))
                
                k=[model_t(input_t) for input_t,model_t in zip(k,self.gc3['relu'])]
                k=torch.stack(k,0)
                
                k=[model_t(input_t,self.index) for input_t,model_t in zip(k,self.gc3['gcn2'])]
                k=torch.stack(k,0)
                # print("kkkk:"+str(k.size()))
                if self.GRAPH=="GAT":
                    dim_k=k.size()
                    k=k.repeat([args.BATCH_SIZE,1,1])
                    k=k.view([dim_k[0],args.BATCH_SIZE,dim_k[1],dim_k[2]])
                #kun结构
                
                feature1=self.kun4['conv11'](feature)#torch.Size([16, 2048, 1, 1])
                feature1=feature1.view(feature1.size()[0],feature1.size()[1],1)#torch.Size([16, 2048, 1])
                feature1=feature1.repeat(args.HEADS_NUM,1,1,1)
                
                q=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun3['Q'])]
                q=torch.stack(q,0)
                # print("qqqq:"+str(q.size()))
                
                v=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun3['V'])]
                v=torch.stack(v,0)
                
                feature1=[torch.bmm(q_t,k_t) for q_t,k_t in zip(q,k)]
                feature1=torch.stack(feature1,0)
                # print("feature1:"+str(feature1.size()))
                #torch.Size([5, 16, 2048, 2048])
                feature1=feature1*math.sqrt(args.AT_SIZE)
                feature1=F.softmax(feature1,dim=-1)
                
                feature1=[torch.bmm(f_t,v_t) for f_t,v_t in zip(feature1,v)]
                feature1=torch.stack(feature1,0)
                # print("hahaha:"+str(feature1.size()))
                #torch.Size([5, 16, 2048, 1])
                
                feature1=feature1.view(feature1.size()[0],feature1.size()[1],feature1.size()[2],1,1)#[4, 16, 1024, 1, 1]
                feature1=feature1.permute(1,0,2,3,4)
                s=feature1.size()
                feature1=feature1.reshape([s[0],s[1]*s[2]])
                feature1=self.fc_last(feature1)#[16, 512]
                feature1=feature1.reshape([feature1.size()[0],feature1.size()[1],1,1])#[16, 512, 1, 1]
                feature1=feature1*feature
                feature=feature+feature1


#             #layer4
            feature=self.model.layer4(feature)#256,56,56
#             # print("feature2:"+str(feature.size()))
            
#             #GNN处理
            if args.LAYER=="4":
                if self.GRAPH=="GAT":
                    inp=inp[0]
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
                else:
                    inp=inp.to(ctx)
                    # print("inp:"+str(inp.size()))
                    inp=inp.repeat(args.HEADS_NUM,1,1,1)
                    # print(self.index.size())#
                    index=self.index.repeat(args.HEADS_NUM,1,1)
                    # print("inp:"+str(inp.size()))
                k=[model_t(inp_t,index_t) for inp_t,index_t,model_t in zip(inp,index,self.gc4['gcn1'])]
                k=torch.stack(k,0)
                # print("kkkk:"+str(k.size()))
                
                k=[model_t(input_t) for input_t,model_t in zip(k,self.gc4['relu'])]
                k=torch.stack(k,0)
                
                k=[model_t(input_t,self.index) for input_t,model_t in zip(k,self.gc4['gcn2'])]
                k=torch.stack(k,0)#:torch.Size([2, 16, 17, 2048])
                if self.GRAPH=="GAT":
                    dim_k=k.size()
                    k=k.repeat([args.BATCH_SIZE,1,1])
                    k=k.view([dim_k[0],args.BATCH_SIZE,dim_k[1],dim_k[2]])
                #kun结构
                
                feature1=self.kun4['conv11'](feature)#torch.Size([16, 2048, 1, 1])

                feature1=feature1.view(feature1.size()[0],feature1.size()[1],1)#torch.Size([16, 2048, 1])
                feature2=feature1.repeat(args.HEADS_NUM,1,1,1)##################################
                feature1=feature1.repeat(args.HEADS_NUM,1,1,1)
                
                q=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun4['Q'])]
                q=torch.stack(q,0)
                # print("qqqq:"+str(q.size()))
                
                v=[model_t(f_t) for f_t,model_t in zip(feature1,self.kun4['V'])]
                v=torch.stack(v,0)
                
                feature1=[torch.bmm(q_t,k_t) for q_t,k_t in zip(q,k)]
                feature1=torch.stack(feature1,0)
                feature1=feature1*math.sqrt(args.AT_SIZE)
                feature1=F.softmax(feature1,dim=-1)
                # print("feature1:"+str(feature1.size()))
                #torch.Size([5, 16, 2048, 2048])

                feature1=[torch.bmm(f_t,v_t) for f_t,v_t in zip(feature1,v)]
                feature1=torch.stack(feature1,0)
                # print("hahaha:"+str(feature1.size()))
                #torch.Size([5, 16, 2048, 1])
                
                feature1=feature1.view(feature1.size()[0],feature1.size()[1],feature1.size()[2],1,1)##[4, 16, 2048, 1, 1]
                feature1=feature1.permute(1,0,2,3,4)
                s=feature1.size()
                feature1=feature1.reshape([s[0],s[1]*s[2]])
                feature1=self.fc_last(feature1)#[16, 512]
                feature1=feature1.reshape([feature1.size()[0],feature1.size()[1],1,1])#[16, 512, 1, 1]
                feature1=feature1*feature
                feature=feature+feature1
                
            feature=self.model.avgpool(feature)
            # print("pooling:"+str(feature.size()))
            feature=feature.view(feature.size()[0],-1)
            feature=self.fc(feature)
            # print("fc:"+str(feature.size()))
            
            
            return feature
            
    

    def get_config_optim(self, lr, lrp):
        return [
                {'params': [p for n, p in self.model.named_parameters() if p.requires_grad], 'lr': lr * lrp},

                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                {'params': self.gc3['gcn1'].parameters(), 'lr': lr},
                {'params': self.gc3['relu'].parameters(), 'lr': lr},
                {'params': self.gc3['gcn2'].parameters(), 'lr': lr},
                {'params': self.gc4['gcn1'].parameters(), 'lr': lr},
                {'params': self.gc4['relu'].parameters(), 'lr': lr},
                {'params': self.gc4['gcn2'].parameters(), 'lr': lr},              
                {'params': self.kun1["conv11"].parameters(), 'lr': lr* lrp},
                {'params': self.kun1["Q"].parameters(), 'lr': lr* lrp},
                {'params': self.kun1["V"].parameters(), 'lr': lr* lrp},
                {'params': self.kun2["conv11"].parameters(), 'lr': lr* lrp},
                {'params': self.kun2["Q"].parameters(), 'lr': lr* lrp},
                {'params': self.kun2["V"].parameters(), 'lr': lr* lrp},
                {'params': self.kun3["conv11"].parameters(), 'lr': lr* lrp},
                {'params': self.kun3["Q"].parameters(), 'lr': lr* lrp},
                {'params': self.kun3["V"].parameters(), 'lr': lr* lrp},
                {'params': self.kun4["conv11"].parameters(), 'lr': lr* lrp},
                {'params': self.kun4["Q"].parameters(), 'lr': lr* lrp},
                {'params': self.kun4["V"].parameters(), 'lr': lr* lrp},
                {'params': self.kun4["Q2"].parameters(), 'lr': lr* lrp},
                {'params': self.kun4["V2"].parameters(), 'lr': lr* lrp},
                {'params': self.kun4["K2"].parameters(), 'lr': lr* lrp},
                {'params': self.v.parameters(), 'lr': lr* lrp},
                {'params': self.k.parameters(), 'lr': lr* lrp},
                {'params': self.relu1.parameters(), 'lr': lr* lrp},
                {'params': self.relu2.parameters(), 'lr': lr* lrp},
                {'params': self.relu3.parameters(), 'lr': lr* lrp},
                {'params': self.relu4.parameters(), 'lr': lr* lrp},
        
        ]
            
            
            
            
            
            
def gcn_resnet101(num_classes=None, t=None, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=True)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,graph_type=args.GRAPH,res=args.RES).to(ctx)

def gcn_resnet50(num_classes=None, t=None, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet50(pretrained=True)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,graph_type=args.GRAPH,res=args.RES).to(ctx)

def gcn_resnet18(num_classes=None, t=None, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet18(pretrained=True)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,graph_type=args.GRAPH,res=args.RES).to(ctx)
          
def gcn_resnet34(num_classes=None, t=None, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet34(pretrained=True)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,graph_type=args.GRAPH,res=args.RES).to(ctx)

def gcn_vgg16(num_classes=None, t=None, pretrained=False, adj_file=None, in_channel=300):
    model = models.vgg16(pretrained=True)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,graph_type=args.GRAPH,res=args.RES).to(ctx)

def gcn_wideresnet50(num_classes=None, t=None, pretrained=False, adj_file=None, in_channel=300):
    model = models.wide_resnet50_2(pretrained=True)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,graph_type=args.GRAPH,res=args.RES).to(ctx)

def get_model():
    adj_file="/Public/YongkunLiu/2022/2021/GCN-ML-new/data/"
    num_classes=-1
    
    if args.DATASET_NAME=="UCMerced":
        adj_file=adj_file+"ucm_adj.pkl"
        num_classes=17
        args.AT_SIZE=17
    elif args.DATASET_NAME=="AID_multilabel":
        adj_file=adj_file+"aid_adj.pkl"
        num_classes=17
        args.AT_SIZE=17
    elif args.DATASET_NAME=="DFC15_multilabel":
        adj_file=adj_file+"dfc15_adj.pkl"
        num_classes=8
        args.AT_SIZE=8
    
    if args.MODEL=="resnet101":
        model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file=adj_file)
    if args.MODEL=="resnet50":
        model = gcn_resnet50(num_classes=num_classes, t=0.4, adj_file=adj_file)
    if args.MODEL=="resnet34":
        model = gcn_resnet34(num_classes=num_classes, t=0.4, adj_file=adj_file)
    if args.MODEL=="resnet18":
        model = gcn_resnet18(num_classes=num_classes, t=0.4, adj_file=adj_file)
    if args.MODEL=="vgg16":
        model = gcn_vgg16(num_classes=num_classes, t=0.4, adj_file=adj_file)
    if args.MODEL=="wideresnet50":
        model = gcn_wideresnet50(num_classes=num_classes, t=0.4, adj_file=adj_file)
    return model
        
    
      
            
def compute_f1(labels,outputs):
    cpu=torch.device( "cpu")
    labels = labels.to(cpu).detach().numpy()
    outputs = outputs.to(cpu).detach().numpy()
    
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            if outputs[i][j]>Q:
                outputs[i][j]=1
            else:
                outputs[i][j]=0
    
    F1=[]
    for i in range(labels.shape[0]):
        F1.append(f1_score(labels[i],outputs[i]))
    return np.mean(F1)

def compute_f1_f2_p_r(labels,outputs):
    cpu=torch.device( "cpu")
    labels = labels.to(cpu).detach().numpy()
    outputs = outputs.to(cpu).detach().numpy()
    
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            if outputs[i][j]>Q:
                outputs[i][j]=1
            else:
                outputs[i][j]=0
    
    F1=[]
    F2=[]
    P=[]
    R=[]

    for i in range(labels.shape[0]):
        R.append(recall_score(labels[i],outputs[i],zero_division=0))
    for i in range(labels.shape[0]):
        P.append(precision_score(labels[i],outputs[i],zero_division=0))
    # print(P,R)
    for i in range(len(R)):
        F1.append(2*(P[i]*R[i])/(P[i]+R[i]+0.00000001))
    for i in range(len(R)):
        F2.append(5*(P[i]*R[i])/(4*P[i]+R[i]+0.00000001))
    # print("-------")
    # print(F1)
    return np.mean(F1),np.mean(F2),np.mean(P),np.mean(R)
    
            
def compute_mAP(labels,outputs):
    cpu=torch.device( "cpu")
    y_true = labels.to(cpu).detach().numpy()
    y_pred = outputs.to(cpu).detach().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)
            
def evaluate_loss(data_iter, net, ctx,epoch):
    net.eval()
    loss_sum, n,val_map_sum,b_n= 0.0, 0,0.0,0
    with torch.no_grad():
        for bidx,(input,target,imgpath) in enumerate(data_iter):
            input_img=input[0]
            input_glove=input[1]
            input_img=input_img.to(ctx)
            input_glove=input_glove.to(ctx)
            target=target.to(ctx)
            
            output=net(input_img,input_glove)
            #loss = criterion(y_hat, y)
            n+=target.size()[0]
            criterion = nn.MultiLabelSoftMarginLoss()
            loss = criterion(output, target).item()
#             loss=criterion(y_hat, y).item()
            loss_sum +=loss
            # mAP=compute_f1(target,output)
            F1,F2,P_score,R_score=compute_f1_f2_p_r(target,output)
            val_map_sum+=F1
            b_n+=1
    
            t={}
            # t["epoch"],t["trainval"],t["batch"],t["loss"],t["F1"]=epoch,"val",bidx,loss,mAP
            # print_json.append(t)
            logger.info('{{\"epoch\":{},\"trainval\":\"{}\",\"batch\":{},\"loss\":{:.5f},\"F1\":{}}},'.format\
                    (epoch,'val',bidx,loss,F1))
        t={}
        # t["epoch"],t["trainval"],t["batch"],t["loss"],t["F1"]=epoch,'val',-1,loss_sum/b_n,val_map_sum/b_n
        # print_json.append(t)
    logger.info('{{\"epoch\":{},\"trainval\":\"{}\",\"batch\":{},\"loss\":{:.5f},\"F1\":{}}},'.format\
                (epoch,'val',-1,loss_sum/b_n,val_map_sum/b_n))
        
    return loss_sum/b_n,val_map_sum/b_n 
     
#new train
    # def get_config_optim(self, lr, lrp):
    #     return [
    #             {'params': self.features.parameters(), 'lr': lr * lrp},
    #             {'params': self.gc1.parameters(), 'lr': lr},
    #             {'params': self.gc2.parameters(), 'lr': lr},]


def test(net,test_iter):
    train_loss_sum, n, start ,train_map_sum,train_pred_sum,correct,b_n= 0.0, 0, time.time(),0,0,0,0
    F1_sum,F2_sum,P_sum,R_sum=0,0,0,0
    net.eval()
    
    save_path=args.DATASET_NAME+'log/'
    if os.path.exists(save_path)==False:
        os.mkdir(save_path)
    output_dcit={}
    with torch.no_grad():
        for bidx,(input,target,imgpath) in enumerate(test_iter):
            input_img=input[0]
            input_glove=input[1]
            input_img=input_img.to(ctx)
            input_glove=input_glove.to(ctx)
            target=target.to(ctx)
            print("test",input_img.size(),input_glove.size())
            output=net(input_img,input_glove)



            n+=target.size()[0]
            b_n+=1

            F1,F2,P_score,R_score=compute_f1_f2_p_r(target,output)
            F1_sum+=F1
            F2_sum+=F2
            P_sum+=P_score
            R_sum+=R_score
            
            output=output.cpu().detach().numpy()[0]
            img_name=imgpath[0].split('/')[-1].split('.')[0]
            output_dcit[img_name]=output
            
    with open(save_path+localtime+".pkl",'wb') as f:
        pickle.dump(output_dcit, f)
    print("test_F1:")
    print(F1_sum/b_n)
    logger.info("{{\"test_F1\":{},\"test_F2\":{},\"test_P\":{},\"test_R\":{}}}".format(F1_sum/b_n,F2_sum/b_n,P_sum/b_n,R_sum/b_n))
    return F1_sum/b_n,F2_sum/b_n,P_sum/b_n,R_sum/b_n,

def train(net, train_iter, valid_iter, num_epochs, lr,lr_p,  ctx, lr_period, lr_decay,file_name):
    Max_map=0.0
    optimizer = torch.optim.Adam(model.get_config_optim(lr,lr_p))
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_period, gamma=lr_decay)
    
    if not os.path.exists(args.WORK_DIR+'/pth'):
        os.makedirs(args.WORK_DIR+'/pth')
        
    for epoch in range(num_epochs):
        train_loss_sum, n, start ,train_map_sum,train_pred_sum,correct,b_n= 0.0, 0, time.time(),0,0,0,0
        
        lr = scheduler.get_last_lr()
        # if epoch!=0 and epoch%args.LR_PERIOD==0:
        #     lr=lr*args.LR_DECAY
        #     optimizer = torch.optim.Adam(model.get_config_optim(lr,lr_p))
        # print(optimizer)
        net.train()
        for bidx,(input,target,imgpath) in enumerate(train_iter):
            input_img=input[0]
            input_glove=input[1]
            input_img=input_img.to(ctx)
            input_glove=input_glove.to(ctx)
            target=target.to(ctx)
            
            optimizer.zero_grad()
            
            # print("train",input_img.size(),input_glove.size())
            output=net(input_img,input_glove)
            criterion = nn.MultiLabelSoftMarginLoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            n+=target.size()[0]
            b_n+=1
            train_loss_sum += loss.item()
            
            # mAP=compute_f1(target,output)
            F1,F2,P_score,R_score=compute_f1_f2_p_r(target,output)
            train_map_sum+=F1
            t={}
            # t["epoch"],t["trainval"],t["batch"],t["loss"],t["F1"]=epoch,"train",bidx,float(loss),mAP
            # print_json.append(t)
            logger.info("{{\"epoch\":{},\"trainval\":\"{}\",\"batch\":{},\"loss\":{:.5f},\"F1\":{}}},".format\
                    (epoch,"train",bidx,float(loss),F1))
        t={}
        # t["epoch"],t["trainval"],t["batch"],t["loss"],t["F1"]=epoch,"train",-1,float(train_loss_sum/b_n),train_map_sum/b_n
        # print_json.append(t)
        
        writer.add_scalar('Loss-train', train_loss_sum/b_n, epoch)
        writer.add_scalar('F1-train', train_map_sum/b_n, epoch)
        
        logger.info("{{\"epoch\":{},\"trainval\":\"{}\",\"batch\":{},\"loss\":{:.5f},\"F1\":{}}},".format\
                    (epoch,"train",-1,train_loss_sum/b_n,train_map_sum/b_n))
        valid_loss,valid_map = evaluate_loss(valid_iter, net, ctx,epoch)
        
        writer.add_scalar('Loss-val', valid_loss, epoch)
        writer.add_scalar('F1-val', valid_map, epoch)
        
        if(valid_map>Max_map):
            Max_map=valid_map
            if args.SAVE=="True":
                torch.save(net,'{}/{}.pth'.format(args.WORK_DIR+'/pth',file_name))
            logger.info("{{\"epoch\":{},\"save\":\"save\"}},".format(epoch))
        scheduler.step()
        
            
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WILDCAT Training')

    parser.add_argument("--DATASET_NAME", default="UCMerced",help="AID_multilabel,UCMerced,DFC15_multilabel", type=str)
    parser.add_argument("--NUM_EPOCHS",default=80, help="display a square of a given number", type=int)
    parser.add_argument("--BATCH_SIZE",default=16, help="display a square of a given number", type=int)
    parser.add_argument("--IMG_SIZE",default=256, help="display a square of a given number", type=int)
    parser.add_argument("--LR",default=0.001, help="display a square of a given number", type=float)
    parser.add_argument("--LRP",default=0.1, help="display a square of a given number", type=float)
    parser.add_argument("--LR_PERIOD",default=25, help="display a square of a given number", type=int)
    parser.add_argument("--LR_DECAY",default=0.1, help="display a square of a given number", type=float)
    parser.add_argument("--WORK_DIR",default='/Public/YongkunLiu/workdir/', help="display a square of a given number", type=str)
    parser.add_argument("--GRAPH",default='SAGE', help="GAT or GCN or SAGE", type=str)
    parser.add_argument("--MODEL",default='vgg16', help="resnet101,resnet50,resnet34,resnet18,vgg16,wideresnet50", type=str)
    parser.add_argument("--RES",default="True", help="True or False；", type=str)
    parser.add_argument("--SELF_ATTENTION",default="False", help="True or False or Self;True表示label-atten+self-atten；False表示不加入该分支", type=str)
    parser.add_argument("--AT_SIZE",default=17, help="display a square of a given number", type=int)
    parser.add_argument("--DES",default="添加描述信息", help="添加描述信息", type=str)
    parser.add_argument("--LAYER",default="2", help="123", type=str)
    parser.add_argument("--SAVE",default="False", help="True or False", type=str)
    parser.add_argument("--HEADS_NUM",default=6, help="2个头", type=int)
    
    

    #lr_period,lr_decay
    args = parser.parse_args()
    file_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(args.LR)

    dir_pth='/Public/YongkunLiu/DataSet/'
    UCM_ML_npy = os.path.join(dir_pth, 'ucm_ml.npy')
    AID_ML_npy = os.path.join(dir_pth, 'aid_ml.npy')
    DFC_ML_npy = os.path.join(dir_pth, 'dfc15_ml.npy')
    Q=0.5
    
    if args.MODEL=="vgg16":
        args.IMG_SIZE=224
    # print(args.IMG_SIZE,"IMG_SIZE")

        
    print_json={}
    imgExt='jpg'
    if args.DATASET_NAME=="UCMerced":
        imgExt='tif'
    if args.DATASET_NAME=="DFC15_multilabel":
        imgExt='png'
        
    inp_name="/Public/YongkunLiu/2022/2021/GCN-ML-new/data/aid_glove.pkl"
    if args.DATASET_NAME=="DFC15_multilabel":
        inp_name="/Public/YongkunLiu/2022/2021/GCN-ML-new/data/dfc15_glove.pkl"
    
    ctx=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger=get_logger(file_name)
    localtime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
    writer = SummaryWriter(args.WORK_DIR+'tensorboard/'+'/'+args.DATASET_NAME+"/"+'/'+args.MODEL+"/"+localtime)
    print_json["time"],print_json["BATCH_SIZE"],print_json["NUM_EPOCHS"],print_json["LR"],print_json["LRP"],print_json["LR_PERIOD"],print_json["LR_DECAY"],print_json["DATASET_NAME"],print_json["GRAPH_TYPE"],print_json["MODEL"],print_json["RES"],print_json["SELF_ATTENTION"],print_json["AT_SIZE"],print_json["DES"],print_json["LAYER"],print_json["HEADS_NUM"]=localtime,args.BATCH_SIZE,args.NUM_EPOCHS,args.LR,args.LRP,args.LR_PERIOD,args.LR_DECAY,args.DATASET_NAME,args.GRAPH,args.MODEL,args.RES,args.SELF_ATTENTION,args.AT_SIZE,args.DES,args.LAYER,args.HEADS_NUM
    # print_json.append(t)
    
    logger.info("{{\"time\":\"{}\",\"BATCH_SIZE\":{},\"NUM_EPOCHS\":{},\"LR\":{},\"LRP\":{},\"LR_PERIOD\":{},\"LR_DECAY\":{},\"DATASET_NAME\":\"{}\",\"GRAPH_TYPE\":\"{}\",\"MODEL\":\"{}\",\"RES\":\"{}\",\"SELF_ATTENTION\":\"{}\",\"DES\":\"{}\",\"AT_SIZE\":\"{}\",\"LAYER\":\"{}\",\"HEADS_NUM\":\"{}\"}}".format\
            (localtime,args.BATCH_SIZE,args.NUM_EPOCHS,args.LR,args.LRP,args.LR_PERIOD,args.LR_DECAY,args.DATASET_NAME,args.GRAPH,args.MODEL,args.RES,args.SELF_ATTENTION,args.DES,args.AT_SIZE,args.LAYER,args.HEADS_NUM))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
                    transforms.Resize([args.IMG_SIZE,args.IMG_SIZE]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=(0.6,1.4), contrast=(0.6,1.4), saturation=(0.6,1.4), hue=0),
                    transforms.ToTensor(),
                    normalize])
    val_transform = transforms.Compose([
                    transforms.Resize([args.IMG_SIZE,args.IMG_SIZE]),
                    transforms.ToTensor(),
                    normalize])
    
    train_dataGen = DataGeneratorML(data='/Public/YongkunLiu/DataSet', 
                                                dataset=args.DATASET_NAME, 
                                                imgExt=imgExt,
                                                imgTransform=train_transform,
                                                phase='train',
                                                inp_name=inp_name)
    val_dataGen = DataGeneratorML(data='/Public/YongkunLiu/DataSet', 
                                                dataset=args.DATASET_NAME, 
                                                imgExt=imgExt,
                                                imgTransform=val_transform,
                                                phase='val',
                                              inp_name=inp_name)
    test_dataGen = DataGeneratorML(data='/Public/YongkunLiu/DataSet', 
                                                dataset=args.DATASET_NAME, 
                                                imgExt=imgExt,
                                                imgTransform=val_transform,
                                                phase='test',
                                              inp_name=inp_name)
    model=get_model()
    model=model.to(ctx)
    train_loader = torch.utils.data.DataLoader(train_dataGen,batch_size=args.BATCH_SIZE, shuffle=True,drop_last=True,num_workers=5)
    val_loader = torch.utils.data.DataLoader(val_dataGen,batch_size=args.BATCH_SIZE, shuffle=True,drop_last=True,num_workers=5)
    if args.GRAPH=="GAT":
        test_loader = torch.utils.data.DataLoader(test_dataGen,batch_size=args.BATCH_SIZE, shuffle=True,num_workers=5,drop_last=True)
    else:
        test_loader = torch.utils.data.DataLoader(test_dataGen,batch_size=1, shuffle=True,num_workers=5)
    # input=torch.ones()
    train(model,train_loader,val_loader,args.NUM_EPOCHS,args.LR,args.LRP,ctx,args.LR_PERIOD,args.LR_DECAY,file_name)
    print_json["F1"],print_json["F2"],print_json["P"],print_json["R"]=test(model,test_loader)
    
    
    if not os.path.exists(args.WORK_DIR+"new_mh2.json"):
        j=[]
        j_string=json.dumps(j)
        with open(args.WORK_DIR+"new_mh2.json",'w+') as f:
            f.write(j_string)
    with open(args.WORK_DIR+"new_mh2.json",'r+') as f:
        all=json.load(f)
    all.append(print_json)
    json.dump(all, open(args.WORK_DIR+"new_mh2.json", "w"))
    
    #args.WORK_DIR +'log/'+ file_name+'.log'
    # os.rename(args.WORK_DIR +'log/'+ file_name+'.log',args.WORK_DIR +'log/'+ file_name+"_"+str('{:.6f}'.format(print_json["F1"]))+"_"+args.DES+'.log')
    
    # json.dump(print_json, open(args.WORK_DIR +'json/'+file_name+".json", "w"))
#
