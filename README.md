# Semantic Interleaving Global Channel Attention for Multilabel Remote Sensing Image Classification


# Abstract
MLRSIC(Multi-Label Remote Sensing Image Classification), as one of the fundamental tasks in remote sensing area, has received increasing research interest. Taking the co-occurrence relationship of multiple labels as additional information helps to improve the performance of this task, but this method has not been fully utilized. Current methods focus on using it to constrain the final feature output of a Convolutional Neural Network (CNN). On the one hand, these methods do not make full use of label correlation to form feature representation. On the other hand, they increase the label noise sensitivity of the system, resulting in poor robustness. In this paper, a novel method called "Semantic Interwoven Global Channel Attention" (SIGNA) is proposed for MLRSIC. SIGNA can make full use of the co-occurrence relationship of labels to form better feature representation within CNN, thereby reducing the uncertainty and risk caused by explicit use of semantic features for classification and enhancing the generalization ability of the system. First, the label co-occurrence graph is obtained according to the statistical information of the data set. The label co-occurrence graph is used as the input of the Graph Neural Network (GNN) to generate optimal feature representations. Then, the semantic features and visual features are interleaved, to guide the feature expression of the image from the original feature space to the semantic feature space with embedded label relations. SIGNA triggers global attention of feature maps channels in a new semantic feature space to extract more important visual features. Multi-head SIGNA based feature adaptive weighting networks are proposed to act on any layer of CNN in a plug-and-play manner. For remote sensing images, better classification performance can be achieved by inserting CNN into the shallow layer. We conduct extensive experimental comparisons on three data sets: UCM data set, AID data set, and DFC15 data set. Experimental results demonstrate that the proposed SIGNA achieves superior classification performance compared to state-of-the-art (SOTA) methods.    

# Results
![image](https://user-images.githubusercontent.com/44631434/182540022-051e6a3d-13e4-44f1-bdf7-afa3f246b9f0.png)




