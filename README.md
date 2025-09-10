<h2><b>Overview</b></h2>

This repo contains code for NeuroNur's first exploratory project: instance segmentation of neuronal cells. Neurological diseases such as Alzheimer's and Lewy Body Dementia are still poorly understood on a cellular level. Instance segmentation is a powerful tool in object detection and tracking of individual cells. Light microscopy is a non-invasive and easy way to study cells and instance segmentation is an effective way to collect quantifiable information that can help with disease detection and treatment. Over the years, there have been different kinds of deep learning models created to improve the quality of these segmentations, but each have their advantages and disadvantages. 

The goal is to explore and compare how well different models train on brightfield images of neurons, astrocytes and the neuroblastoma cell line, SH-SY5Y. Three different deep learning neural networks will be used and the accuracy of the resulting predictions will be compared using the Dice Score, Average Precision and Recall. The data used for training is the Satorius Dataset and contains 484 images. The ground truth labels are run-length encoding (RLE) annotations. Models of interest: Mask R-CNN, U-Net CNN and Mask2Transformer. 

<h2><b>About NeuroNur</b></h2>

NeuroNur is a newly-launched, neuroscienced based research initiative whose goal is to make research more available for the global community through collaboration, open science, and inclusive research. Our initial focus is early detection of Alzheimer's and Lewy Body Detention visa blood and CSF biomarkers such as alpha-synuclein, amyloid-beta and tau. Another point of interest is exploring how cell morphology differs begins to change at the onset of these diseases. 

<h2><b>Mask R-CNN</b></h2>

The first model explored is the Mask Regional Connvolutional Neural Network (R-CNN). This is a popular option for instance segmentation because it is able to predict multiple overlaying objects on the pixel level. The RLE annotations were used as ground truth labels to train the model. The model being used was the Resnet50 Mask R-CNN from PyTorch. 

Example 1: astrocytes
![alt text](image-1.png)
Raw image (left) Ground truth (center) Mask R-CNN Prediction (right)

Example 2: neuroglioblastoma
![alt text](image-2.png)
Raw image (left) Ground truth (center) Mask R-CNN Prediction (right)

Example 3: neurons
![alt text](image-3.png)
Raw image (left) Ground truth (center) Mask R-CNN Prediction (right)

<h2>Mask2Transformer (Ongoing) </h2>

Training of this transformer-based model is currently on going. Mask2Transformer is a powerful deep learning algorithm that, like Mask RCNN, is able to detect overlapping objects on the pixel level. This transformer is backed with a pre-trained Swin Transformer model. This model is trained for semantic, instance and panoptic segmentation.Transformers have become very popular for segmentation because of their use of self-attention. As of now, COCO style ground truths have been generated from the ground truths.

Example 1
![alt text](image-5.png)
Raw image (left) generated ground truth label (right)

Example 2
![alt text](image-6.png)
Raw image (left) generated ground truth label (right)

<h2>U-Net CNN</h2>
Coming soon.