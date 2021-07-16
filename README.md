# Multi-label image classification using transfer learning 
--------------------------------
#### 
--------------------------------
## Table of contents
1. [Introduction](#introduction)
2. [Dataset creation](#Dataset_creation)
3. [Training the model](#Training_the_model)
4. [Results](#Results)
5. [The team](#The_team)

--------------------------------
## 1. Introduction <a name="introduction"></a>
The goal of our project was to build a model that can classify 5 types of images:
* Indoor Selfie
* Outdoor Selfie
* Indoor Pose
* Outdoor Pose
* Photos Without Human

The project was divided in 3 phases:
* Dataset collection and preparation
* Training the model
* Model evaluation and deployment

--------------------------------
## 2. Dataset creation <a name="Dataset_creation"></a>

In order to build a robust classifier, we prepared a custom dataset with balanced distribution of the five classes, each containing 500 images:
    * 0_selfie_indoor  
    * 1_selfie_outdoor
    * 2_pose_indoor  
    * 3_pose_outdoor 
    * 4_without_human  

### Data Collection <a name="Data_collection"></a>
Selfie images were collected from a pre-made dataset available at: https://www.crcv.ucf.edu/data/Selfie/, and then divided to indoor/outdoor selfies. The dataset includes both regular and mirror selfies. Other images were collected from various web sources.

### Preview of dataset <a name="Preview_of_dataset"></a>

| **Class**|**1**|**2** |**3**|**4**|**5**|  
|:---:|:---:|:---:|:---:|:---:|:---:|
|**Indoor Selfie**|![1](https://user-images.githubusercontent.com/86669701/125687459-3af5d805-d1e3-4c41-8ca8-9f03740b6dc2.jpg)|![2](https://user-images.githubusercontent.com/86669701/125687462-646ef80e-4dfa-4e96-861b-79fe159ac166.jpg)|![3](https://user-images.githubusercontent.com/86669701/125687463-e69fc1cc-9458-4e32-ab1d-7b61c79a3205.jpg)|![4](https://user-images.githubusercontent.com/86669701/125687465-fc33ec8f-3ebc-4448-9ca1-5221b2483aca.jpg)|![5](https://user-images.githubusercontent.com/86669701/125687467-eef6a818-d2a8-425d-9bf8-4558985322b9.jpg)|
|**Outdoor Selfie**|![10013066_501314863314102_1500970753_a](https://user-images.githubusercontent.com/86669701/125687915-e165e01e-3bbe-4c5c-897c-2617959c3115.jpg)|![10175136_1450536605181202_1947453693_a](https://user-images.githubusercontent.com/86669701/125687919-e176e8ec-da7f-473e-aed3-68fb99ba8bc0.jpg)|![OIP](https://user-images.githubusercontent.com/86669701/125687921-939ae5ad-ab55-4658-b408-eed5d78169eb.jpg)|![so00010](https://user-images.githubusercontent.com/86669701/125687922-c058628c-2f48-4515-a9c9-0628ae4912b9.jpg)|![so00024](https://user-images.githubusercontent.com/86669701/125687924-3b43bc33-f82f-41e7-a209-6bdd103dc945.jpg)|
|**Indoor Pose**    |![pi00078](https://user-images.githubusercontent.com/86669701/125688412-8b8e2e37-dbf8-4245-bb6f-123c6c3c6137.jpg)|![pexels-ali-pazani-2787341](https://user-images.githubusercontent.com/86669701/125688414-4894b4f6-d42e-4752-8cfc-77a1a72e2a75.jpg)|![pi00013](https://user-images.githubusercontent.com/86669701/125688415-0283f669-94eb-4872-8225-0543b909e97d.jpg)|![pi00016](https://user-images.githubusercontent.com/86669701/125688418-0573f09e-af93-40ed-84c1-2fb467f8c973.jpg)|![pi00050](https://user-images.githubusercontent.com/86669701/125688420-48fda318-1cbd-4c9d-98d0-a7e30c5a9732.jpg)|
|**Outdoor Pose**  |![62532-340x509-outdoor4](https://user-images.githubusercontent.com/86669701/125688710-bc1298ed-01e0-487a-b386-23f65109b08d.jpg)|![10013020_624911064230364_1577490982_a](https://user-images.githubusercontent.com/86669701/125688711-5ac6999e-06b2-46f2-bc88-83ca5792714d.jpg)|![4032](https://user-images.githubusercontent.com/86669701/125688712-518cb511-64f1-4c0d-80df-fa30e1dcd8d7.jpg)|![5824bfab157992b7b554f1de9dae131d](https://user-images.githubusercontent.com/86669701/125688714-a7b10ad7-ee03-4532-b48c-08ba856a44ec.jpg)|![54127e2476780e4a045ddaae65062928--men-fashion-photography-editorial-photography](https://user-images.githubusercontent.com/86669701/125688715-addd608f-2495-49b8-acec-4263e977ede2.jpg)|
|**Without Human** |![a-boat-sails-on-the-water](https://user-images.githubusercontent.com/86669701/125689199-1f068465-804c-4cb7-a4b4-290c2cce1e54.jpg)|![Image_118](https://user-images.githubusercontent.com/86669701/125689201-992354eb-376a-4418-84f2-f8163f2daf89.jpg)|![Image_182](https://user-images.githubusercontent.com/86669701/125689203-151fe723-42b3-4aaf-979e-840dbf22da94.jpg)|![lighthouse-blue-sky-and-beach-pools](https://user-images.githubusercontent.com/86669701/125689204-2d375e5a-3ded-4f2b-a656-a18ea95e6fcd.jpg)|![treeline-below-mountain-and-blue-sky](https://user-images.githubusercontent.com/86669701/125689205-0aeec6fe-d8a0-4d54-b06e-c69927cda525.jpg)|
## 3. Training the model <a name="Training_the_model"></a>
Our dataset has been trained on different ML models

1.Convolutional Neural Network

2.Transfer Learning models-its trained on different pre-trained models
### 3.i. <a name="xyz"></a>
Xception

InceptionV3

ResNet50
### 3.ii. <a name="xyz"></a>
VGG16

MobileNet

--------------------------------
## 4. Results <a name="Results"></a>
CNN-we got an accuracy of around 50%,so done augmentation and tried again but not much major improvement
### 4.i. <a name="xyz"></a>
Xception-got an accuracy of 60 and specific improvement after augmenting

InceptionV3- got a decent accuracy around 83

ResNet50- not good selection for this dataset,very low
### 4.ii. <a name="xyz"></a>
VGG16- a decent accuracy but not the best so far

MobileNet-this was the better pretrained model used with a good accuracy of 94

--------------------------------
## 5. The team <a name="The_team"></a>
* [Sabina Dzafic](https://github.com/sabinadzafic)
* [Sushma Timmaraju](https://github.com/sushmavenu)
* [Daniel Varga](https://github.com/IndaPerpetuum)



This project was supervised by [Igor Trpevski]()  *NOTE:*Need to get Github


