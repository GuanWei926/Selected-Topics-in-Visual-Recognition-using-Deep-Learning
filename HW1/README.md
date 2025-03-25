# NYCU Selected Topics in Visual Recognition using Deep Learning HW1
Student ID: 111550061   
Name: 邱冠崴

## Introduction
This is a task to classify images into one of 100 categories. As a requirement, we must use ResNet or its variants as the backbone model. Finally, the prediction results will be submitted to Codabench to evaluate the performance.
In my work, I use ResNeXt-101 (64×4d) as the backbone and experiment with an ensemble approach, specifically bagging, to improve accuracy.

## How to install
1. Start by cloning the repository to your local machine. Use the following Git command:
```bash 
git clone https://github.com/GuanWei926/Selected-Topics-in-Visual-Recognition-using-Deep-Learning.git   
cd Selected-Topics-in-Visual-Recognition-using-Deep-Learning/HW1    
```

2. download the dataset
```bash 
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u" -O hw1-data.tar.gz   
```

## Performance snapshot
![alt text](image.png)
![alt text](image-1.png)