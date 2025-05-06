# NYCU Selected Topics in Visual Recognition using Deep Learning HW3
Student ID: 111550061   
Name: 邱冠崴

## Introduction
This assignment focuses on performing cell instance segmentation. As a constraint, Mask R-CNN or any other two-stage models must be used to perform the detection. Moreover, the final predictions must be encoded in RLE (Run-Length Encoding) format and submitted to Codabench for evaluation.

In my approach, I utilize Mask R-CNN with a ResNet-50 FPN v2 backbone, modified to use smaller anchor sizes to better detect small objects. To further enhance accuracy, I conducted additional experiments using an ensemble strategy.


## How to install
### 1. Clone the Repository 
Begin by cloning the repository to your local machine and navigating to the project directory:  
```bash 
git clone https://github.com/GuanWei926/Selected-Topics-in-Visual-Recognition-using-Deep-Learning.git   
cd Selected-Topics-in-Visual-Recognition-using-Deep-Learning/HW3
```

### 2. Download the dataset 
Use the following command to download the dataset:  
```bash 
pip install gdown
gdown 1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI   
```

### 3. Install Dependencies  
Install the required dependencies using pip:    
```bash 
pip install -r requirements.txt 
```

## How to execute
After downloading the dataset, you do not need to extract it manually. Instead, you can use the "extract the compressed data" section in ```Training.ipynb``` to handle extraction automatically. Once completed, create a directory named hw3-data-release and place the extracted content inside it.

Once the data has been extracted, you do not need to run those sections again.
### Training.ipynb
&nbsp;•   The training.ipynb is used to train a Mask R-CNN model.  

•   You can execute the notebook sequentially from the first cell to the "Training" section without issues.  

•   There many get_model_xxx function in my code, including maskrcnn_resnet50_fpn_v2 model, maskrcnn with mobilenet_v2 backbone model, and maskrcnn with resnext50_32x4d backbone model. You can choose one of the section to execute.

•   In "train the model" section, it defaultly uses get_model_resnet50v2 in the code now. If you want to use another model, you should edit manually.

•   A "plot learning curve" section is also included to make you visualize the learning trend and analyze how the model learns over time. You can use the two cells in this section to plot training vs. validation loss plot and training vs. validation mAP.

•   After training, if you want to test the model and save predictions to a json file in RLE formatf, you can execute the "Testing" section.  

•   In the last "Ensemble" section, you should put the RLE format .json file, which you want to combine, in the directory called predictions first. Next, execute this section then the code will automatically help you to combine all of the .json file into one .json file. 

### 111550061_HW3.pdf
•  This file is the report for the HW3 assignment. It provides information on the methods, experiments, and results.

## Performance snapshot
