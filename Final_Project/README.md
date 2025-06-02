# NYCU Selected Topics in Visual Recognition using Deep Learning HW4
Group: weak baseline
Student ID: 111550061 111550066 111550089 111550098   
Name: 邱冠崴 王裕昕 李宗諺 楊宗儒

## Introduction
Steller sea lions in the western Aleutian Islands have undergone a dramatic population decline of approximately 94% over the past 30 years. To monitor their recovery, researchers at NOAA Fisheries' Alaska Fisheries Science Center conduct annual aerial surveys, capturing thousands of high-resolution images of sea lion rookeries and haulouts. Currently, trained biologists manually annotate and count sea lions in these images — a time-consuming task that can take up to four months to complete.

The goal of this project is to develop an automated system for accurately counting sea lions in aerial imagery, thereby reducing the time and human effort required for population monitoring.

This assignment focuses on building a model to estimate the number of sea lions in each image. The final predictions must be saved in a .csv file, reporting the count of sea lions in each class, and submitted to Kaggle for evaluation using Root Mean Square Error (RMSE).

In our approach, we utilize two-stage models to perform this task. Firstly, we will train the Unet. Subsequently, utilize the trained Unet to generate the per-class probability map of each images which is not used to train Unet. Next, use the probability map to train the regressors. After doing all of this, we use the Unet to generate the per-class probability map of test data first. Then, use the regressors to decide the numbers of sea lions of each classes and record it csv file.


## How to install
### 1. Clone the Repository 
Begin by cloning the repository to your local machine and navigating to the project directory:  
```bash 
git clone https://github.com/GuanWei926/Selected-Topics-in-Visual-Recognition-using-Deep-Learning.git   
cd Selected-Topics-in-Visual-Recognition-using-Deep-Learning/HW4
```

### 2. Download the dataset 
Use the following command to download the dataset and sample code:  
```bash 
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul 
```

### 3. Install Dependencies  
Install the required dependencies by re-create the environment:    
```bash 
conda env create -f environment.yaml 
```

## How to execute
After downloading the dataset, manually extract the hw4_realset_dataset.zip file. Then, move the extracted contents into the data directory located within the PromptIR project folder.

Please ensure that the training and testing images are stored separately. Different types of training data should be placed in their corresponding subfolders under Derain or Desnow. Additionally, you must update the .txt files located in each subfolder under the data_dir directory to specify the filenames of the training and validation datasets accordingly.

To execute the following code, you should:
```bash 
cd ./PromptIR 
```

### train.py
&nbsp;•   The training.py is used to train a PromptIR model.    

•   To start training, use the following command in your terminal:
```bash 
python train.py --de_type derain desnow --epochs 450 --num_gpus 2 --batch_size 2 --lr 2e-4 --patch_size 224 
```
You can modify the parameters to experiment with different settings.

### inference.py
•   The inference.py script is used to generate restored images using the trained model.

•   Note: Make sure to update the checkpoint path in line 54 to point to your own trained model.

•   Run the code by using:
```bash 
python inference.py
```

### ensemble.py
•   The ensemble.py script performs ensemble inference by averaging the predictions from multiple model checkpoints.

•   To use it, simply add the paths of the checkpoints you want to combine to the `ckpt_paths` list.

•   Run the code by using:
```bash 
python ensemble.py
```

### visualization.py
•   The visualizatioon.py generates comparison images between the original degraded images and the predicted restored outputs.

•   Note: Update the checkpoint path in line 56 to point to your own trained model.

•   You should also specify the filenames of the test images you want to visualize by editing the `image` list.

•   Run the code by using:
```bash 
python visualization.py
```

### 111550061_HW4.pdf
•  This file is the report for the HW4 assignment. It provides information on the methods, experiments, and results.

## Performance snapshot
![alt text](image.png)
