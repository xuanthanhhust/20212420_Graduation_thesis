# 20212420_Graduation_thesis

This project implements an ECG classification pipeline using Continuous Wavelet Transform (CWT) and Convolutional Neural Networks (CNN). The steps include preprocessing the raw ECG data, transforming it into scalogram images using CWT, splitting the data into training, testing and validating sets, defining a CNN model, and training and evaluating the model. After that, we optimize the tested model by 2 stage: magnitude-based channel pruning and quantization. The last step is implemented at Edge AI: Jetson Nano developer kit.


GitHub Source: (https://github.com/xuanthanhhust/20212420_Graduation_thesis.git)

PROJECT STRUCTURE: (you should get the final result by running these files in the following orders)
1. 1_preprocessing.py

Description: This script processes the raw MIT-BIH ECG data. It performs the following tasks:

  - Baseline wandering removal.
  - Invalid label filtering.
  - R-peak alignment and signal normalization.
  - Conversion of ECG beats into scalogram images using Continuous Wavelet Transform (CWT).
  - Output: A folder containing all the scalogram images for each ECG beat.

2. 2_train_test_valid_split.py

Description: This script accesses the folder containing the scalogram images generated in the preprocessing.py step and splits the data into train, test sets and validation sets. It creates three  folders: train_images, test_images and val_images containing the corresponding images for each set.

3. Class_Model.py

Description: This file defines the class for the Convolutional Neural Network (CNN) model that will be used for ECG classification. The model is designed to accept scalogram images and classify them into different categories based on the type of heartbeats.

4. Class_Dataset.py

Description: This script contains the class definition for the dataset used in the training process. It handles loading images from the train_images and test_images folders, as well as preprocessing them to be ready for CNN input.

5. 3_training.py

Description: This script handles the training process. It loads the dataset, initializes the CNN model, and trains the model using the training data.

6. 4_testing.py

Description: This script is used to evaluate the trained model. It uses the test dataset to predict classifications and then evaluates the performance using accuracy and other metrics.

7. 5_magnitude_channel_pruning.py 

Description: This script implements channel-level pruning based on the magnitude of weights, aiming to reduce model complexity while preserving performance. It prunes entire convolutional channels (i.e., filters) by evaluating their importance using the L1-norm of the kernel weights.

8. 6_AIMET_quantization.py 
   7_AIMET_onnx.py 

Description: This script performs Quantization Aware Training (QAT) using the AIMET (AI Model Efficiency Toolkit) library developed by Qualcomm. QAT simulates quantization effects during training to minimize accuracy loss when deploying low-precision INT8 models on edge devices.

The script firstly equalizes layer, folds batch-normalization layer into convolutional layer and finally, applies fake quantization modules to a pretrained model and fine-tunes it to adapt to quantization noise, enabling export to ONNX or TensorRT with near-FP32 accuracy.
