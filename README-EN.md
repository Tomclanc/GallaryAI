Gallary Assistant, categorizing and filtering photos, such as selfies and landscape pictures.


# GallaryAI

[简体中文](README-CN.md) | [English] | [日本語](README_JA.md)

GallaryAI, a photo classification and sorting tool, such as for selfies and landscape photos. This project includes a series of Python scripts for image classification and management, aiming to intelligently categorize and organize photos in a gallery using deep learning models, such as identifying and sorting selfies and landscape photos. The project is implemented based on Convolutional Neural Networks (CNN) and uses PyTorch as the primary deep learning framework. A model trained by me is available for download here: https://link.jscdn.cn/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBckpVdXJnZURwTHNpNE04aDRleHEyaUtKY0lFd1E_ZT1jMERFY0k.pth

### File Structure

```bash
├───gpu.py
├───PyTorch.py
├───cnn.py
├───ceshi.py
├───Start.ps1
├───requirement.txt
└───best_model.pth
```

The project mainly includes scripts for detecting GPU (gpu.py and PyTorch.py), the implementation of Convolutional Neural Networks (cnn.py), model testing scripts (ceshi.py), subsequent automation start scripts (Start.ps1), and the trained model file (best_model.pth).

## Running Guide
This project is based on the Python programming language and primarily uses external libraries such as PyTorch, Pillow (PIL), and NumPy for image processing and training and inference of deep learning models. The program runs on Python version 3.8.18, and it is recommended to configure the Python environment using Anaconda. The following setup process has been tested on the Windows 11 system. Below are the Windows Terminal commands.

### Environment Setup
```bash
# Create a conda environment, name it GallaryAI, and activate it
conda create -n GallaryAI python=3.8.18
conda activate GallaryAI
```


Windows:
```bash 
# Go to the official website to download the corresponding version of PyTorch. Manual installation of the full version of PyTorch is required for GPU training
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Run script programs to test if PyTorch can successfully call GPU
python gpu.py
python PyTorch.py

# Install external libraries
pip install -r requirements.txt
```

### Running Tests

The following commands can be directly run in the project folder:

```bash
cd "Directory"
# Run the Convolutional Neural Network model training script
python cnn.py

# Run the model testing script
python ceshi.py
```

The model weight file is stored in the directory where cnn.py is located, named best_model.pth. The test script ceshi.py by default calls the model file in the directory, which is the model after training completion. To observe the model's performance at different training stages, you can modify the model save path in cnn.py.

If you need to retrain the model, you can run this file in the directory where cnn.py is located.

### Viewing Curves
The project includes Tensorboard curve graphs of the training process, which can be viewed using Tensorboard. It is recommended to use the integrated Tensorboard plugin in VSCode for direct viewing, or you can use the traditional method:

```bash
cd "Directory"
tensorboard --logdir=logs/
```

Open the default address of the Tensorboard service http://localhost:6006/ in a browser to view the interactive curve graphs of the training process.
