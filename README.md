# Gender Classification Model

This project implements a gender classification system using a MobileNetV3-Small convolutional neural network trained on the UTKFace facial image dataset.

The model predicts gender from facial images:

- 0 = Male  
- 1 = Female  

The model is implemented in PyTorch using transfer learning from ImageNet pretrained weights and optimized for efficient CPU-based inference.

**Final Training Accuracy:** 96.9%

## Inference

To perform prediction, use `inference.py` and call:

predict(image_path)

The function returns:

- `label` (0 = Male, 1 = Female)
- `confidence` (probability score)
