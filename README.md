# Gender Classification Model

This project implements a gender classification system using a MobileNetV3-Small convolutional neural network trained on the UTKFace dataset.

The model predicts gender from facial images:

- 0 = Male  
- 1 = Female  

The model is built using PyTorch with transfer learning from ImageNet pretrained weights and is optimized for efficient CPU inference.

**Training Accuracy:** 96.9%

---

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run prediction:

python

>>> from inference import predict  
>>> predict("image.jpg")

---

## Project Structure

model.pth  
inference.py  
requirements.txt  
model_card.pdf  
README.md  

---

## Dataset

The model is trained on the UTKFace dataset, which contains facial images labeled with age, gender, and ethnicity.

---

## Notes

- This model assumes binary gender classification  
- Performance may vary on unseen real-world data  
