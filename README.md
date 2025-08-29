# AI-Driven Pneumonia Detection

## Overview
This project develops an AI-powered assistant that can accurately detect pneumonia from chest X-ray images.  
By leveraging deep learning techniques, the system reduces reliance on radiologists, improves diagnostic accuracy, and enables faster treatment decisions.  
The project integrates data preprocessing, model training, transfer learning, and a Streamlit-based user interface to deliver predictions in real time.

## Features
- Automated pneumonia detection from chest X-rays using Convolutional Neural Networks (CNNs)
- Transfer learning with VGG16 for improved accuracy and generalization
- Data preprocessing and augmentation for robust training
- Streamlit-based interface for easy image upload and diagnosis visualization
- Integrated AI explanation to improve trust and interpretability
- Optimized for clinical use and resource-limited healthcare environments

## Project Structure
data/
    pneumonia_xray_dataset/          - Training and validation chest X-ray datasets  
    test_samples/                    - Example test X-rays for evaluation  

Pneumonia Prediction.ipynb           - Jupyter Notebook with model training and evaluation  
report/                              - Annotated project report (PDF/DOCX)  

app/  
    streamlit_app.py                 - Interactive Streamlit interface  

## Technical Implementation

### Data Collection
- Public datasets:  
  - Zhejiang University/Kaggle Pneumonia Dataset (~5,800 images)  
  - Mendeley Pneumonia Dataset (thousands of annotated samples)  

### Preprocessing
- Convert all images to RGB
- Resize to 512x512 pixels
- Normalize pixel values to [0,1]
- Split into training/validation/test sets (80/20)

### Data Augmentation
- Rotation ±15 degrees
- Width/height shifts up to 10%
- Shear and zoom ±10%
- Horizontal flip
- Rescale 1./255

### Model Development
- **Baseline CNN:** 3 convolutional layers, batch normalization, max pooling, dropout, sigmoid output  
- **VGG16 Transfer Learning:** pretrained convolutional base + custom dense layers, dropout, sigmoid output  

### Training Setup
- Binary cross-entropy loss
- Adam optimizer (learning rate = 0.0001)
- EarlyStopping and ModelCheckpoint callbacks
- Trained for multiple epochs, best model saved

### Performance
- Baseline CNN: ~62% accuracy, biased toward pneumonia class  
- VGG16 transfer model: ~93% accuracy, balanced precision and recall  
- Validation learning curves show high accuracy and low loss after 2–3 epochs

## Results
- Achieved 93% accuracy using VGG16 transfer learning  
- Balanced performance across “Normal” and “Pneumonia” classes  
- Streamlit UI provides instant, user-friendly predictions for uploaded X-rays  
- Business relevance: reduces workload on radiologists, improves speed, delivers expert-level diagnostics to underserved regions

## Future Work
- Extend to multi-class classification (bacterial vs viral pneumonia)
- Integrate Explainable AI (Grad-CAM) for transparency
- Use federated learning for privacy-preserving collaboration across hospitals
- Deploy in cloud/edge environments for real-time clinical use

## Challenges
- Limited computational resources restricted batch sizes and training depth
- Need for interpretability of deep learning outputs to support clinical adoption

## Tech Stack
- Python, TensorFlow/Keras
- VGG16 pretrained model (ImageNet)
- Streamlit (for deployment UI)
- Pandas, NumPy, OpenCV (data preprocessing)

## Author
Prasham Shah   
LinkedIn: https://www.linkedin.com  
GitHub: https://github.com  
