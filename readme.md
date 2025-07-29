# Remote Sensing Spectral Data Classification System (MS-PD Detection)

This is a deep learning-based remote sensing spectral data classification system for distinguishing between diseased and healthy vegetation areas.

## Project Structure

```
ms-pd-detect/
├── data/
│   ├── diseased/          # Diseased spectral data folder (71 .tif images)
│   └── healthy/           # Healthy spectral data folder (130 .tif images)
├── image_classifier.py    # Main classifier class
├── predict_example.py     # Prediction example script
├── requirements.txt       # Dependencies list
└── README.md             # Project documentation
```

## Features

- **Deep Learning Model**: Convolutional Neural Network (CNN) built with PyTorch for spectral data classification
- **GPU Acceleration**: Automatic detection and use of GPU for accelerated training and inference
- **Data Augmentation**: Automatic application of rotation, scaling, flipping and other data augmentation techniques
- **Model Evaluation**: Detailed performance evaluation metrics and visualization
- **Easy to Use**: Simple API interface supporting single spectral image prediction
- **Model Persistence**: Automatic model saving after training with support for subsequent loading

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model

Run the main training script:

```bash
python image_classifier.py
```

This will:
- Automatically load all spectral images from the data folder
- Create and train the CNN model
- Display training progress and performance metrics
- Generate confusion matrix and training history charts
- Save the trained model as `disease_classifier_model.pth`

### 2. Use Trained Model for Prediction

Run the prediction example:

```bash
python predict_example.py
```

Or use in Python code:

```python
from image_classifier import ImageClassifier

# Create classifier instance
classifier = ImageClassifier()

# Load trained model
classifier.load_model('disease_classifier_model.pth')

# Predict single spectral image
result, confidence = classifier.predict_single_image('path/to/your/image.tif')
print(f"Prediction result: {result}, Confidence: {confidence:.4f}")
```

## Model Architecture

The model uses the following architecture:

- **Input Layer**: 224x224x3 RGB spectral images
- **Convolutional Layers**: 4 convolutional blocks, each containing convolution, batch normalization, max pooling and Dropout
- **Fully Connected Layers**: Two fully connected layers with Dropout regularization
- **Output Layer**: Sigmoid activation function for binary classification

## Performance Metrics

After training completion, the system automatically generates:

1. **Accuracy Report**: Shows accuracy on test set
2. **Classification Report**: Includes precision, recall, F1-score
3. **Confusion Matrix**: Visualizes classification results
4. **Training History**: Shows accuracy and loss changes during training

## Data Requirements

- Supports TIFF format spectral images
- Images are automatically resized to 224x224 pixels
- Automatic conversion to RGB format (if original is grayscale)
- Data is automatically normalized to [0,1] range

## Custom Configuration

You can customize by modifying the `ImageClassifier` class parameters:

```python
# Custom image size and data path
classifier = ImageClassifier(
    data_dir='your_data_path',
    img_size=(256, 256)  # Custom image size
)

# Custom training parameters
history, X_test, y_test = classifier.train_model(
    X, y, 
    test_size=0.3,      # Test set ratio
    epochs=50,          # Training epochs
    batch_size=16       # Batch size
)
```

## Notes

1. Ensure sufficient GPU memory (8GB+ recommended)
2. Training time depends on data volume and hardware configuration
3. Model file size is approximately tens of MB
4. Recommend backing up original data before training

## Troubleshooting

**Issue**: Out of memory error
**Solution**: Reduce batch_size or image size

**Issue**: Image loading failure
**Solution**: Check if image file format and path are correct

**Issue**: Poor model performance
**Solution**: Try increasing training epochs or adjusting model architecture

## Technology Stack

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision tools
- **scikit-learn**: Machine learning tools
- **PIL/Pillow**: Image processing
- **matplotlib/seaborn**: Data visualization
- **NumPy**: Numerical computing
- **tqdm**: Progress bar display

## License

This project is for learning and research purposes only.