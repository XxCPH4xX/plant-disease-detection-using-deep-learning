# Plant Disease Detection Using Deep Learning

A deep learning-based plant disease recognition system that identifies **38 different diseases** across **14 plant species** from leaf images. Built with TensorFlow/Keras and deployed as a Streamlit web app.

## Features

- **38 disease classes** covering major crops (Apple, Tomato, Potato, Corn, Grape, etc.)
- **CNN model** with 10 convolutional layers achieving high accuracy
- **Streamlit web UI** - upload an image and get instant predictions
- **Training visualization** - accuracy plots, confusion matrix, classification report
- **90K+ images** dataset with train/validation/test splits

## Plant Classes

| Plant | Diseases |
|-------|----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

## Project Structure

```
plant-disease-detection-using-deep-learning/
|-- Train_Plant_Disease.ipynb    # Model training notebook
|-- Test_Plant_Disease.ipynb     # Model evaluation notebook
|-- app.py                       # Streamlit web application
|-- trained_model.keras          # Trained Keras model
|-- trained_plant_disease_model.keras
|-- training_hist.json           # Training history
|-- home_page.jpg                # Web app home image
|-- train/                       # Training images (70,295)
|   |-- Apple___Apple_scab/
|   |-- Tomato___Bacterial_spot/
|   +-- ... (38 classes)
|-- valid/                       # Validation images (17,572)
|   +-- ... (38 classes)
|-- test/                        # Test images (33)
|   +-- *.JPG
+-- Results Visualization/
    |-- confusion_matrix.png
    +-- train_val_acc.png
```

## Model Architecture

```
Conv2D(32) -> Conv2D(32) -> MaxPool
Conv2D(64) -> Conv2D(64) -> MaxPool
Conv2D(128) -> Conv2D(128) -> MaxPool
Conv2D(256) -> Conv2D(256) -> MaxPool
Conv2D(512) -> Conv2D(512) -> MaxPool
Dropout(0.25) -> Flatten -> Dense(1500) -> Dropout(0.5) -> Dense(38, softmax)
```

- **Input:** 128x128x3 RGB images
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Categorical Crossentropy
- **Epochs:** 10

## Quick Start

```bash
# Install dependencies
pip install tensorflow streamlit numpy matplotlib seaborn scikit-learn

# Run the web app
streamlit run app.py
```

## Results

- Training and validation accuracy visualized in `Results Visualization/`
- Full classification report and confusion matrix in the evaluation notebook

## Dataset

~87K RGB images of healthy and diseased crop leaves, split 80/20 for training and validation. Original dataset available on [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).

## License

MIT
