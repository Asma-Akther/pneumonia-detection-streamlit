# Pneumonia Detection using Deep Learning

## Project Overview

This project focuses on detecting pneumonia from chest X-ray images using deep learning techniques. The system classifies images into two categories:

* NORMAL
* PNEUMONIA

Multiple models were implemented and compared, including a custom Convolutional Neural Network (CNN) and transfer learning models such as ResNet50 and EfficientNetB0.

---

## Objectives

* Build an automated system for pneumonia detection
* Compare performance of different deep learning models
* Apply transfer learning for improved accuracy
* Evaluate models using proper classification metrics
* Deploy the trained model using Streamlit

---

## Project Structure

```
Pneumonia-Detection/
│
├── Pneumonia_Detection_Group07_Reorganized.ipynb
├── pneumonia_streamlit_app.py
├── models/
├── dataset/
├── README.md
└── requirements.txt
```

---

## Models Used

### 1. Custom CNN

* Built from scratch using Conv2D, MaxPooling, Dense layers
* Serves as a baseline model

### 2. ResNet50 (Transfer Learning)

* Pretrained on ImageNet
* Feature extraction with custom classification head

### 3. EfficientNetB0

* Advanced architecture with improved efficiency and performance
* Used for better accuracy and generalization

---

## Technologies and Libraries

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit

---

## Workflow

1. Data Loading

   * Images loaded using directory-based structure

2. Data Preprocessing

   * Resizing images
   * Normalization
   * Data augmentation

3. Model Training

   * Training CNN, ResNet50, EfficientNetB0
   * Using callbacks such as EarlyStopping and ReduceLROnPlateau

4. Evaluation

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix

5. Prediction

   * Classifying new X-ray images

6. Deployment

   * Streamlit application for user interaction

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics evaluate how well the model distinguishes between NORMAL and PNEUMONIA cases.

---

## How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook

```
jupyter notebook
```

### 4. Run Streamlit App

```
streamlit run pneumonia_streamlit_app.py
```

---

## Streamlit Application (User Interface)

The project includes an interactive Streamlit-based user interface for real-time pneumonia detection.

### Key Features

* Upload one or multiple chest X-ray images
* Select model for prediction (CNN, ResNet50, EfficientNetB0)
* View predictions with confidence scores
* Display results in tabular format
* Download prediction results as CSV
* Visual display of uploaded images with predicted labels
* Summary metrics (NORMAL, PNEUMONIA, Invalid/Error cases)
* Supports screen recording of the interface for demonstration purposes using external tools

### How It Works

* Users upload X-ray images through the interface
* Selected model processes the images
* Predictions are generated along with confidence scores
* Results are displayed instantly in the browser

---

## Results

* Transfer learning models outperform the basic CNN
* EfficientNetB0 provides strong performance with efficiency
* Model evaluation confirms reliability for classification tasks

---

## Limitations

* Model performance depends on dataset quality
* May not generalize well to unseen medical datasets
* Not a replacement for professional medical diagnosis

---

## Future Improvements

* Use larger and more diverse datasets
* Apply model explainability techniques such as Grad-CAM
* Deploy as a web or mobile application
* Perform hyperparameter tuning for better performance

---

## License

This project is for academic and educational purposes only.

---

## Acknowledgements

* Dataset providers
* TensorFlow and Keras documentation
* Open-source community
