# VisionGuard: CNN-Based Multi-Class Eye Disease Detection

## Overview

**VisionGuard** is a deep learning-powered system designed for the automated, rapid, and accurate detection of multiple eye diseases from retinal fundus images. Eye diseases such as Cataract, Glaucoma, and Diabetic Retinopathy are leading causes of preventable blindness worldwide. Early detection is critical, but access to specialist ophthalmologists is often limited, leading to delayed diagnoses.

This project leverages a Convolutional Neural Network (CNN) to classify retinal images into four distinct categories: **Normal, Cataract, Glaucoma, and Diabetic Retinopathy**. By deploying a lightweight and efficient model (MobileNetV2), VisionGuard aims to provide an accessible web-based tool for healthcare professionals and patients, facilitating remote screening and improving diagnostic efficiency.

---

## Problem Statement

The primary challenge in ophthalmology is the timely diagnosis of eye diseases. Traditional screening methods are often slow, costly, and require on-site expert analysis, creating a significant barrier in remote or underserved regions. This leads to a high risk of irreversible vision loss that could have been prevented with early intervention. There is a critical need for an automated system that can:
*   Accurately classify multiple eye diseases from a single fundus image.
*   Provide instant diagnostic feedback.
*   Be easily accessible to non-experts for preliminary screening.

---

## Features

*   **Multi-Class Classification:** Accurately identifies and distinguishes between Normal, Cataract, Glaucoma, and Diabetic Retinopathy.
*   **High Efficiency:** Utilizes the **MobileNetV2** architecture, a lightweight model perfect for deployment in web or mobile applications without sacrificing significant accuracy.
*   **Robust Preprocessing Pipeline:** Employs a series of image enhancement techniques, including resizing, normalization, contrast enhancement, and data augmentation, to improve model performance and generalization.
*   **Web-Based User Interface:** A simple and intuitive UI allows users to upload a retinal image and receive an instant prediction, making it accessible for remote screening.
*   **Proven Performance:** Achieves an overall accuracy of **86.5%** and a weighted average F1-score of **82%**, with particularly high precision (90.9%) in detecting Cataracts.

---

## Technical Workflow

The project follows a structured deep learning pipeline from data preparation to model deployment.

1.  **Dataset:** The model is trained on the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset, which contains thousands of high-quality, labeled fundus images across different disease categories.
2.  **Image Preprocessing:**
    *   **Resizing:** All images are standardized to `224x224` pixels to ensure consistent input for the CNN.
    *   **Normalization:** Pixel values are scaled to a range of [0, 1] to stabilize the training process.
    *   **Contrast Enhancement:** Applied to make key retinal features, such as optic discs and blood vessels, more prominent.
    *   **Data Augmentation:** Techniques like rotation, flipping, and zooming are used to artificially expand the dataset, helping the model learn more robust features and prevent overfitting.
3.  **Model Training (Transfer Learning):**
    *   The **MobileNetV2** model, pre-trained on ImageNet, is used as the base architecture.
    *   **Fine-tuning** is performed by unfreezing the **last 30 layers** of the network. This allows the model to adapt its learned high-level features (like textures and shapes) to the specific task of identifying eye diseases.
    *   The model is trained using the **Adam optimizer** and **Categorical Cross-Entropy** loss function for 30 epochs with early stopping.
4.  **Evaluation:** The model's performance is rigorously evaluated using standard metrics, including accuracy, precision, recall, F1-score, and a confusion matrix to analyze misclassification patterns.
5.  **Deployment:** The trained model is integrated into a web application where users can upload an image and receive a real-time diagnosis.

---

## Performance and Results

*   **Overall Accuracy:** 86.5%
*   **Weighted F1-Score:** 82%

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Normal** | 69.1% | 91.0% | 78.6% |
| **Cataract** | 90.9% | 88.3% | 89.6% |
| **Glaucoma** | 81.3% | 72.8% | 76.8% |
| **Diabetic Retinopathy**| 84.1% | 80.5% | 82.3% |

**Key Insights:**
*   The model performs exceptionally well on **Cataract** detection due to the distinct visual opacity in the images.
*   The primary challenge lies in distinguishing **Normal** cases from early-stage **Diabetic Retinopathy**, as the visual features can be subtle and overlapping.
*   The lower precision for the **Normal** class indicates the model sometimes misclassifies a diseased eye as healthy, highlighting an area for future improvement to ensure clinical reliability.

---

## How to Set Up and Run

1.  **Clone the Repository:**
    ```
    git clone https://github.com/your-username/VisionGuard-Eye-Disease-Detection.git
    cd VisionGuard-Eye-Disease-Detection
    ```
2.  **Install Dependencies:**
    *   Create a virtual environment (recommended).
    *   Install the required Python libraries.
    ```
    pip install -r requirements.txt
    ```
3.  **Download the Dataset:**
    *   Download the ODIR-5K dataset from Kaggle or the provided source.
    *   Organize the images into separate folders for each class (Normal, Cataract, Glaucoma, Diabetic_Retinopathy).

4.  **Train the Model:**
    *   Open the training script (e.g., `train_model.ipynb` or `train.py`).
    *   Update the dataset paths.
    *   Run the script to start the training and fine-tuning process. The trained model will be saved as a `.h5` or `.onnx` file.

5.  **Run the Web Application:**
    *   Launch the web application (e.g., using Flask or Streamlit).
    ```
    python app.py
    ```
    *   Open your web browser and navigate to the provided local URL (e.g., `http://127.0.0.1:5000`).
    *   Upload a retinal fundus image to get a diagnosis.

---

## Future Improvements

*   **Improve Accuracy:** Experiment with more advanced architectures like EfficientNet or Vision Transformers (ViT).
*   **Predict Disease Severity:** Extend the model to classify the stage of a disease (e.g., mild, moderate, severe Diabetic Retinopathy).
*   **Explainable AI (XAI):** Integrate features like Grad-CAM to generate heatmaps that show which parts of the image influenced the model's decision.
*   **Mobile Deployment:** Deploy the model on mobile devices for offline use in low-resource settings.
