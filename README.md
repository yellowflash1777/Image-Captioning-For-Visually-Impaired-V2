# Automated Image Captioning for Visually Impaired People 

## Overview

Automated Image Captioning for Visually Impaired People is a project designed to provide descriptive captions for images, enhancing accessibility for individuals with visual impairments. The project utilizes deep learning techniques, specifically the ExpansionNet v2 model, to generate natural language captions for a diverse set of images.

## Project Structure

The project is organized into several key components:

1. **Dataset Preparation:**
   - Utilized the Microsoft COCO dataset, a diverse image dataset with multiple captions per image.
   - Split the dataset into training, validation, and test sets.

2. **Model Training:**
   - Employed the ExpansionNet v2, a deep learning architecture for image captioning.
   - Utilized a pre-trained model as the backbone.
   - Trained the model using the training dataset to generate captions.

3. **Deployment:**
   - Deployed the trained model on an embedded computing board for real-time caption generation.
   - Captured images with a camera or from a local device.
   - Resized and preprocessed the captured images for input to the model.

4. **Caption Generation:**
   - Passed the preprocessed images through the deployed model to generate descriptive captions.

5. **Accessibility:**
   - Transformed the generated captions into an accessible format, such as audio or text-to-speech, to cater to the needs of visually impaired users.

6. **Inference Optimization:**
   - Employed optimization techniques to enhance inference speed and efficiency for real-time performance.

7. **Software Development:**
   - Implemented in Python, utilizing libraries such as OpenCV for image processing and Tesseract OCR for text extraction.
   - Leveraged deep learning frameworks like PyTorch for building and training AI models.
   - Used Flask for backend development, providing a web interface for users.
