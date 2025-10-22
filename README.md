# Brain Tumor Detection System

## Overview
This project is a **Brain Tumor Detection System** that uses deep learning and natural language processing (NLP) to classify brain tumors from MRI images and analyze symptoms. It includes a Tkinter-based GUI, a Streamlit web application, and scripts for training and evaluating a MobileNetV2 model. The system leverages the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) for training and testing.

## Project Structure
- `BrainGui.py`: A Tkinter-based GUI for analyzing MRI images and symptoms.
- `app.py`: A Streamlit web application for interactive brain tumor analysis with patient history management.
- `Brain.py`: A Jupyter notebook (converted to Python) for training and evaluating the MobileNetV2 model.
- `accuracy.py`: A script to evaluate the model's test accuracy.
- `my_brain_tumor_mobilenetv2.h5`: Pre-trained MobileNetV2 model (not included in the repository due to size; referenced from the dataset).
- `requirements.txt`: Lists the Python dependencies required to run the project.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/itsmeusaid/brain-tumor-detection.git
   cd brain-tumor-detection

Set Up a Virtual Environment (recommended):
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
bashpip install -r requirements.txt

Download the Dataset:

Download the Brain Tumor MRI Dataset from Kaggle.
Place the Training and Testing folders in the directory D:\Intern base\Brain\ or update the paths in Brain.py, accuracy.py, and app.py to match your local dataset location.


Download the Pre-trained Model:

Ensure the pre-trained model my_brain_tumor_mobilenetv2.h5 is placed in D:\Intern base\Project\ or update the model path in BrainGui.py, accuracy.py, and app.py.



Usage
Running the Tkinter GUI
bashpython BrainGui.py

Upload an MRI image and enter symptoms to get predictions.

Running the Streamlit App
bashstreamlit run app.py

Access the web app in your browser, upload MRI images, input symptoms, and manage patient history.

Training the Model
bashpython Brain.py

Trains the MobileNetV2 model and saves it as my_brain_tumor_mobilenetv2.h5.

Evaluating Model Accuracy
bashpython accuracy.py

Evaluates the model's accuracy on the test dataset.

Dependencies
Listed in requirements.txt:

tensorflow
numpy
matplotlib
pillow
scikit-learn
streamlit
pandas
python-dateutil

Notes

The dataset and pre-trained model are not included in the repository due to their size. Update file paths in the scripts to match your local setup.
For research and educational purposes only. Consult healthcare professionals for medical diagnoses.

License
This project is licensed under the MIT License.
