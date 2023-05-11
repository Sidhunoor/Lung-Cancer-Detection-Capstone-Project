# Lung-Cancer-Detection-Capstone-Project
The objective of this application is to detect lung cancer early in order to provide patients with the greatest chance of recovery and survival by employing a CNN model.
# Lung-Cancer-Detection

This application is designed to use a Convolutional Neural Network (CNN) model for early detection of lung cancer to improve patients' chances of recovery and survival. The CNN model will be trained on a large dataset of high-resolution lung scans to accurately distinguish between cancerous and non-cancerous lesions in the lungs. By doing so, it aims to reduce the false positive rate that is a common issue with current detection technologies. Early detection of lung cancer can lead to earlier access to life-saving interventions and better outcomes for patients. Additionally, the use of the CNN model can give radiologists more time to spend with their patients, as they can focus on more complex cases that require human expertise.

![Lung-Cancer-Detection](https://user-images.githubusercontent.com/68781375/162584408-450580c0-3354-470b-a69c-180a19802fd4.jpg)



## Dataset

We have taken 50 patients as a sample dataset for training and validation. Link is available below: 

Sample Dataset Images Link: https://qnm8.sharepoint.com/:f:/g/Ep5GUq573mVHnE3PJavB738Bevue4plkiXyNkYfxHI-a-A?e=UVMWne
For EDA
Link : https://www.kaggle.com/datasets/jillanisofttech/lung-cancer-detection/code?resource=download  


## Output Screenshots

![OutputScreenshot-1](https://user-images.githubusercontent.com/68781375/162584315-359fba81-6827-437f-ab54-b8dee534f1d8.JPG)

The implementation and deployment phase is crucial for making our lung cancer detection model accessible to users. In this phase, we will deploy our application as a desktop GUI app with tkinter and/or as a web app with any of the python web frameworks like Flask, Django, Snowflake, Streamlit, or AWS.

## Desktop GUI app:

A desktop GUI app will allow users to run the application on their local machine without the need for an internet connection. We will use tkinter, a standard GUI package in Python, to develop the desktop app.
Web app:

A web app will allow users to access the application from any device with an internet connection. We will use one of the popular Python web frameworks such as Flask, Django, Snowflake, or Streamlit to develop the web app.

We can also deploy our application on AWS, a cloud computing platform that provides a range of services to build and deploy applications. This will allow us to scale our application according to user demand and make it accessible to a larger audience.

The deployment process involves converting our trained model into a format that can be used by our application. We will use tools such as TensorFlow Serving or Flask to create an API that exposes our model's predictions to the user interface.

Finally, we will deploy the application on a server or a cloud platform such as AWS, Heroku, or Google Cloud. This will make our application available to users worldwide.

In conclusion, the implementation and deployment phase is critical for making our lung cancer detection model accessible to users. We will deploy our application as a desktop GUI app with tkinter and/or as a web app with any of the python web frameworks like Flask, Django, Snowflake, Streamlit, or AWS, and make it available to users worldwide.
