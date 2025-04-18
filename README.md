# Dog Breed Detection App



## Introduction

The Dog Breed Detection App is a simple Streamlit application that uses a pre-trained machine learning model to predict the breed of a dog based on an uploaded image. Whether you're a dog enthusiast or just curious about your dog's breed, this app can provide you with an answer. Also if you would want to compare dog breeds, there is a feature of comparing breeds. It also a unique feature of "guessing the breed" game. Users can engage in this as it refreshes new dog every time.

## Features
- **Breed Prediction**: Upload a picture of your dog, and the app will predict the most likely breed from a list of 70 different breeds. 
- **Breed comparision**: Select two breeds, then the machine will show brief info about them.
- **Guessing the breed game**: Engaging users with guess the breed game, where users will be provided with a dog picture and four options. They should select right 
                                one.
- **Easy to Use**: The user-friendly interface makes it simple for anyone to use.

## Demo

You can see a live demo of the app [here](https://dog-breed-detector.streamlit.app/)

## Installation

To run the Dog Breed Detection App locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/RishikaVittapu/DogBreed_detection

2. Change the directory to the project folder:

   ```bash
    cd DogBreed_detection
   
3. Install the required dependencies using pip:

   ```bash
    pip install -r requirements.txt


4. Run the Streamlit app:

   ```bash
    streamlit run app1.py

### The app will open in your web browser, allowing you to upload dog images for classification.





# Folder Structure
### The model has been trained on a dataset organized in the following folder structure:

    
        ./dog
        │
        └── Afghan
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   ...
        │
        └── African Wild Dog
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   ...
        ...
        
  Each breed has its own folder containing multiple images.


# Model Details
The Dog Breed Detection App uses a pre-trained machine learning model based on the InceptionV3 architecture. The model has been fine-tuned on a dataset of 70 different dog breeds.

   
