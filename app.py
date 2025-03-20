import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("dogclassification.h5")

# Define class names
class_names = {
    "0": "Afghan", "1": "African Wild Dog", "2": "Airedale", "3": "American Hairless",
    "4": "American Spaniel", "5": "Basenji", "6": "Basset", "7": "Beagle",
    "8": "Bearded Collie", "9": "Bermaise", "10": "Bichon Frise", "11": "Blenheim",
    "12": "Bloodhound", "13": "Bluetick", "14": "Border Collie", "15": "Borzoi",
    "16": "Boston Terrier", "17": "Boxer", "18": "Bull Mastiff", "19": "Bull Terrier",
    "20": "Bulldog", "21": "Cairn", "22": "Chihuahua", "23": "Chinese Crested",
    "24": "Chow", "25": "Clumber", "26": "Cockapoo", "27": "Cocker",
    "28": "Collie", "29": "Corgi", "30": "Coyote", "31": "Dalmation",
    "32": "Dhole", "33": "Dingo", "34": "Doberman", "35": "Elk Hound",
    "36": "French Bulldog", "37": "German Shepherd", "38": "Golden Retriever",
    "39": "Great Dane", "40": "Great Pyrenees", "41": "Greyhound",
    "42": "Groenendael", "43": "Irish Spaniel", "44": "Irish Wolfhound",
    "45": "Japanese Spaniel", "46": "Komondor", "47": "Labradoodle",
    "48": "Labrador", "49": "Lhasa", "50": "Malinois", "51": "Maltese",
    "52": "Mex Hairless", "53": "Newfoundland", "54": "Pekinese", "55": "Pit Bull",
    "56": "Pomeranian", "57": "Poodle", "58": "Pug", "59": "Rhodesian",
    "60": "Rottweiler", "61": "Saint Bernard", "62": "Schnauzer",
    "63": "Scotch Terrier", "64": "Shar-Pei", "65": "Shiba Inu",
    "66": "Shih-Tzu", "67": "Siberian Husky", "68": "Vizsla", "69": "Yorkie"
}

# Style the app background
st.markdown("""
    <style>
    .stApp {
        background: url('https://source.unsplash.com/1600x900/?dog') no-repeat center center fixed;
        background-size: cover;
    }
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>🐶 Dog Breed Detection App 🐶</h1>", unsafe_allow_html=True)

# Upload an image
uploaded_image = st.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_image is None:
    st.image("https://source.unsplash.com/500x300/?cute-dog", caption="Upload an image to get started!", use_container_width=True)
else:
    image_data = tf.image.decode_image(uploaded_image.read(), channels=3)
    image_data = tf.image.resize(image_data, (224, 224))
    image_data = np.expand_dims(image_data, axis=0) / 255.0

    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        prediction = model.predict(image_data)
        predicted_class = np.argmax(prediction, axis=-1)
        
        st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color:#DC143C; text-align: center;">
                <h2> 🐶 Predicted Breed: {class_names[str(predicted_class[0])]} </h2>
            </div>
            """, unsafe_allow_html=True)
