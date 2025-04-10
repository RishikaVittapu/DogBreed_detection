import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import requests
import random

# Load the trained model with error handling
try:
    model = tf.keras.models.load_model("dogclassification.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define class names as a dictionary
class_names = {
    "0": "Afghan", "1": "African Wild Dog", "2": "Airedale", "3": "American Hairless", "4": "American Spaniel",
    "5": "Basenji", "6": "Basset", "7": "Beagle", "8": "Bearded Collie", "9": "Bermaise", "10": "Bichon Frise",
    "11": "Blenheim", "12": "Bloodhound", "13": "Bluetick", "14": "Border Collie", "15": "Borzoi",
    "16": "Boston Terrier", "17": "Boxer", "18": "Bull Mastiff", "19": "Bull Terrier", "20": "Bulldog",
    "21": "Cairn", "22": "Chihuahua", "23": "Chinese Crested", "24": "Chow", "25": "Clumber", "26": "Cockapoo",
    "27": "Cocker", "28": "Collie", "29": "Corgi", "30": "Coyote", "31": "Dalmation", "32": "Dhole",
    "33": "Dingo", "34": "Doberman", "35": "Elk Hound", "36": "French Bulldog", "37": "German Sheperd",
    "38": "Golden Retriever", "39": "Great Dane", "40": "Great Perenees", "41": "Greyhound", "42": "Groenendael",
    "43": "Irish Spaniel", "44": "Irish Wolfhound", "45": "Japanese Spaniel", "46": "Komondor", "47": "Labradoodle",
    "48": "Labrador", "49": "Lhasa", "50": "Malinois", "51": "Maltese", "52": "Mex Hairless", "53": "Newfoundland",
    "54": "Pekinese", "55": "Pit Bull", "56": "Pomeranian", "57": "Poodle", "58": "Pug", "59": "Rhodesian",
    "60": "Rottweiler", "61": "Saint Bernard", "62": "Schnauzer", "63": "Scotch Terrier", "64": "Shar_Pei",
    "65": "Shiba Inu", "66": "Shih-Tzu", "67": "Siberian Husky", "68": "Vizsla", "69": "Yorkie"
}
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #FF4500;
        padding: 10px;
        background: linear-gradient(to right, #99ccff, #FFD000);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .feedback-buttons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    </style>
    <div class='title'>🐶 Dog Breed Detection & Game</div>
    """,
    unsafe_allow_html=True
)


st.markdown("Upload an image, take a photo, or play the game!")

# Upload an image
uploaded_image = st.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg"])

# Button to activate/deactivate camera
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False

if st.button("📷 Open Camera"):
    st.session_state["camera_active"] = True
if st.button("❌ Close Camera"):
    st.session_state["camera_active"] = False

camera_image = None
if st.session_state["camera_active"]:
    camera_image = st.camera_input("Take a photo")

# Image preprocessing function
def preprocess_image(image_data):
    try:
        img = tf.image.decode_image(image_data, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Image classification
if uploaded_image or camera_image:
    input_image = uploaded_image if uploaded_image else camera_image
    img = preprocess_image(input_image.read())

    if img is not None:
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Classify"):
            try:
                prediction = model.predict(img)
                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                
                st.subheader("Top Predictions:")
                valid_predictions = []
                for i, idx in enumerate(top_3_indices):
                    confidence = prediction[0][idx] * 100
                    if confidence >= 30:
                        valid_predictions.append((class_names[str(idx)], confidence))
                        color = "#90EE90" if i == 0 else "#FF6347"
                        icon = "✅" if i == 0 else "❌"
                        st.markdown(f'<div style="background-color:{color};padding:10px;border-radius:5px;">'
                                    f'<b>{icon} {class_names[str(idx)]}</b> ({confidence:.2f}% confidence)</div>',
                                    unsafe_allow_html=True)
                
                if valid_predictions:
                    selected_breed, _ = valid_predictions[0]
                    wiki_breed_name = selected_breed.replace(" ", "_")
                    info_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_breed_name}"
                    response = requests.get(info_url)

                    if response.status_code == 200:
                        breed_info = response.json().get("extract", "No additional information available.")
                        st.subheader(f"About {selected_breed}:")
                        st.write(breed_info)
                    else:
                        st.write("Could not fetch breed details. Try searching manually on Wikipedia.")
                else:
                    st.write("It is not a dog.")
            except Exception as e:
                st.error(f"Error during classification: {e}")

# Compare dog breeds
st.subheader("Compare Dog Breeds")
breed1 = st.selectbox("Select first breed:", list(class_names.values()))
breed2 = st.selectbox("Select second breed:", list(class_names.values()))
if st.button("Compare Breeds"):
    for breed in [breed1, breed2]:
        wiki_breed_name = breed.replace(" ", "_")
        info_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_breed_name}"
        response = requests.get(info_url)
        if response.status_code == 200:
            st.subheader(f"About {breed}:")
            st.write(response.json().get("extract", "No additional information available."))

# GUESS THE BREED GAME
@st.cache_data(show_spinner="Fetching dog image...")
def get_dog_image(breed):
    breed_url = breed.lower().replace(" ", "").replace("_", "")
    api_url = f"https://dog.ceo/api/breed/{breed_url}/images/random"
    response = requests.get(api_url)

    # If this didn't work, try handling it gracefully
    if response.status_code == 200 and response.json()["status"] == "success":
        return response.json().get("message", "")
    else:
        return "https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg"  # fallback image


def get_random_dog_image():
    # Breeds that work well with Dog CEO API (you can expand this list later)
    supported_breeds = [
        "Beagle", "Golden Retriever", "Labrador", "Pomeranian", "Poodle", "Boxer",
        "German Sheperd", "Doberman", "Bulldog", "Chihuahua", "Shih-Tzu", "Siberian Husky",
        "Pug", "Pit Bull", "Rottweiler", "Yorkie", "Boston Terrier", "Dalmation"
    ]
    correct_breed = random.choice(supported_breeds)
    image_url = get_dog_image(correct_breed)
    return image_url, correct_breed


def start_guess_the_breed():
    st.subheader("🎮 Guess the Breed Game!")

    # Button to start a new game or refresh with a new image
    if st.button("🔄 New Dog / Play Again"):
        for key in ["game_image_url", "correct_breed", "options", "user_guess"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()  # <- Make sure this is inside the button block!

    # If game not already initialized, do it now
    if "game_image_url" not in st.session_state:
        st.session_state["game_image_url"], st.session_state["correct_breed"] = get_random_dog_image()
        st.session_state["options"] = random.sample(list(class_names.values()), 3)
        if st.session_state["correct_breed"] not in st.session_state["options"]:
            st.session_state["options"].append(st.session_state["correct_breed"])
        random.shuffle(st.session_state["options"])

    # Show dog image and options
    st.image(st.session_state["game_image_url"], caption="Guess the Breed!", use_container_width=True)
    user_guess = st.radio("Select the correct breed:", st.session_state["options"], key="user_guess")

    # Handle guess submission
    if st.button("Submit Guess"):
        if user_guess == st.session_state["correct_breed"]:
            st.success("🎉 Correct! You guessed it right!")
        else:
            st.error(f"❌ Wrong! The correct breed was {st.session_state['correct_breed']}.")






# Add the game to the app
start_guess_the_breed()
