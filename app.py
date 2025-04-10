import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ✅ Cache the model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dogclassification.h5")

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
breed_facts = {
    "Golden Retriever": {
        "fact": "Always ready for a swim and obsessed with fetch!",
        "favorite_food": "Peanut butter and chicken bites.",
        "personality": "Friendly, loyal, and family-loving."
    },
    "Labrador": {
        "fact": "Can sniff out food in your bag from miles away.",
        "favorite_food": "Grilled fish and sweet potatoes.",
        "personality": "Gentle, intelligent, and playful."
    },
    "Beagle": {
        "fact": "Incredible nose — they follow scents like detectives!",
        "favorite_food": "Apple slices and turkey treats.",
        "personality": "Curious, merry, and stubborn."
    },
    "Pug": {
        "fact": "They snore adorably and love cuddling.",
        "favorite_food": "Pumpkin puree and soft kibble.",
        "personality": "Goofy, loving, and social."
    },
    "Bulldog": {
        "fact": "They overheat easily — fans are their best friends.",
        "favorite_food": "Banana chunks and beef stew.",
        "personality": "Chill, brave, and affectionate."
    },
    "German Shepherd": {
        "fact": "They excel in police and military tasks.",
        "favorite_food": "Cooked eggs and lamb chunks.",
        "personality": "Loyal, confident, and smart."
    },
    "Siberian Husky": {
        "fact": "Talkative and drama queens when bored.",
        "favorite_food": "Frozen blueberries and salmon.",
        "personality": "Energetic, vocal, and adventurous."
    },
    "Boxer": {
        "fact": "Named for their playful “boxing” behavior.",
        "favorite_food": "Chicken liver and peas.",
        "personality": "Cheerful, brave, and goofy."
    },
    "Doberman": {
        "fact": "Built for speed and elegance — guard dog royalty.",
        "favorite_food": "Turkey meatballs and green beans.",
        "personality": "Alert, fearless, and loyal."
    },
    "Dalmation": {
        "fact": "Born without spots — they develop as they grow!",
        "favorite_food": "Liver biscuits and rice.",
        "personality": "Energetic, smart, and loyal."
    },
    "Rottweiler": {
        "fact": "Descended from Roman cattle dogs!",
        "favorite_food": "Beef chunks and bone broth.",
        "personality": "Protective, calm, and confident."
    },
    "Shih-Tzu": {
        "fact": "Were once palace lap warmers in ancient China.",
        "favorite_food": "Egg bits and rice.",
        "personality": "Affectionate, alert, and confident."
    },
    "Corgi": {
        "fact": "Their bums became memes — and we love it.",
        "favorite_food": "Cheddar bits and carrots.",
        "personality": "Bold, alert, and silly."
    },
    "French Bulldog": {
        "fact": "Can’t swim well — those big heads are heavy!",
        "favorite_food": "Cooked veggies and duck treats.",
        "personality": "Charming, calm, and snorty."
    },
    "Chihuahua": {
        "fact": "Small dog, big drama.",
        "favorite_food": "Boiled chicken and rice.",
        "personality": "Alert, feisty, and fiercely loyal."
    },
    "Yorkie": {
        "fact": "Tiny but fearless — rat-hunters turned fashion icons.",
        "favorite_food": "Salmon bits and blueberries.",
        "personality": "Bold, affectionate, and lively."
    },
    "Great Dane": {
        "fact": "Despite their size, they think they’re lap dogs.",
        "favorite_food": "Steamed veggies and turkey.",
        "personality": "Gentle, friendly, and huge-hearted."
    },
    "Saint Bernard": {
        "fact": "Used for Alpine rescue — with brandy barrels!",
        "favorite_food": "Beef stew and kibble.",
        "personality": "Gentle giants, sweet and patient."
    },
    "Pit Bull": {
        "fact": "Nicknamed ‘nanny dogs’ for their love of children.",
        "favorite_food": "Peanut butter and beef treats.",
        "personality": "Strong, loyal, and loving."
    },
    "Maltese": {
        "fact": "Royal lapdogs of ancient Rome — always glamorous.",
        "favorite_food": "Pumpkin puree and boiled chicken.",
        "personality": "Playful, gentle, and confident."
    },
    "Poodle": {
        "fact": "The iconic haircut once helped them swim better.",
        "favorite_food": "Turkey bits and carrots.",
        "personality": "Elegant, clever, and friendly."
    },
    "Boston Terrier": {
        "fact": "Nicknamed ‘American Gentleman’ for a reason!",
        "favorite_food": "Peas, carrots, and salmon.",
        "personality": "Lively, smart, and friendly."
    },
    "Shar-Pei": {
        "fact": "Those wrinkles helped them in dog fights — genius armor.",
        "favorite_food": "Rice and fish.",
        "personality": "Reserved, loyal, and clean."
    },
    "Basset": {
        "fact": "They have more scent receptors than any dog except Bloodhounds.",
        "favorite_food": "Chicken stew and apples.",
        "personality": "Calm, loyal, and stubborn."
    },
    "Bloodhound": {
        "fact": "Used by law enforcement due to unmatched tracking skills.",
        "favorite_food": "Ground beef and carrots.",
        "personality": "Determined, loving, and easygoing."
    },
    "Pomeranian": {
        "fact": "Fluffy drama queens with big dog energy.",
        "favorite_food": "Cheese cubes and eggs.",
        "personality": "Bold, lively, and curious."
    },
    "Collie": {
        "fact": "Thanks to Lassie, they became heroic dog icons.",
        "favorite_food": "Turkey and peas.",
        "personality": "Loyal, smart, and graceful."
    },
    "Cocker": {
        "fact": "Cocker Spaniels are champions at fetch and cuddles.",
        "favorite_food": "Chicken chunks and sweet potatoes.",
        "personality": "Playful, sweet, and eager to please."
    },
    "Dachshund": {
        "fact": "Bred to hunt badgers — fearless sausages.",
        "favorite_food": "Boiled chicken and pumpkin.",
        "personality": "Lively, clever, and courageous."
    },
    "Lhasa": {
        "fact": "Considered lucky charms in ancient Tibet.",
        "favorite_food": "Rice and turkey.",
        "personality": "Independent, alert, and loyal."
    },
    "Maltipoo": {
        "fact": "A designer breed known for their hypoallergenic coat.",
        "favorite_food": "Pumpkin and salmon.",
        "personality": "Gentle, playful, and charming."
    },
    "Newfoundland": {
        "fact": "They are incredible swimmers and can save people from drowning.",
        "favorite_food": "Lamb and potatoes.",
        "personality": "Patient, sweet, and strong."
    },
    "Pekinese": {
        "fact": "Were carried in the sleeves of Chinese royalty.",
        "favorite_food": "Boiled chicken and rice.",
        "personality": "Independent, affectionate, and proud."
    },
    "Rhodesian": {
        "fact": "Used to track lions in Africa — no joke!",
        "favorite_food": "Meaty bones and rice.",
        "personality": "Strong-willed, loyal, and alert."
    },
    "Scotch Terrier": {
        "fact": "Was President Roosevelt’s favorite breed!",
        "favorite_food": "Beef stew and carrots.",
        "personality": "Bold, independent, and dignified."
    },
    "Schnauzer": {
        "fact": "Great at detecting pests — and being silly.",
        "favorite_food": "Salmon bites and rice.",
        "personality": "Alert, spirited, and friendly."
    },
    "Shiba Inu": {
        "fact": "They starred in the Doge meme — much wow!",
        "favorite_food": "Cooked chicken and pumpkin.",
        "personality": "Alert, bold, and independent."
    },
    "Vizsla": {
        "fact": "Known as 'Velcro dogs' — they love staying close.",
        "favorite_food": "Turkey and spinach.",
        "personality": "Energetic, affectionate, and sensitive."
    },
    "Whippet": {
        "fact": "Can reach 35 mph — fastest couch potato!",
        "favorite_food": "Lamb and broccoli.",
        "personality": "Calm, athletic, and loving."
    },
    "Alaskan Malamute": {
        "fact": "Used in Arctic to pull heavy sleds — pure strength.",
        "favorite_food": "Fish and beef.",
        "personality": "Strong, friendly, and loyal."
    },
    "Elk Hound": {
        "fact": "Viking companion dogs — tough and loyal.",
        "favorite_food": "Meaty treats and potatoes.",
        "personality": "Confident, dependable, and alert."
    }
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
        predicted_breed = class_names[str(predicted_class[0])]

        st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color:#DC143C; text-align: center;">
                <h2> 🐶 Predicted Breed: {predicted_breed} </h2>
            </div>
            """, unsafe_allow_html=True)

        # Show breed fact (inside the same block)
        fact_data = breed_facts.get(predicted_breed)
        if fact_data:
            st.success(f"📚 **Fun Fact:** {fact_data['fact']}")
            st.write(f"💖 **Personality:** {fact_data['personality']}")
            st.write(f"🍗 **Favorite Food:** {fact_data['favorite_food']}")
        else:
            st.warning("No extra info available for this breed yet.")

        if st.button("🔁 Try Another"):
             st.experimental_rerun()
