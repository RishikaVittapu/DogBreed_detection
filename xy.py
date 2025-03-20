import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model("dogclassification.keras")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("dogclassification.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")
