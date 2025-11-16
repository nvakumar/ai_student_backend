from tensorflow.keras.models import load_model

# Load the old model
model = load_model("best_mobilenet_model.h5")

# Save updated model
model.save("best_mobilenet_model_v2.h5")  # Standard updated H5 format

# Or save as TensorFlow SavedModel format
model.save("saved_model/", save_format="tf")
