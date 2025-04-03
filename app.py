from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import shutil  # To move files

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("sports_model.h5")

# Define class labels
class_labels = ['american_football', 'baseball', 'basketball', 'billiard_ball',
                'bowling_ball', 'cricket_ball', 'football', 'golf_ball',
                'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock',
                'table_tennis_ball', 'tennis_ball', 'volleyball']

# Configure upload folder
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Save file temporarily
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(temp_path)

            # Move file to static folder for display
            static_path = os.path.join(app.config["STATIC_FOLDER"], file.filename)
            shutil.move(temp_path, static_path)

            # Load image for prediction
            img = image.load_img(static_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict class
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]

            return render_template("index.html", prediction=predicted_class, filename=file.filename)

    return render_template("index.html", prediction=None, filename=None)

# Route to serve uploaded images
@app.route('/static/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
