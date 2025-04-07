from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import shutil

app = Flask(__name__)

model = tf.keras.models.load_model("sports_model.h5")

class_labels = ['american_football', 'baseball', 'basketball', 'billiard_ball',
                'bowling_ball', 'cricket_ball', 'football', 'golf_ball',
                'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock',
                'table_tennis_ball', 'tennis_ball', 'volleyball']

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(temp_path)

            static_path = os.path.join(app.config["STATIC_FOLDER"], file.filename)
            shutil.move(temp_path, static_path)

            img = image.load_img(static_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]

            return render_template("index.html", prediction=predicted_class, filename=file.filename)

    return render_template("index.html", prediction=None, filename=None)

@app.route('/static/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)

# ✅ Make it Render-compatible
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
