import os
import json
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
from six import BytesIO
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

DATA_DIR = os.path.join(os.getcwd(), 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
LABEL_FILENAME = 'label_map.pbtxt'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MIN_SCORE_THRESH = 0.5



# Глобальные переменные для хранения моделей и индекс категорий
loaded_models = {}
category_indexes = {}

# Функция загрузки модели
def load_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name], category_indexes[model_name]

    PATH_TO_MODEL_DIR = os.path.join(MODELS_DIR, model_name)
    PATH_TO_CKPT = os.path.join(PATH_TO_MODEL_DIR, 'checkpoint')
    PATH_TO_CFG = os.path.join(PATH_TO_MODEL_DIR, 'pipeline.config')
    PATH_TO_LABELS = os.path.join(PATH_TO_MODEL_DIR, LABEL_FILENAME)

    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    loaded_models[model_name] = detection_model
    category_indexes[model_name] = category_index

    return detection_model, category_index

model_names = [name for name in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, name))]
for model_name in model_names:
    load_model(model_name)

app = Flask(__name__)

@app.route('/')
def index():
    model_options = "".join([f'<option value="{name}">{name}</option>' for name in model_names])
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Images</title>
    </head>
    <body>
        <h1>Upload Images for Detection</h1>
        <form id="upload-form" enctype="multipart/form-data">
            <label for="model">Choose model:</label>
            <select id="model" name="model">
                ''' + model_options + '''
            </select>
            <br><br>
            <label for="class">Class ID to display:</label>
            <input type="number" id="class" name="class" min="1" required>
            <br><br>
            <input type="file" id="files" name="files" accept="image/*" multiple>
            <br><br>
            <label for="output_path">Output Directory:</label>
            <input type="text" id="output_path" name="output_path" placeholder="Output directory">
            <br><br>
            <button type="button" onclick="previewImages()">Preview</button>
        </form>
        <div id="preview-results"></div>
        <form id="save-form" style="display:none;">
            <button type="button" onclick="saveImages()">Save Images</button>
        </form>
        <script>
            async function previewImages() {
                const formData = new FormData(document.getElementById('upload-form'));
                const response = await fetch('/preview', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (response.ok) {
                    let previewDiv = document.getElementById('preview-results');
                    previewDiv.innerHTML = '<h2>Preview of Detected Images:</h2>';
                    for (let [filename, base64Image] of Object.entries(result.preview_images)) {
                        previewDiv.innerHTML += `<div>
                            <h3>${filename}</h3>
                            <img src="data:image/png;base64,${base64Image}" style="max-width: 500px;"/>
                        </div>`;
                    }
                    document.getElementById('save-form').style.display = 'block';
                } else {
                    alert('Error: ' + result.error);
                }
            }

            async function saveImages() {
                const formData = new FormData(document.getElementById('upload-form'));
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (response.ok) {
                    alert('Images saved successfully!');
                } else {
                    alert('Error: ' + result.error);
                }
            }
        </script>
    </body>
    </html>
    ''')

def load_image_into_numpy_array(data):
    image = Image.open(BytesIO(data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def encode_image_to_base64(image_np):
    image = Image.fromarray(image_np)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/preview', methods=['POST'])
def preview_image():
    if 'files' not in request.files or 'model' not in request.form or 'class' not in request.form:
        return jsonify({"error": "No file part, model, or class selected"}), 400

    model_name = request.form['model']
    class_id = int(request.form['class'])
    detection_model, category_index = load_model(model_name)

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    preview_images = {}
    output_path = request.form.get('output_path', IMAGES_DIR)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in files:
        image_np = load_image_into_numpy_array(file.read())

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor, detection_model)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []

        for i in range(detections['detection_boxes'][0].shape[0]):
            if (detections['detection_scores'][0][i].numpy() >= MIN_SCORE_THRESH and
                (detections['detection_classes'][0][i].numpy() + label_id_offset).astype(int) == class_id):
                filtered_boxes.append(detections['detection_boxes'][0][i].numpy())
                filtered_classes.append((detections['detection_classes'][0][i].numpy() + label_id_offset).astype(int))
                filtered_scores.append(detections['detection_scores'][0][i].numpy())

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            np.array(filtered_boxes),
            np.array(filtered_classes),
            np.array(filtered_scores),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=MIN_SCORE_THRESH,
            agnostic_mode=False)

        preview_images[file.filename] = encode_image_to_base64(image_np_with_detections)

    return jsonify({"preview_images": preview_images})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'files' not in request.files or 'model' not in request.form or 'class' not in request.form:
        return jsonify({"error": "No file part, model, or class selected"}), 400

    model_name = request.form['model']
    class_id = int(request.form['class'])
    detection_model, category_index = load_model(model_name)

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    output_path = request.form.get('output_path', IMAGES_DIR)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in files:
        image_np = load_image_into_numpy_array(file.read())

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor, detection_model)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []

        for i in range(detections['detection_boxes'][0].shape[0]):
            if (detections['detection_scores'][0][i].numpy() >= MIN_SCORE_THRESH and
                (detections['detection_classes'][0][i].numpy() + label_id_offset).astype(int) == class_id):
                filtered_boxes.append(detections['detection_boxes'][0][i].numpy())
                filtered_classes.append((detections['detection_classes'][0][i].numpy() + label_id_offset).astype(int))
                filtered_scores.append(detections['detection_scores'][0][i].numpy())

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            np.array(filtered_boxes),
            np.array(filtered_classes),
            np.array(filtered_scores),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=MIN_SCORE_THRESH,
            agnostic_mode=False)

        detected_image_path = os.path.join(output_path, file.filename)
        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.axis('off')
        plt.savefig(detected_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        valid_detections = {
            "detection_boxes": filtered_boxes,
            "detection_classes": filtered_classes,
            "detection_scores": filtered_scores
        }

        json_filename = os.path.join(output_path, os.path.splitext(file.filename)[0] + '.json')
        with open(json_filename, 'w') as json_file:
            json.dump(valid_detections, json_file, indent=4)

    return jsonify({"message": "Detection completed", "output_directory": output_path})

@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

if __name__ == '__main__':
    app.run(debug=True)