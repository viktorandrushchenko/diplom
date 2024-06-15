#!/usr/bin/env python
# coding: utf-8
"""
Object Detection Test
=====================
"""

import os
import json
import tensorflow as tf
import numpy as np
from six import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import warnings

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Настройка директорий и файлов
DATA_DIR = os.path.join(os.getcwd(), 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
IMAGE_FILENAMES = ['image2.jpg', 'image3.jpg']
MODEL_NAME = 'mymodel1'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
LABEL_FILENAME = 'label_map.pbtxt'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))

# Настройка TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Загрузка конфигурации модели
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Восстановление контрольной точки модели
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

warnings.filterwarnings('ignore')

def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

MIN_SCORE_THRESH = 0.5  # Порог уверенности

for image_filename in IMAGE_FILENAMES:
    print('Running inference for {}... '.format(image_filename), end='')

    image_path = os.path.join(IMAGES_DIR, image_filename)
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=MIN_SCORE_THRESH,
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.savefig(os.path.join(IMAGES_DIR, f'detected_{image_filename}'))
    print('Done')

    # Сохранение разметки в формате JSON, только для детекций с порогом уверенности выше MIN_SCORE_THRESH
    valid_detections = {
        "detection_boxes": [],
        "detection_classes": [],
        "detection_scores": []
    }

    for i in range(detections['detection_boxes'][0].shape[0]):
        if detections['detection_scores'][0][i].numpy() >= MIN_SCORE_THRESH:
            valid_detections["detection_boxes"].append(detections['detection_boxes'][0][i].numpy().tolist())
            valid_detections["detection_classes"].append((detections['detection_classes'][0][i].numpy() + label_id_offset).astype(int).tolist())
            valid_detections["detection_scores"].append(detections['detection_scores'][0][i].numpy().tolist())

    json_filename = os.path.join(IMAGES_DIR, f'{os.path.splitext(image_filename)[0]}.json')
    with open(json_filename, 'w') as json_file:
        json.dump(valid_detections, json_file, indent=4)

plt.show(block=True)
print("Ok")
