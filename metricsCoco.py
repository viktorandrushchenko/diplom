import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO

# Загрузка модели
loaded_model = tf.saved_model.load('C:/Users/vitya/PycharmProjects/syg2/data/models/mymodel1/saved_model')
model = loaded_model.signatures['serving_default']

# Загрузка тестовых данных COCO
ann_file = 'D:/publaynet/val.json'
coco = COCO(ann_file)

# Идентификатор класса "title"
class_id = coco.getCatIds(catNms=['title'])[0]

# Получение изображений с аннотациями для класса "title"
img_ids = coco.getImgIds(catIds=[class_id])
images = coco.loadImgs(img_ids)

# Предобработка и подготовка тестовых данных
test_data_processed = []
test_labels = []

for image in images:
    img_path = 'D:/publaynet/val/' + image['file_name']
    print(img_path)
    img = tf.io.decode_image(tf.io.read_file(img_path), channels=3)
    img = tf.image.resize(img, (640, 640))  # Измените размер изображения на нужный
    img = img / 255.0  # Нормализация значений пикселей
    test_data_processed.append(img.numpy())
    test_labels.append(2)

test_data_processed = np.array(test_data_processed, dtype=np.uint8)
test_labels = np.array(test_labels)

# Получение предсказаний модели на тестовых данных
predictions = model(tf.constant(test_data_processed))  # Получение предсказаний модели
predicted_classes = np.argmax(predictions)  # Получение индексов классов с наибольшей вероятностью
#print(class_id)
# for label in test_labels:
#     print(label)

# Вычисление метрик precision, recall и F1 для класса "title"
true_positives = np.sum(np.logical_and(predicted_classes == class_id, test_labels == 2))
false_positives = np.sum(np.logical_and(predicted_classes == class_id, test_labels != 2))
false_negatives = np.sum(np.logical_and(predicted_classes != class_id, test_labels == 2))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

# Вывод результатов
print('Precision (class "title"):', precision)
print('Recall (class "title"):', recall)
print('F1 (class "title"):', f1)