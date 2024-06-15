import tensorflow as tf
import numpy as np

# Загрузка модели
loaded_model = tf.saved_model.load('C:/Users/vitya/PycharmProjects/syg2/data/models/mymodel1/saved_model')
model = loaded_model.signatures['serving_default']


# Определение функции для разбора примеров из tfrecord
def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=""),
        'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value="")
    }
    example = tf.io.parse_single_example(example, feature_description)
    data = tf.io.decode_raw(example['image/encoded'], tf.uint8)
    label = tf.cast(example['image/object/class/text'], tf.string)
    return data, label


# Загрузка тестовых данных из tfrecord
test_tfrecord_path = 'D:/TensorFlow/workspace/training_demo/annotations/coco_val.record-00001-of-00050'
test_dataset = tf.data.TFRecordDataset(test_tfrecord_path)
test_dataset = test_dataset.map(parse_tfrecord_fn)

# Предобработка и подготовка тестовых данных
test_data_processed = []
test_labels = []

for data, label in test_dataset:
    test_data_processed.append(data.numpy())
    test_labels.append(label.numpy())
    print(test_data_processed)

np.reshape(test_data_processed, (400,292,4))
test_data_processed = np.array(test_data_processed)
test_labels = np.array(test_labels)

# Получение предсказаний модели на тестовых данных
predictions = model(tf.constant(test_data_processed))['output_1'].numpy()
predicted_classes = np.argmax(predictions, axis=1)

# Вычисление метрик precision, recall и F1 для класса "title"
true_positives = np.sum(np.logical_and(predicted_classes == 'title', test_labels == 'title'))
false_positives = np.sum(np.logical_and(predicted_classes == 'title', test_labels != 'title'))
false_negatives = np.sum(np.logical_and(predicted_classes != 'title', test_labels == 'title'))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

# Вывод результатов
print('Precision (class "title"):', precision)
print('Recall (class "title"):', recall)
print('F1 (class "title"):', f1)
