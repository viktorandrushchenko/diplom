import tensorflow as tf
dataset = tf.data.TFRecordDataset('D:/TensorFlow/workspace/training_demo/annotations/coco_val.record-00002-of-00050')

for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

# tfrecord_path = 'D:/TensorFlow/workspace/training_demo/annotations/coco_val.record-00001-of-00050'
#
# # Определение функции для разбора примеров из tfrecord
# def parse_tfrecord_fn(example):
#     feature_description = {
#         'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing = True, default_value=""),
#         'image/encoded': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value="")
#     }
#     example = tf.io.parse_single_example(example, feature_description)
#     return example
#
# # Загрузка TFRecord файла
# dataset = tf.data.TFRecordDataset([tfrecord_path])
#
# # Получение первого примера из TFRecord файла
# raw_record = next(iter(dataset))
# example = parse_tfrecord_fn(raw_record)
#
# # Вывод ключей примера
# print(example.keys())