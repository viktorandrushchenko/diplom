import cv2

# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Увеличение контрастности изображения
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(gray_image)

# Применение фильтра гаусса для сглаживания изображения
blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image1.jpg'
cv2.imwrite(output_image_path, blurred_image)

print("Обработанное изображение успешно сохранено.")



# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение оператора Собеля для выделения границ
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_image = cv2.magnitude(sobel_x, sobel_y)

# Применение адаптивного порогового преобразования для усиления границ
thresh = cv2.adaptiveThreshold(sobel_image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image2.jpg'
cv2.imwrite(output_image_path, thresh)

print("Обработанное изображение успешно сохранено.")

# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение адаптивного порогового преобразования для усиления контрастности текста
thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image3.jpg'
cv2.imwrite(output_image_path, thresh)

print("Обработанное изображение успешно сохранено.")



# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image3.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение гистограммного выравнивания для усиления контрастности
equalized_image = cv2.equalizeHist(gray_image)

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image4.jpg'
cv2.imwrite(output_image_path, equalized_image)

print("Обработанное изображение успешно сохранено.")

import numpy as np

# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image3.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение адаптивной бинаризации для выделения текста
binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Применение морфологического преобразования для улучшения выделения текста
kernel = np.ones((2, 2), np.uint8)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image5.jpg'
cv2.imwrite(output_image_path, opened_image)

print("Обработанное изображение успешно сохранено.")

import cv2
import numpy as np

# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image3.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение бинаризации для выделения текста
_, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Применение морфологического преобразования для улучшения выделения текста
kernel = np.ones((2, 3), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image6.jpg'
cv2.imwrite(output_image_path, eroded_image)

print("Обработанное изображение успешно сохранено.")

import cv2

# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение пороговой бинаризации для сегментации
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Сохранение сегментированного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image7.jpg'
cv2.imwrite(output_image_path, binary_image)

print("Сегментированное изображение успешно сохранено.")



image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image3.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Загрузка изображения в оттенках серого

def transform_image(img):
    # Преобразование изображения в двоичное
    _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Вычисление евклидова расстояния
    b_dist = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_L2, maskSize=5)
    g_dist = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_L1, maskSize=5)
    r_dist = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_C, maskSize=5)
    img_dist = cv2.merge((b_dist, g_dist, r_dist))  # Объединение каналов обратно в трехканальное изображение
    return img_dist

# Преобразование изображения в двоичное изображение
transformed_image = transform_image(image)

output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image8.jpg'
cv2.imwrite(output_image_path, transformed_image)

print("Сегментированное изображение успешно сохранено.")

import cv2

image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

_, bimage = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

b = cv2.distanceTransform(bimage, distanceType=cv2.DIST_L2, maskSize=5)
g = cv2.distanceTransform(bimage, distanceType=cv2.DIST_L1, maskSize=5)
r = cv2.distanceTransform(bimage, distanceType=cv2.DIST_C, maskSize=5)
bimage = cv2.merge((b, g, r))

output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image9.jpg'
cv2.imwrite(output_image_path, bimage)

print("Сегментированное изображение успешно сохранено.")


import numpy as np

# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Увеличение контрастности с помощью адаптивной гистограммной эквализации
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# Уменьшение шума с помощью медианного фильтра
denoised_image = cv2.medianBlur(clahe_image, 5)

# Применение морфологического преобразования для улучшения выделения текста
kernel = np.ones((3, 3), np.uint8)
opened_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)

# Увеличение резкости изображения
sharpness_image = cv2.filter2D(opened_image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image10.jpg'
cv2.imwrite(output_image_path, sharpness_image)

print("Обработанное изображение успешно сохранено.")



# Загрузка изображения документа
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Увеличение контрастности с помощью адаптивной гистограммной эквализации
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# Увеличение яркости и контрастности для темных областей (заголовков)
bright_image = cv2.addWeighted(clahe_image, 1.5, np.zeros_like(clahe_image), 0, 0)

# Аугментация данных: поворот изображения на случайный угол
rows, cols = bright_image.shape
angle = np.random.randint(-10, 10)
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
augmented_image = cv2.warpAffine(bright_image, rotation_matrix, (cols, rows))

# Применение морфологического преобразования для улучшения выделения текста
kernel = np.ones((3, 3), np.uint8)
opened_image = cv2.morphologyEx(augmented_image, cv2.MORPH_OPEN, kernel)

# Сохранение обработанного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image11.jpg'
cv2.imwrite(output_image_path, opened_image)

print("Обработанное изображение успешно сохранено.")

import cv2


def image_transformation(image):
    # Преобразование изображения в двоичное (черно-белое)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Вычисление евклидова расстояния
    b = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_L2, maskSize=5)

    # Вычисление линейного расстояния
    g = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_L1, maskSize=5)

    # Вычисление максимального расстояния
    r = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_C, maskSize=5)

    # Объединение каналов
    P = cv2.merge((b, g, r))

    return P


image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image3.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

_, transformed_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
smoothed = cv2.GaussianBlur(transformed_image, (3, 3), 0)
# Сохранение преобразованного изображения
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image12.jpg'
cv2.imwrite(output_image_path, smoothed)

print("Преобразованное изображение успешно сохранено.")

def preprocess_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Перевод изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения с использованием адаптивного порога
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Морфологические преобразования для устранения шума
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Выделение границ с помощью оператора Собеля
    sobel_x = cv2.Sobel(opening, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(opening, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Нормализация результата выделения границ
    sobel_normalized = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)

    return sobel_normalized

preprocessed_image = preprocess_image('C:/Users/vitya/PycharmProjects/syg2/data/images/image3.jpg')

output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image13.jpg'
cv2.imwrite(output_image_path, preprocessed_image)


def enhance_dark_headers(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Перевод изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # для улучшения локального контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Адаптивная пороговая обработка для выделения темных областей
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    return binary

enhanced_image = enhance_dark_headers('C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg')

output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image14.jpg'
cv2.imwrite(output_image_path, enhanced_image)

def enhance_dark_headers(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Перевод изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # для улучшения локального контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Адаптивная пороговая обработка для выделения темных областей
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    return binary

enhanced_image = enhance_dark_headers('C:/Users/vitya/PycharmProjects/syg2/data/images/image2.jpg')

output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image15.jpg'
cv2.imwrite(output_image_path, enhanced_image)



class ImageBinarizer:
    def __init__(self):
        self.BLOCK_SIZE = 40
        self.DELTA = 25

    def _adjust_gamma(self, image: np.ndarray, gamma: float = 1.2):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.medianBlur(image, 3)
        return 255 - image

    def _postprocess(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image

    def _get_block_index(self, image_shape, yx, block_size):
        y = np.arange(max(0, yx[0] - block_size), min(image_shape[0], yx[0] + block_size))
        x = np.arange(max(0, yx[1] - block_size), min(image_shape[1], yx[1] + block_size))
        return np.meshgrid(y, x)

    def _adaptive_median_threshold(self, img_in: np.ndarray) -> np.ndarray:
        med = np.median(img_in)
        img_out = np.zeros_like(img_in)
        img_out[img_in - med < self.DELTA] = 255
        kernel = np.ones((3, 3), np.uint8)
        img_out = 255 - cv2.dilate(255 - img_out, kernel, iterations=2)
        return img_out

    def _block_image_process(self, image, block_size):
        out_image = np.zeros_like(image)
        for row in range(0, image.shape[0], block_size):
            for col in range(0, image.shape[1], block_size):
                idx = (row, col)
                block_idx = self._get_block_index(image.shape, idx, block_size)
                out_image[tuple(block_idx)] = self._adaptive_median_threshold(image[tuple(block_idx)])
        return out_image

    def _get_mask(self, img: np.ndarray) -> np.ndarray:
        image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_in = self._preprocess(image_in)
        image_out = self._block_image_process(image_in, self.BLOCK_SIZE)
        image_out = self._postprocess(image_out)
        return image_out

    def _sigmoid(self, x, orig, rad):
        k = np.exp((x - orig) * 5 / rad)
        return k / (k + 1.)

    def _combine_block(self, img_in, mask):
        img_out = np.zeros_like(img_in)
        img_out[mask == 255] = 255
        fig_in = img_in.astype(np.float32)

        idx = np.where(mask == 0)
        if idx[0].shape[0] == 0:
            img_out[idx] = img_in[idx]
            return img_out

        lo = fig_in[idx].min()
        hi = fig_in[idx].max()
        v = fig_in[idx] - lo
        r = hi - lo

        img_in_idx = img_in[idx]
        ret3, th3 = cv2.threshold(img_in[idx], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.alltrue(th3[:, 0] != 255):
            img_out[idx] = img_in[idx]
            return img_out

        bound_value = np.min(img_in_idx[th3[:, 0] == 255])
        bound_value = (bound_value - lo) / (r + 1e-5)
        f = (v / (r + 1e-5))
        f = self._sigmoid(f, bound_value + 0.05, 0.2)

        img_out[idx] = (255. * f).astype(np.uint8)
        return img_out

    def _combine_block_image_process(self, image, mask, block_size):
        out_image = np.zeros_like(image)
        for row in range(0, image.shape[0], block_size):
            for col in range(0, image.shape[1], block_size):
                idx = (row, col)
                block_idx = self._get_block_index(image.shape, idx, block_size)
                out_image[tuple(block_idx)] = self._combine_block(
                    image[tuple(block_idx)], mask[tuple(block_idx)])
        return out_image

    def binarize(self, img: np.ndarray, block_size: int = 20) -> np.ndarray:
        img = self._adjust_gamma(img)
        mask = self._get_mask(img)
        image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_out = self._combine_block_image_process(image_in, mask, block_size)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)
        return image_out



class ValleyEmphasisBinarizer:
    def __init__(self, N: int = 15) -> None:
        self.N = N

    def binarize(self, image: np.ndarray) -> np.ndarray:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = self.get_threshold(gray_img)

        gray_img[gray_img <= threshold] = 50
        gray_img[gray_img > threshold] = 255
        return gray_img

    def get_threshold(self, gray_img: np.ndarray) -> int:
        c, x = np.histogram(gray_img, bins=255)
        h, w = gray_img.shape
        total = h * w

        sum_val = 0
        for t in range(255):
            sum_val = sum_val + (t * c[t] / total)

        var_max = 0
        threshold = 0

        omega_1 = 0
        mu_k = 0

        for t in range(254):
            omega_1 = omega_1 + c[t] / total
            omega_2 = 1 - omega_1
            mu_k = mu_k + t * (c[t] / total)
            mu_1 = mu_k / omega_1
            mu_2 = (sum_val - mu_k) / omega_2
            sum_of_neighbors = np.sum(c[max(1, t - self.N):min(255, t + self.N)])
            denom = total
            current_var = (1 - sum_of_neighbors / denom) * (omega_1 * mu_1 ** 2 + omega_2 * mu_2 ** 2)
            # Check if new maximum found
            if current_var > var_max:
                var_max = current_var
                threshold = t

        return threshold


# Создание экземпляра класса ImageBinarizer
binarizer = ValleyEmphasisBinarizer()

# Загрузка изображения
image_path = 'C:/Users/vitya/PycharmProjects/syg2/data/images/image3.jpg'  # Укажите путь к вашему изображению
image = cv2.imread(image_path)

# Вызов метода binarize
binarized_image = binarizer.binarize(image)

# Сохранение результата
output_image_path = 'C:/Users/vitya/PycharmProjects/syg2/image16.jpg'  # Укажите путь для сохранения изображения
cv2.imwrite(output_image_path, binarized_image)