import cv2
import os
from albumentations import MotionBlur, OpticalDistortion, RandomBrightnessContrast, Compose

# Определяем необходимые трансформации
blur_transform = MotionBlur(p=4.0)
optical_distortion_transform = OpticalDistortion(p=4.0)
brightness_contrast_transform = RandomBrightnessContrast(p=4.0)

# Композитная трансформация, включающая в себя только указанные
transforms = Compose([
    blur_transform,
    optical_distortion_transform,
    brightness_contrast_transform
])

def apply_transforms(image, transforms):
    augmented_images = []
    for _ in range(len(transforms.transforms)):
        augmented = transforms(image=image)
        augmented_images.append(augmented['image'])
    return augmented_images

input_folder = 'C:/Users/vitya/PycharmProjects/syg2/ybeforeAug'
output_folder = 'C:/Users/vitya/PycharmProjects/syg2/yafterAug'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            augmented_images = apply_transforms(image, transforms)
            for i, augmented_image in enumerate(augmented_images):
                output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
                cv2.imwrite(output_path, augmented_image)
                print(f'Saved augmented image: {output_path}')
        else:
            print(f'Failed to read image: {image_path}')