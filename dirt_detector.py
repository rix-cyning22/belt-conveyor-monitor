from tensorflow.io import read_file, decode_jpeg
from tensorflow.image import resize, random_crop 
from tensorflow.keras.models import load_model
from random import uniform
import numpy as np

class DirtDetection:
    def __init__(self):
        self.model = load_model("dirt-detector.h5")
        
    def __preprocess_image(self, img_file):
        img = read_file(img_file)
        img = decode_jpeg(img)
        height, width, _ = img.shape
      
        k = uniform(0.65, 1)
        new_height = int(k * height)
        new_width = int(k * width)
    
        _image = random_crop(img, size=[new_height, new_width, 3])
        aspect_ratio = uniform(0.75, 1.33)
        new_height = int(new_width * aspect_ratio)
      
        _image = resize(_image,
                      size=(new_height, new_width))
        _image = resize(_image,
                      size=(256, 256))
        return np.reshape(_image, (1, 256, 256, 3))
    
    def detect(self, img_file):
        img = self.__preprocess_image(img_file)
        return self.model.predict(img)[0, 0]

detector = DirtDetection()
print(detector.detect("D:\\CV\Project\\WhatsApp Image 2023-10-06 at 1.57.13 PM.jpeg"))