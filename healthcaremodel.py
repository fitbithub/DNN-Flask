import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array


with open('resnet-50-MRI.json', 'r') as json_file:
    json_savedModel= json_file.read()
# load the model  
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('weights.hdf5')



def resnetPrediction(file_path):

    '''
    import cv2
    
    img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)
    
    # resize image
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    
    cv2.imshow("Resized image", resized)
    '''
    img = load_img(file_path, target_size=(256,256))
    img_tensor = img_to_array(img)
    # img_tensor = img_tensor[np.newaxis, : , : ]
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /= 255.


    pred_prob = model(img_tensor)

    pred = str(np.argmax(pred_prob))
    
    if pred == "1":
        message = "Tumor Detected"
    else:
        message = "Tumor Not Detected"

    return message


