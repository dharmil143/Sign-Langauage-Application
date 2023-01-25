import cv2 as cv
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('C:\\Users\\admin\\Desktop\\ASL\\Saved_model.h5')
 

cam_port=0
cam= cv.VideoCapture(cam_port)

while(True):
   result, image = cam.read()
   #image = (image * 255).round().astype(np.uint8)
   resized_image= cv.resize(image, (32,32))
   img_batch=np.expand_dims(resized_image, axis=0)
   prediction = model.predict(img_batch)
   print(prediction) 
   if result:
      cv.imshow("test",resized_image)
      cv.waitKey(0)
      cv.destroyWindow("test")
   else:
      print("Error Detected")





