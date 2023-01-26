import cv2 as cv
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('C:\\Users\\admin\\Desktop\\ASL\\Saved_model.h5')
 

cam_port=0
cam= cv.VideoCapture(cam_port)

while(True):
   result, image = cam.read()
   imagenew = cv.resize(image, (32, 32))
   imagenew=np.expand_dims(imagenew, axis=0)
   np_X = np.array(imagenew)
   normalised_X = np_X.astype('float32')/255.0
   prediction = model.predict(normalised_X)
   label = tf.math.argmax(prediction, axis=1)
   print(label) 
   if result:
      cv.imshow("test",image)
      cv.waitKey(0)
      cv.destroyWindow("test")
   else:
      print("Error Detected")





