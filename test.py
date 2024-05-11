from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import cv2
import random

i = random.randint(1500, 1999)

model_path = "C:/Users/burak/.vscode/yapayZeka/catdog/cat_dog_test.h5"
img_path = "C:/Users/burak/.vscode/yapayZeka/catdog/test/cats/" + str(i) + ".jpg"

prediction_model = load_model(model_path)
test_img = load_img(img_path, target_size = (150, 150))

test_img = img_to_array(test_img)

test_img = np.expand_dims(test_img, axis = 0)

result = prediction_model.predict(test_img)

if result[0][0] == 1:
    label = "dog"

else:
    label = "cat"

print("This is a " + label)

test_img = cv2.imread(img_path)

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 0)
cv2.putText(test_img, label, (20, 40), font, 1.0, color, 3)

cv2.imshow("Prediction", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

