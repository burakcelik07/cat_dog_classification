from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.4))

model.add(layers.Dense(512, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

#model.summary()

model.compile(loss = "binary_crossentropy", optimizer = optimizers.RMSprop(learning_rate = 1e-4), metrics = ["acc"])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 45,
                                   width_shift_range = 0.3,
                                   height_shift_range = 0.3,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = "nearest")

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_directory = "C:\\Users\\burak\\.vscode\\yapayZeka\\catdog\\train"
validation_directory = "C:\\Users\\burak\\.vscode\\yapayZeka\\catdog\\validation"

train_generator = train_datagen.flow_from_directory(train_directory, 
                                                    target_size = (150, 150),
                                                    batch_size = 16,
                                                    class_mode = "binary")

validation_generator = train_datagen.flow_from_directory(validation_directory, 
                                                         target_size = (150, 150),
                                                         batch_size = 16,
                                                         class_mode = "binary")

history = model.fit(train_generator, epochs = 100, validation_data = validation_generator, validation_steps = 50, steps_per_epoch = 100)

plt.style.use("seaborn-darkgrid")
plt.figure

plt.plot(np.arange(0, 100), history.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 100), history.history["val_loss"], label = "val_loss")

plt.plot(np.arange(0, 100), history.history["acc"], label = "train_acc")
plt.plot(np.arange(0, 100), history.history["val_acc"], label = "val_acc")

plt.title("Training | Loss and Accuracy")
plt.xlabel("100 - epoch")
plt.ylabel("Loss and Accuracy")
plt.legend(loc="lower left")
plt.show()

model.save("cat_dog_test.h5")