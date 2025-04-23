import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Flatten, Dense





# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
image1=x_train[1]
plt.imshow(image1)
plt.show()
'''image2=x_train[2]
plt.imshow(image2)
plt.show()
image3=x_train[3]
plt.imshow(image3)
plt.show()'''

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(Flatten())
model.add(layers.Dense(100, activation ="relu"))
model.add(layers.Dense(50, activation ="relu"))
model.add(layers.Dense(10, activation ="softmax"))
model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss = "categorical_crossentropy",
optimizer = "adam",
metrics = ["accuracy",])
batch_size = 32
epochs = 10
history = model.fit(x_train_s,
y_train_s,
batch_size = batch_size,
epochs = epochs,
validation_split = 0.1)



# TODO: provedi ucenje mreze

predictions = model.predict(x_test_s)
score = model.evaluate(x_test_s, y_test_s, verbose=0)

# TODO: Prikazi test accuracy i matricu zabune

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test_s, axis=1)

print(f"y_pred shape: {y_pred.shape}")
print(f"y_true shape: {y_true.shape}")

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(num_classes))
disp.plot(cmap="Blues", xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.show()

print(f"Test accuracy: {score[1] * 100:.2f}%")

# TODO: spremi model

model.save("mnist_model.h5")
print("Model saved as 'mnist_model.h5'")