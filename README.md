from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np

# Generate random data
data = np.random.random((1000, 10))
Y = np.random.randint(2, size=(1000, 1))

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a lower learning rate
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Visualize the model architecture and save the image
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Train the model with a validation set
model.fit(data, Y, epochs=200, batch_size=64, validation_split=0.2)

# Generate random test data with the same number of features
X_test = np.random.random((100, 10))
Y_test = np.random.randint(2, size=(100, 1))

# Evaluate the model
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
