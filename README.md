# neural-network
Python- neural network model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Generate random data
data = np.random.random((1000, 10))
Y = np.random.randint(2, size=(1000, 1))

# Build the model
model = Sequential()
model.add(Dense(32, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(data, Y, epochs=10, batch_size=32)

# Generate random test data with the same number of features
X_test = np.random.random((100, 10))

# Assuming Y_test is binary, change this accordingly if it's different
Y_test = np.random.randint(2, size=(100, 1))

# Evaluate the model
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
