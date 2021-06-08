from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0, math.pi*2, .1)
y = np.sin(x)

ACTIVE_FUN = 'tanh'
BATCH_SIZE = 1

model = Sequential()
model.add(Dense(5, input_shape=(1,), activation=ACTIVE_FUN))
model.add(Dense(5, activation=ACTIVE_FUN))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])

model.fit(x, y, epochs=1000, batch_size=BATCH_SIZE, verbose=0)

scores = model.evaluate(x, y, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

y_pred = model.predict(x)

# Plot
plt.plot(x, y, color='blue', linewidth=1, markersize='1')
plt.plot(x, y_pred, color='green', linewidth=1, markersize='1')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

