import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train = X_train.reshape((60000, 28 * 28))/255.0
X_test = X_test.reshape((10000, 28 * 28))/255.0
model = Sequential()
#the in-put layer = 784 neurons ,,hidden-layer = 30 neurons ,,output-layer = 10
neurons
model.add(layers.Flatten(input_shape=(28*28,)))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
sgd = tf.keras.optimizers.SGD(learning_rate=1)
model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics
=['accuracy'])
model.fit(X_train,y_train,batch_size=60000,epochs=30)#Batch = 1, 20 ,60000
print('when the learning rate η is (0.1) and batch is (60000) - mini-batch
####' )
