import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib.pyplot as plt
import numpy as np
import random



fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-Shirt/Top', 'Calça', 'Suéter', 'Vestidos', 'Casaco', 'Sandálias', 'Camisas', 'Tênis', 'Bolsa', 'Botas']


train_images = train_images / 255.0
test_images = test_images / 255.0



"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np_img, cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""
def treina_modelo():
    modelo = keras.Sequential()
    modelo.add(Flatten(input_shape=(28,28)))
    modelo.add(Dense(units=250, activation='LeakyReLU'))
    modelo.add(Dense(units=200, activation='relu'))
    modelo.add(Dense(units=10, activation='softmax'))

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0015,
        decay_steps=6000,
        decay_rate=0.25
    )

    modelo.compile(optimizer=keras.optimizers.Adam(lr_scheduler), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    resultado = modelo.fit(train_images,train_labels, batch_size=100, epochs=22, validation_data=(test_images,test_labels))
    modelo.save('Modelo.0.1')
    return resultado

#Treinar Novo Modelo
resultado = treina_modelo()

modelo = keras.models.load_model('Modelo.0.1')

"""
#Criando grafico de log de treino
plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.title('Histórico de Treinamento')
plt.ylabel('Função de Custo')
plt.xlabel('Épocas de Treinamento')
plt.legend(['Erro Treino','Erro Teste'])
plt.show()
"""
predictions = modelo.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color, fontsize=6)



def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



num_rows = 10
num_cols = 10
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  r = random.randrange(1,10000)
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(r, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(r, predictions, test_labels)
plt.show()


