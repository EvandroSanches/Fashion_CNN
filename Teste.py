import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from PIL import Image

#Carrega Modelo
dir_imagens = os.listdir('Imagens')
modelo = load_model('Modelo.0.1')
class_names = ['T-Shirt/Top', 'Calça', 'Suéter', 'Vestidos', 'Casaco', 'Sandálias', 'Camisas', 'Tênis', 'Bolsa', 'Botas']

#Carrega e trata imagens da pasta Imagens
def carrega_imagens():
    np_image = np.empty((0,28,28), float)
    for arquivo in dir_imagens:
        imagens = (Image.open('Imagens/'+arquivo))
        new_array = np.array(imagens)
        new_array = new_array.reshape(new_array.shape[0],-1)
        new_array = cv2.resize(new_array, dsize=(28, 28), interpolation=cv2.COLOR_BGR2GRAY)
        new_array = cv2.bitwise_not(new_array)
        new_array = new_array / 255.0
        np_image = np.append(np_image, np.array([new_array]), axis=0)

    return np_image

#Cria resultado de teste através de dados graficos
def plot_image(i, predictions_array,img):
    predictions_array,img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label], 100*np.max(predictions_array),fontsize=6))




dados = carrega_imagens()
predictions = modelo.predict(dados)

#Cria resultado de teste através de dados graficos
num_rows = 5
num_cols = 5
num_images = len(predictions)
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  print(np.argmax(predictions[i]))
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, dados)
plt.show()


