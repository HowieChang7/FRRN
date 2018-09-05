import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import frrn1

model = frrn1.FRRN_A()

image_size = 256
batch_size = 10
classes = [0., 38., 75.]
n_label = 2+1


labelencoder = LabelEncoder()
labelencoder.fit(classes)

df_test = pd.read_csv('input1/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

def predict():
    # load the trained convolutional neural network
    print("loading network...")
    model.load_weights(filepath='weights/best_weights.hdf5')
    for start in tqdm(range(0, len(ids_test), batch_size)):
        x_batch = []
        y_batch = []
        batch = 0
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            batch += 1
            img = cv2.imread('input1/test/{}.jpg'.format(id))
            img = np.array(img, dtype="float") / 255.0
            x_batch.append(img)
            label = cv2.imread("input1/test_masks/{}_mask.png".format(id), cv2.IMREAD_GRAYSCALE)
            label = img_to_array(label).reshape((image_size * image_size,))
            y_batch.append(label)
            if batch % batch_size == 0:
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch).flatten()
                y_batch = labelencoder.transform(y_batch)
                y_batch = to_categorical(y_batch, num_classes=n_label)
                y_batch = y_batch.reshape((batch_size, image_size * image_size, n_label))
        # predict
        preds = model.predict(x_batch, verbose=2)
        for pred in preds:
            pred = np.argmax(pred, axis=-1)
            pred = labelencoder.inverse_transform(pred)
            pred = pred.reshape((256, 256)).astype(np.uint8)
            plt.imshow(pred)
            plt.title("Predict Images")
            plt.show()

        loss_and_metrics = model.evaluate(x_batch, y_batch, batch_size=10, verbose=1)
        print(loss_and_metrics)

if __name__ == '__main__':
    predict()



