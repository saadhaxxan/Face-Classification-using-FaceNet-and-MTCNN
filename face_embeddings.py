import numpy as np
from keras.models import  load_model

# load the extracted faces dataset
data = np.load('5-celebrity-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)


model = load_model('facenet_keras.h5')
print('Loaded Model')
model.summary()
# This specific implementation of the FaceNet model expects that the pixel values are standardized
# So, Standardizing the inputs
# scale pixel values
def get_embedding(model,face_pixels):
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # To make predictions with FaceNet input vector needs to be in one sample 
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)

    # using model predictions to get face embedding for the images
    yhat = model.predict(samples)
    embedding = yhat[0]
    return embedding

newtrainX = list()
newtestX = list()
for face_pixels in trainX:
    embedding = get_embedding(model,face_pixels)
    newtrainX.append(embedding)
newtrainX = np.asarray(newtrainX)

for face_pixels in testX:
    embedding = get_embedding(model,face_pixels)
    newtestX.append(embedding)
newtestX = np.asarray(newtestX)

np.savez_compressed('faces-embeddings.npz',newtrainX,trainy,newtestX,testy)
