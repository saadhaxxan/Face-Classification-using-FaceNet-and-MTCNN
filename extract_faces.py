# demonstrate face detection on 5 Celebrity Faces Dataset
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn import MTCNN
import numpy as np
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

#Uncomment this to show extracted faces in a plot
# # specify folder to plot
# folder = '5-celebrity-faces-dataset/train/ben_afflek/'
# i = 1
# # enumerate files
# for filename in listdir(folder):
# 	# path
# 	path = folder + filename
# 	# get face
# 	face = extract_face(path)
# 	print(i, face.shape)
# 	# plot
# 	pyplot.subplot(2, 7, i)
# 	pyplot.axis('off')
# 	pyplot.imshow(face)
# 	i += 1
# pyplot.show()

def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory+filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    X,Y = list(),list()
    for subdir in listdir(directory):
        path = directory+subdir+"/"
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        Y.extend(labels)
    return asarray(X),asarray(Y)


# load train dataset
trainX, trainy = load_dataset('dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('dataset/val/')
print(testX.shape, testy.shape)
# save arrays to one file in compressed format
np.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)