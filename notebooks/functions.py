import numpy as np

import random

import matplotlib.pyplot as plt
from matplotlib import offsetbox

def plot_digits(X, digits, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    y = digits.target

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def plot_faces(X, faces, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)

	plt.figure(figsize=(10, 10))
	ax = plt.subplot(111)

	if hasattr(offsetbox, 'AnnotationBbox'):
		shown_images = np.array([[1., 1.]])  
		for i in range(faces.data.shape[0]):
			dist = np.sum((X[i] - shown_images) ** 2, 1)
			if np.min(dist) < 4e-5:
				continue
			shown_images = np.r_[shown_images, [X[i]]]		
			imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(faces.images[i], zoom=0.4, cmap=plt.cm.gray_r),
                X[i],
				frameon=False)
			ax.add_artist(imagebox)
	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)
	plt.show()

def pointCluster(numPoints):
	colors1 = []
	colors2 = []
	L1 = genCluster([-10, -10], 5, numPoints, 2)
	colors1 = ['b' for x in range(numPoints)]
	L2 = genCluster([10, 10], 5, numPoints, 2)
	colors2 = ['g' for x in range(numPoints)]
	return np.concatenate([L1, L2]), colors1 + colors2

def plot2D(X, colors):
    fig = plt.figure()
    ax = fig.add_subplot(212)
    ax.scatter(X[:, 0], X[:, 1], c = colors, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.show()

def genCluster(source, deviationFromPoint, numberOfPoints, dim):
	L = np.zeros((numberOfPoints, dim)) 

	for i in range(numberOfPoints):
		newCoords = [source[m] + random.random() * deviationFromPoint for m in range(dim)]
		for y in range(dim):
			L[i][y] = newCoords[y]
	return L
