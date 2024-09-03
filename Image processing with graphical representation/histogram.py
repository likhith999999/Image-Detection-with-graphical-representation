import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def getSize(person, chairObject, doc):
    heightPerson, widthPerson = person.shape[:2]
    heightChair, widthChair = chairObject.shape[:2]
    heightDoc, widthDoc = doc.shape[:2]
    print("Size of Images")
    print("Person(height, width): (", heightPerson, ",", widthPerson, ")", "\n\n")
    print("Chair Object(height, width): (", heightChair, ",", widthChair, ")", "\n\n")
    print("Document(height, width): (", heightDoc, ",", widthDoc, ")", "\n\n")

def sharpen(person, chairObject, doc):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return [cv2.filter2D(person, -1, kernel), cv2.filter2D(chairObject, -1, kernel), cv2.filter2D(doc, -1, kernel)]

def emboss(person, chairObject, doc):
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    return [cv2.filter2D(person, -1, kernel), cv2.filter2D(chairObject, -1, kernel), cv2.filter2D(doc, -1, kernel)]

def blur(person, chairObject, doc):
    return [cv2.GaussianBlur(person, (35, 35), 0), cv2.GaussianBlur(chairObject, (35, 35), 0), cv2.GaussianBlur(doc, (35, 35), 0)]

def grayScale(person, chairObject, doc):
    return [cv2.cvtColor(person, cv2.COLOR_BGR2GRAY), cv2.cvtColor(chairObject, cv2.COLOR_BGR2GRAY), cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)]

def negative(person, chairObject, doc):
    return [cv2.bitwise_not(person), cv2.bitwise_not(chairObject), cv2.bitwise_not(doc)]

def flipWarping(person, chairObject, doc):
    return [cv2.flip(person, 0), cv2.flip(chairObject, 0), cv2.flip(doc, 0)]

def mirrorWarping(person, chairObject, doc):
    return [cv2.flip(person, 1), cv2.flip(chairObject, 1), cv2.flip(doc, 1)]

def minFilter(person, chairObject, doc):
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (11, 11))
    return [cv2.erode(person, kernel), cv2.erode(chairObject, kernel), cv2.erode(doc, kernel)]

def maxFilter(person, chairObject, doc):
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (11, 11))
    return [cv2.dilate(person, kernel), cv2.dilate(chairObject, kernel), cv2.dilate(doc, kernel)]

def applyConcaveEffect(image):
    rows, columns = image.shape[:2]
    imageResult = np.zeros(image.shape, dtype=image.dtype)
    for i in range(rows):
        for j in range(columns):
            moveX = int(128.0 * math.sin(2 * 3.14 * i / (2*columns)))
            moveY = 0
            if j + moveX < columns:
                imageResult[i, j] = image[i, (j + moveX) % columns]
            else:
                imageResult[i, j] = 0
    return imageResult

def concaveEffect(person, chairObject, doc):
    return [applyConcaveEffect(person), applyConcaveEffect(chairObject), applyConcaveEffect(doc)]

def allFilters(person, chairObject, doc):
    sharpens = sharpen(person, chairObject, doc)
    embossing = emboss(person, chairObject, doc)
    gaussianBlur = blur(person, chairObject, doc)
    grayScales = grayScale(person, chairObject, doc)
    negatives = negative(person, chairObject, doc)
    flipWarpingImg = flipWarping(person, chairObject, doc)
    mirrorWarpingImg = mirrorWarping(person, chairObject, doc)
    minimumFilter = minFilter(person, chairObject, doc)
    maximumFilter = maxFilter(person, chairObject, doc)
    concaveFilterEffect = concaveEffect(person, chairObject, doc)

    return [sharpens, embossing, gaussianBlur, grayScales, negatives, flipWarpingImg, mirrorWarpingImg, minimumFilter, maximumFilter, concaveFilterEffect]

def plot_histogram(image, ax, title):
    if len(image.shape) == 3:
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='k')
    ax.set_xlim([0, 256])
    ax.set_title(f'{title} - Histogram')

def showFilters():
    person = cv2.imread('me.jpeg', 1)
    chairObject = cv2.imread('chair.jpeg', 1)
    doc = cv2.imread('doc.jpeg', 1)

    getSize(person, chairObject, doc)
    filters = allFilters(person, chairObject, doc)
    
    nameOfFilters = ["Sharpen", "Embossing", "GaussBlur", "GrayScale", "Negative", "FlipWarping", "MirrorWarping", "MinimumFilter", "MaximumFilter", "ConcaveFilterEffect"]
    nameOfImages = ["Person", "Chair", "Doc"]

    fig, axes = plt.subplots(len(nameOfFilters), len(nameOfImages), figsize=(15, 20))
    fig.tight_layout(pad=3.0)
    
    for i, filterSet in enumerate(filters):
        for j, img in enumerate(filterSet):
            ax_hist = axes[i, j]

            plot_histogram(img, ax_hist, f'{nameOfImages[j]} - {nameOfFilters[i]}')
    
    plt.show()

showFilters()
