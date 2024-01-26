import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def extractPixelFeatures(image):
    numRows, numCols, _ = image.shape
    rgbFeatures = np.zeros((numRows * numCols, 3))
    locationFeatures = np.zeros((numRows * numCols, 2))

    index = 0
    for row in range(numRows):
        for col in range(numCols):
            rgb = image[row, col] / 255.0
            location = np.array([row / numRows, col / numCols])
            rgbFeatures[index] = rgb
            locationFeatures[index] = location
            index += 1

    rgbLocationFeatures = np.hstack((rgbFeatures, locationFeatures))
    return rgbFeatures, rgbLocationFeatures

def KMeans(features, k):
    numSamples, numFeatures = features.shape
    centroids = features[np.random.choice(numSamples, k, replace=False)]

    for iteration in range(100):
        distances = np.zeros((k, numSamples))
        for i in range(k):
            for j in range(numSamples):
                distance = np.sqrt(np.sum((features[j] - centroids[i]) ** 2))
                distances[i, j] = distance

        closestCentroid = np.argmin(distances, axis=0)
        newCentroids = np.zeros((k, numFeatures))

        for i in range(k):
            clusterPoints = features[closestCentroid == i]
            if len(clusterPoints) > 0:
                newCentroids[i] = np.mean(clusterPoints, axis=0)
            else:
                newCentroids[i] = features[np.random.choice(numSamples, 1, replace=False)]

        if np.array_equal(centroids, newCentroids):
            break

        centroids = newCentroids

    return closestCentroid, centroids

def segmentImage(image, features, k):
    labels, centroids = KMeans(features, k)
    numRows, numCols, _ = image.shape
    segmentedImg = np.zeros_like(image)

    for i in range(numRows):
        for j in range(numCols):
            index = i * numCols + j
            label = labels[index]
            segmentedImg[i, j] = centroids[label][:3] * 255

    return segmentedImg

def displayResults(originalImage, segmentedImages, k, titles,imageName):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(segmentedImages) + 1, 1)
    plt.imshow(originalImage)
    plt.title("Original Image")
    for i, (segmentedImg, title) in enumerate(zip(segmentedImages, titles), start=2):
        plt.subplot(1, len(segmentedImages) + 1, i)
        plt.imshow(segmentedImg)
        plt.title(f"{title}, K={k}")
    plt.savefig(f"SegmentedRGBLocation_{k}{imageName}.jpg")
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def main():
    imagePaths = ['zebra.jpg']  
    kValues = [3, 7, 10, 15, 25, 40]  

    for imagePath in imagePaths:
        imageName = os.path.splitext(os.path.basename(imagePath))[0]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rgbFeatures, rgbLocationFeatures = extractPixelFeatures(image)

        for k in kValues:
            segmentedRgb = segmentImage(image, rgbFeatures, k)
            segmentedRgbLocation = segmentImage(image, rgbLocationFeatures, k)
            displayResults(image, [segmentedRgb, segmentedRgbLocation], k, ["Segmented RGB", "Segmented RGB & Location"],imageName)


if __name__ == "__main__":
    main()