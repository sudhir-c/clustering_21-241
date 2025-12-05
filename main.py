import numpy as np; 
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons 
from sklearn.datasets import make_circles 
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
from sklearn.datasets import make_blobs

colors = ["bo", "ro", "go", "co", "mo", "yo"]


def gaussianSimilarity(node1, node2, sigmaVal):
    xTerm = np.square(node2[1] - node1[1])
    yTerm = np.square(node2[0] - node1[0])
    negatedSquaredDistance = -(xTerm + yTerm)
    return np.exp(negatedSquaredDistance/(2*np.square(sigmaVal))); 

def dataToAdjacencyMatrix(data, sigmaVal):
    finalArray = np.array((1, 2, 3))
    matrix = []
    i = 0
    for node in data: 
        column = []
        for j in data: 
            column.append((gaussianSimilarity(node, j, sigmaVal)))
        if (i == 0):
            finalArray = column
        else: 
            finalArray = np.column_stack((finalArray, column))
        matrix.append(column)
        i = i + 1

    # adjacencyMatrix = np.array(finalArray)
    adjacencyMatrix = np.column_stack(matrix); 

    # Check symmetry
    assert(np.allclose(adjacencyMatrix, adjacencyMatrix.T))
    assert(np.all(adjacencyMatrix >= 0))
    print("Adjacency matrix created")
    return adjacencyMatrix

def createUnnormalizedLaplacianMatrix(rawData, sigmaVal):
    W  = dataToAdjacencyMatrix(rawData, sigmaVal)
    np.fill_diagonal(W, 0)
    S = np.diag(W.sum(axis=1))
    L = S - W
    assert(np.allclose(L, L.T))
    return L

def createNormalizedLaplacianMatrix(rawData, sigmaVal):
    I = np.eye(len(rawData))
    W  = dataToAdjacencyMatrix(rawData, sigmaVal)
    np.fill_diagonal(W, 0)
    S = np.diag(W.sum(axis=1))
    P = np.linalg.inv(sqrtm(S)); 
    L = I - (P@W)@P
    assert(np.allclose(L, L.T))
    print("Laplacian:")
    print(L)
    return L


def firstKEigenvectorsAsMatrix(numClusters, matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    U = np.matrix((1, 2, 3))
    for i in range(numClusters):
        vector = np.array(eigenvectors[:, i])
        if (i == 0):
            U = vector;
        else:
            U = np.column_stack((U, vector));
    return U; 

def visualizeUnlabeledData(originalDataPoints, title):
    for point in originalDataPoints:
        plt.plot(point[0], point[1], "kh")
    plt.title(title)
    plt.show(); 

def visualizeLabeledData(originalDataPoints, labels, title):
    i = 0;
    for label in labels:
        plt.plot(originalDataPoints[i][0], originalDataPoints[i][1], colors[label])
        i = i + 1;
    plt.title(title)
    plt.show(); 


# load datasets from scikit-learn
blob_centers = 6

moonsRawData, moonsRealLabels = make_moons(n_samples=300, noise=0.1, random_state=42)
circlesRawData, circlesRealLabels = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
blobsRawData, blobsRealLabels = make_blobs(n_samples = 300, centers=blob_centers, cluster_std=0.5, random_state=0)
print("data created")

# First, visualize the data 
visualizeUnlabeledData(moonsRawData, "Raw Moons Data")
visualizeUnlabeledData(circlesRawData, "Raw Circles Data")
visualizeUnlabeledData(blobsRawData, "Raw Circles Data")

# Analuze moons and circles using kMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(moonsRawData)
visualizeLabeledData(moonsRawData, kmeans.labels_, "KMeans on Raw Moons Data")

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(circlesRawData)
visualizeLabeledData(circlesRawData, kmeans.labels_, "KMeans on Raw Circles Data")

kmeans = KMeans(n_clusters=blob_centers, random_state=0, n_init="auto").fit(blobsRawData)
visualizeLabeledData(blobsRawData, kmeans.labels_, "KMeans on Blobs Data")


sigmaVals = [0.1, 1.0, 10.0]
for sigmaVal in sigmaVals:
    unnormalizedMoonsLaplacian = createUnnormalizedLaplacianMatrix(moonsRawData, sigmaVal)
    moonsUnormalizedCluster = firstKEigenvectorsAsMatrix(2, unnormalizedMoonsLaplacian)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(moonsUnormalizedCluster)
    visualizeLabeledData(moonsRawData, kmeans.labels_, f"UNNORMALIZED Laplacian Clustering on Raw Moons Data, sigma = {sigmaVal}")

    unnormalizedCirclesLaplacian = createUnnormalizedLaplacianMatrix(circlesRawData, sigmaVal)
    circlesUnormalizedCluster = firstKEigenvectorsAsMatrix(2, unnormalizedCirclesLaplacian)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(circlesUnormalizedCluster)
    visualizeLabeledData(circlesRawData, kmeans.labels_, f"UNNORMALIZED Laplacian Clustering on Raw Circles Data, sigma = {sigmaVal}") 

    unnormalizedBlobsLaplacian = createUnnormalizedLaplacianMatrix(blobsRawData, sigmaVal)
    blobsUnormalizedCluster = firstKEigenvectorsAsMatrix(blob_centers, unnormalizedBlobsLaplacian)
    kmeans = KMeans(n_clusters=blob_centers, random_state=0, n_init="auto").fit(blobsUnormalizedCluster)
    visualizeLabeledData(blobsRawData, kmeans.labels_, f"UNNORMALIZED Laplacian Clustering on Raw Blob Data, sigma = {sigmaVal}") 


    normalizedMoonsLaplacian = createNormalizedLaplacianMatrix(moonsRawData, sigmaVal)
    moonsNormalizedCluster = firstKEigenvectorsAsMatrix(2, normalizedMoonsLaplacian)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(moonsNormalizedCluster)
    visualizeLabeledData(moonsRawData, kmeans.labels_, f"NORMALIZED Laplacian Clustering on Raw Moons Data, sigma = {sigmaVal}")

    normalizedCirclesLaplacian = createNormalizedLaplacianMatrix(circlesRawData, sigmaVal)
    circlesNormalizedCluster = firstKEigenvectorsAsMatrix(2, normalizedCirclesLaplacian)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(circlesNormalizedCluster)
    visualizeLabeledData(circlesRawData, kmeans.labels_, f"NORMALIZED Laplacian Clustering on Raw Circles Data, sigma = {sigmaVal}") 

    normalizedBlobsLaplacian = createNormalizedLaplacianMatrix(blobsRawData, sigmaVal)
    blobsNormalizedCluster = firstKEigenvectorsAsMatrix(blob_centers, normalizedBlobsLaplacian)
    kmeans = KMeans(n_clusters=blob_centers, random_state=0, n_init="auto").fit(blobsNormalizedCluster)
    visualizeLabeledData(blobsRawData, kmeans.labels_, f"NORMALIZED Laplacian Clustering on Raw Blob Data, sigma = {sigmaVal}")



