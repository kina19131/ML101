import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


load_data = load_breast_cancer()
data = load_data.data # data

def initialize_centroids(datas, k):
    centroids = datas.copy()  
    np.random.shuffle(centroids)
    return centroids[:k] #choose random point in datas as initial centroid


def closest_centroid_index(datas, centroids):
    distances = np.sqrt(((datas - centroids[:, np.newaxis])**2).sum(axis=2)) # finding euclidean distances 
    # [:, np.newaxis] creates column vector
    return np.argmin(distances, axis=0) #the arry of indexes with min distance

def move_centroids(datas, closest_cent, centroids):
    # Using the mean of the assigned points to define new centroid 
    return np.array([datas[closest_cent==k].mean(axis=0) for k in range(centroids.shape[0])])
    
def k_means(datas, centroids):
    # main function that utilizes the functions.
    # k_means which will be placed inside a while loop adjust and update centroids until find the optimal one. 
    return(move_centroids(data, closest_centroid_index(data, centroids), centroids))
    
def distoriton (centroids, closest_centroid_index, data):
    #distance between point and it's assigned centroid
    return ((centroids[closest_centroid_index] - data) ** 2).sum() / data.shape[0]


K = []
distortions = []
prev_centroids = np.empty([2, 30])

for k in range (2,8):
    K.append(k)
    centroids = initialize_centroids(data, k) #initialize_centroid
    while(True): #loop through until find the optimal centroid position
        centroids = k_means(data, centroids)
        if (np.array_equiv(centroids, prev_centroids,)):
            #break when there is no longer changes/updates in centroid 
            break
        else:
            prev_centroids = centroids #prev_cent assigned to check if there is any more updates happening to centroids 

    closest = closest_centroid_index(data, centroids) #out of the while loop, holds to which centroids each pts are assigned to
    distortions.append(distoriton(centroids, closest, data)) 

# Problem 1.4 Plotting the Distortion Graph 
plt.plot(K , distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
