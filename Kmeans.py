import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist


# Kmeans function
def Kmeans(X, k):
    C = np.random.choice(len(X), k, replace=False)
    # Randomly choosing Centroids
    centroids = X[C, :]  # Step 1

    # Compute Euclidean distance between data points and centroids
    distances = cdist(X, centroids, 'euclidean')  # Step 2
    points = np.array([np.argmin(i) for i in distances])  # Step 3

    centroids = []
    for C in range(k):
        # Centroids update
        tempC = X[points == C].mean(axis=0)
        centroids.append(tempC)

    centroids = np.vstack(centroids)  # Updated Centroids

    distances = cdist(X, centroids, 'euclidean')
    points = np.array([np.argmin(i) for i in distances])

    return points


# initialize list of lists
data = [[3.600, 79], [1.800, 54], [2.283, 62], [3.333, 74], [2.883, 55], [4.533, 85], [1.950, 51], [1.833, 54],
        [4.700, 88], [3.600, 85], [1.600, 52], [4.350, 85], [3.917, 84], [4.200, 78], [1.750, 62], [1.800, 51],
        [4.700, 83], [2.167, 52], [4.800, 84], [1.750, 47]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Duration', 'Wait'])
df = df.values
# Applying our function
for i in range(100):
    points = Kmeans(df, 3)

# Visualize the results

u_labels = np.unique(points)
print(u_labels)
for i in u_labels:
    plt.scatter(df[points == i, 0], df[points == i, 1], label=i)
plt.legend()
plt.show()
