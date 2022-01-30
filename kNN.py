import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors

data = {'X': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 0.5, 1.5, 2.5, 2.5, 3.0, 3.5, 4.0, 5.0],
        'Y': [3.0, 4.25, 2.0, 2.75, 1.65, 2.7, 1.0, 2.5, 2.1, 2.75, 1.75, 1.5, 4.0, 2.1, 1.5, 1.85, 3.5, 1.45],
        'color': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]}

# Create DataFrame
data_df = pd.DataFrame(data)

# ax1 = data_df.plot.scatter(x='X',
#                            y='Y',
#                            c='color')
# plt.show()

n_neighbors = 3

X = data_df.iloc[:, 0:2]
y = data_df['color']

# print the shape of the data to
# better understand it
print('X.shape:', X.shape)
print('y.shape', y.shape)

# Create color maps
cmap_light = ListedColormap(["#e9967a", "#c9e5ee"])
cmap_bold = ListedColormap(["red", "blue"])

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)
h = 0.02  # step size in the mesh

# Plot the decision boundary
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("K-NN Classifier")

plt.show()
