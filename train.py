from sklearn.linear_model import LinearRegression
import pickle
from imutils import paths
import cv2
import numpy as np
import ntpath

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

#load the dataset
with open("pictures/no_head/coordinates.txt", "rb") as fp:   # Unpickling
    coordinates = pickle.load(fp)
print(coordinates)

coordinates2 = []
for i, coor in coordinates:
    coordinates2.append((i-1, coor))
coordinates = coordinates2
print(coordinates2)
path = "pictures/no_head"
imagePaths = list(paths.list_images(path))
print(imagePaths)

def customComparator(e):
    return e[0]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

paired_paths = []
for left, right in grouped(imagePaths, 2):
    file_name = path_leaf(left)
    i = int(file_name.split("_")[0])
    paired_paths.append((i,(left, right)))

paired_paths = sorted(paired_paths, key=customComparator)
print(paired_paths)

X = []
y_x = []
y_y = []
for (i, (left, right)), (j, (x, y)) in zip(paired_paths, coordinates):
    if (i != j):
        print("Error i={} j={}".format(i,j))
        exit(1)
    limg = cv2.imread(left)
    limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    limg = np.array(limg)
    limg = limg.flatten()
    #print(limg.shape)

    rimg = cv2.imread(right)
    rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    rimg = np.array(rimg)
    rimg = rimg.flatten()
    #print(rimg.shape)

    img = np.concatenate((limg, rimg))
    X.append(img)
    y_x.append(x)
    y_y.append(y)
    #print(img.shape)

#train--------------x-----------------
modelx = LinearRegression()
modelx.fit(X, y_x)

print(modelx.score(X, y_x))
print(modelx.coef_)

# save the model to disk
filename = 'models/first_modelx.sav'
pickle.dump(modelx, open(filename, 'wb'))

#train--------------y-----------------
modely = LinearRegression()
modely.fit(X, y_y)

print(modely.score(X, y_y))
print(modely.coef_)

# save the model to disk
filename = 'models/first_modely.sav'
pickle.dump(modely, open(filename, 'wb'))

'''
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
'''