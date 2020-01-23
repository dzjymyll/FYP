import tensorflow as tf
import cv2
import glob
import numpy as np
import keras
import os
import input
from numpy import linalg as LA
from matplotlib import pyplot as plt




(x_train, x_test,y_train, y_test) = input.input_data()

#(x_train, y_train) = input.input_data()

print('Step 1')
sum_image = np.zeros((100,100),dtype=np.float).flatten()

#x_train = x_train[:10]
for x in range(len(x_train)):
    sum_image = sum_image + x_train[x]

mean_face = np.divide(sum_image,float(len(x_train)))

faces=[]

for x in range(len(x_train)):
    m = x_train[x] - mean_face
    faces.append(m)

faces = np.array(faces)

trans_face = faces.transpose()
print('Step 2')
print(trans_face.shape)


#cov_matrix = np.tensordot(faces,trans_face)

cov_matrix = np.cov(faces)
print(cov_matrix.shape)
cov_matrix = np.divide(cov_matrix,float(len(x_train)))
eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)

print(eigenvectors.shape)
print('Step 3')

print('Covariance matrix of X: \n%s' %cov_matrix)

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the eigen pairs in descending order:
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

'''
N1 = 0;
for i in range(100):
    N1 = N1 + eigvalues_sort[i]

N2 = 0;
for i in range(100,len(eigvalues_sort)):
    N2 = N2 + eigvalues_sort[i]

N = N1 /(N1 + N2)
print("weight of selected eigenvector: " + str(N))

'''

var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

# Show cumulative proportion of varaince with respect to components
print("Cumulative proportion of variance explained vector: \n%s" %var_comp_sum)

# x-axis for number of principal components kept
num_comp = range(1,len(eigvalues_sort)+1)
plt.title('Cum. Prop. Variance Explain and Components Kept')
plt.xlabel('Principal Components')
plt.ylabel('Cum. Prop. Variance Explained')

plt.scatter(num_comp, var_comp_sum)
plt.show()


reduced_data = np.array(eigvectors_sort[:8]).transpose()
# reduced_data size: 7215 * 40

#proj_data = np.tensordot(faces,reduced_data,[(1,2), (0)])
print(faces.shape)
print(reduced_data.shape)

proj_data = np.dot(trans_face,reduced_data)
proj_data = np.array(proj_data,dtype=float)
# 10000 * 40
proj_data = proj_data.transpose()
# 40 * 10000
print(proj_data.shape)

for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(100,100)
    plt.subplot(2,4,1+i)
    plt.imshow(img, cmap='jet')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

w = np.array([np.dot(proj_data,i) for i in faces])

print(w.shape)





'''
test_faces=[]


for x in range(len(x_test)):
    m = x_test[x] - mean_face
    test_faces.append(m)

test_faces = np.array(test_faces)


#np.save('x_train.npy', w) # save
#np.save('y_train.npy', y_train)

#np.save('x_test.npy', test_faces) # save
#np.save('y_test.npy', y_test)
#np.save('proj_data.npy', proj_data)
#np.save('mean_face.npy', mean_face)


unknown = x_test[0]
print(unknown.shape)

unknown_mean = unknown - mean_face

w_unknown = np.dot(proj_data, unknown_mean)
# 40 * 1

diff  = w - w_unknown
norms = np.linalg.norm(diff, axis=1)
print(norms)
print(min(norms))
index =norms.argmin()

plt.subplot(1,2,1)

plt.imshow(unknown.reshape(100, 100), cmap='gray')
plt.title('Unknown face')

plt.subplot(1,2,2)
plt.imshow((faces[index]+mean_face).reshape(100, 100), cmap='gray')
plt.title('Found face')
plt.show()

print(y_test[0])
print(y_train[index])
if(y_test[0] == y_train[index]):
    print("Match!")
else:
    print("Ohhhhh false")
w_restore = np.dot(proj_data.transpose(), w_unknown)
w_restore = w_restore + mean_face

plt.subplot(1,2,1)

plt.imshow(unknown.reshape(100, 100), cmap='gray')
plt.title('Unknown face')

plt.subplot(1,2,2)
plt.imshow((w_restore).reshape(100, 100), cmap='gray')
plt.title('Reconstruct face')
plt.show()

'''


for file in glob.glob('data/test/*.jpg'):
    img=cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    unknown = np.array(img, dtype='float64').flatten()


print(type(unknown))


unknown_mean = unknown - mean_face

w_unknown = np.dot(proj_data, unknown_mean)
# 40 * 1

diff  = w - w_unknown
norms = np.linalg.norm(diff, axis=1)
print(norms)
print(min(norms))
index =norms.argmin()

plt.subplot(1,2,1)

plt.imshow(unknown.reshape(100, 100), cmap='gray')
plt.title('Unknown face')

plt.subplot(1,2,2)
plt.imshow((faces[index]+mean_face).reshape(100, 100), cmap='gray')
plt.title('Found face')
plt.show()

#print(y_test[0])
print(y_train[index])

if('PINS/pins_Aaron Paul' == y_train[index]):
    print("Match!")
else:
    print("Ohhhhh false")
w_restore = np.dot(proj_data.transpose(), w_unknown)
w_restore = w_restore + mean_face

plt.subplot(1,2,1)

plt.imshow(unknown.reshape(100, 100), cmap='gray')
plt.title('Unknown face')

plt.subplot(1,2,2)
plt.imshow((w_restore).reshape(100, 100), cmap='gray')
plt.title('Reconstruct face')
plt.show()





