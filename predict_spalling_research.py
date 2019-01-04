import numpy, os
import keras
from keras import models
from skimage import data
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
#from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

y_test = []
for i in range(1, 251):
  y_test.append(1)
#  print(i)
print(len(y_test))
for i in range(251, 501):
  y_test.append(0)
#  print(i)
print(len(y_test))

y_pred= []

#model = load_model('/home/labuser/sid/research/spalling/data/ready_data_256_1/models_ga_2/0_____4-59-23-63-49-17.hdf5')
model = load_model('/home/labuser/sid/Spalling_with_93_percent/model.hdf5')
test_image_files = [f for f in os.listdir('/home/labuser/sid/research/spalling/data/ready_data_256_1/test/') if '.jpg' in f]
print("test:",len(test_image_files))

for i in range(1, len(test_image_files)):
    image = data.imread('/home/labuser/sid/research/spalling/data/ready_data_256_1/test/'+ str(i) +'.jpg')
    test_images = []
    test_images.append(image)
    test_images = numpy.asarray(test_images)
    test_images = test_images.astype('float32')
    test_images /= 255.0

    rounded_predictions=model.predict_classes(test_images)

    y_pred.insert(int(i),int(rounded_predictions))

    if rounded_predictions == 0:
      print (str(i) + '.jpg' + ' is predicted as spalling')
    if rounded_predictions == 1:
      print (str(i) + '.jpg' + ' is predicted as not a spalling')

#print(y_pred)

print('The confusion matrix is as below: ')
print(confusion_matrix(y_test, y_pred))

accu = accuracy_score(y_test, y_pred)
accu = accu * 100
accu = format(accu, '.2f')
print('The accuracy of prediction is ' + str(accu) + '%.')

avg_preci = precision_score(y_test, y_pred)
avg_preci = 100 * avg_preci
avg_preci = format(avg_preci, '.2f') 
print('The precision of prediction is: ' + str(avg_preci) + '%.')

recall = recall_score(y_test, y_pred)
recall = 100 * recall
recall = format(recall, '.2f') 
print('The recall of prediction is: ' + str(recall) + '%.')

f_measure = f1_score(y_test, y_pred)
f_measure = 100 * f_measure
f_measure = format(f_measure, '.2f') 
print('The f measure of prediction is: ' + str(avg_preci) + '%.')

#########################################################################
#Ploting the 2-class Precision-Recall curve
#########################################################################
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: Average Precision = '+str(avg_preci))
#plt.show()
##########################################################################
