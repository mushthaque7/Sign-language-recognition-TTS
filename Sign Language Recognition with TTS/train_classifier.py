import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./onehand.pickle', 'rb'))
data_dict2 = pickle.load(open('./twohand.pickle', 'rb'))
# data_dict3 = pickle.load(open('./onehand2.pickle', 'rb'))
data_one = np.asarray(data_dict['data'])
data_two = np.asarray(data_dict2['data'])
# data_three = np.asarray(data_dict3['data'])
labels1 = np.asarray(data_dict['labels'])
labels2 = np.asarray(data_dict2['labels'])
# labels3 = np.asarray(data_dict3['labels'])

x_train, x_test, y_train, y_test, = train_test_split(data_one, labels1, test_size=0.2, shuffle=True, stratify=labels1)
x_train2, x_test2, y_train2, y_test2, = train_test_split(data_two, labels2, test_size=0.2, shuffle=True, stratify=labels2)
# x_train3, x_test3, y_train3, y_test3, = train_test_split(data_three, labels3, test_size=0.2, shuffle=True, stratify=labels3)
model1 = RandomForestClassifier()
model2 = RandomForestClassifier()
# model3 = RandomForestClassifier()
model1.fit(x_train, y_train)
model2.fit(x_train2, y_train2)
# model3.fit(x_train3, y_train3)
y_predict = model1.predict(x_test)
y_predict2 = model2.predict(x_test2)
# y_predict3 = model3.predict(x_test3)
score1 = accuracy_score(y_predict, y_test)
score2= accuracy_score(y_predict2, y_test2)
# score3= accuracy_score(y_predict3, y_test3)
print('{}% of samples were classified correctly for model1 !'.format(score1 * 100))
print('{}% of samples were classified correctly for model2 !'.format(score2 * 100))
# print('{}% of samples were classified correctly for model2 !'.format(score3 * 100))
f = open('./model1.p', 'wb')
pickle.dump({'model1': model1}, f)
f.close()
f = open('./model2.p', 'wb')
pickle.dump({'model2': model2}, f)
# f = open('./model3.p', 'wb')
# pickle.dump({'model3': model3}, f)
f.close()
