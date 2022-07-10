from csv import writer
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, timezone

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import recall_score, accuracy_score, classification_report

from sklearn.model_selection import cross_val_score

#from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit

import random


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Для управления 3D-изображением (посредством мышки)
from sklearn import datasets
from sklearn import model_selection # for split
import keras # keras.utils.to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
#
def myLoss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))
#

def print_s(s):
    f_log = open('submission_gritsenko_(new)_log.txt','a')
    f_log.write(s+'\n')
    f_log.close()
    print(s)

f_log = open('submission_gritsenko_(new)_log.txt','w')
f_log.close()

"""
rand_val = 15
print("{:032b}".format(rand_val))
print(rand_val)
g_str = "{:032b}".format(rand_val)
g_str_revers = g_str[::-1]
print(g_str_revers)
print('-----------------------------------')
g = list(map(lambda x: True if int(x) == 1 else False,"{:032b}".format(rand_val)))
g.reverse()
m = 1
g[m-1] = not g[m-1]
if g[m-1] == True:
    #g_str_revers[m-1] = '1'
    #g_str_revers = g_str_revers[:(m-1)] + '1' + g_str_revers[(m-1)+1:]
    temp = list(g_str_revers)
    temp[m-1] = '1'
    g_str_revers = "".join(temp)
else:
    #g_str_revers[m-1] = '0'
    #g_str_revers = g_str_revers[:(m-1)] + '0' + g_str_revers[(m-1)+1:]
    temp = list(g_str_revers)
    temp[m-1] = '0'
    g_str_revers = "".join(temp)
print(g_str_revers)
rand_val = int(g_str_revers[::-1], 2)
print(rand_val)
print("{:032b}".format(rand_val))
print('-----------------------------------')
"""
#print("{:032b}".format(4294967295))

illness_list = ['Артериальная гипертензия', 'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда', 'Сердечная недостаточность', 'Прочие заболевания сердца']

clf1 = RandomForestClassifier(n_estimators=100, random_state=1)#GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)#LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=100, random_state=1)#BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)#RandomForestClassifier(n_estimators=100, random_state=1)
clf3 = RandomForestClassifier(n_estimators=100, random_state=1)#AdaBoostClassifier(n_estimators=100)#GaussianNB()
#clf4 = RandomForestClassifier(n_estimators=100, random_state=1)#AdaBoostClassifier(n_estimators=100)#GaussianNB()
#clf5 = HistGradientBoostingClassifier(max_iter=100)#RandomForestClassifier(n_estimators=100, random_state=1)#AdaBoostClassifier(n_estimators=100)#GaussianNB()


# Используемые методы:
# 1. SGD Classifier (SGD) - Линейный классификатор с SGD-обучением (stochastic gradient descent - стохастический градиентный спуск)
# 2. Support Vector Machines (SVM) - Метод опорных векторов (kernel = 'linear')
# 3. Random Forest Classifier (RF) - Случайный лес (используются деревья решений)
# 4. Gaussian process classification (GP) - Гауссовская классификация (основана на аппроксимации Лапласа)
# 5. AdaBoost (Adaptive Boosting) Classifier (AB) - Адаптивное усиление
# 6. Decision tree classifier (DT) - Дерево решений (http://scikit-learn.org/stable/modules/tree.html)
# 7. Logistic Regression (LR) - Логистическая регрессия
# 8. Gaussian Naive Bayes (NB) - Гауссовский наивный байесовский классификатор
# 9. Support Vector (SV) Classification - Метод опорных векторов http://scikit-learn.org/stable/modules/svm.html#svm-classification
# 10. MLP (Multi-layer Perceptron) Classifier (MLP) - Многослойный перцептрон
# 11. K-Nearest Neighbors (KNN) - Метод K-ближайших соседей
# 12. Quadratic Discriminant Analysis (QDA) - Квадратичный дискриминантный анализ
# 13. Linear Discriminant Analysis (LDA) - Линейный дискриминантный анализ

"""
if cross_val:
    classifiers.append(('SGD', SGDClassifier(max_iter = 1500, tol = 1e-4)))
classifiers.append(('SVL', SVC(kernel = 'linear', C = 0.025, probability = True))) # C - штраф в случае ошибки
classifiers.append(('RF', RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1)))
if cross_val:
    classifiers.append(('GP', GaussianProcessClassifier()))
classifiers.append(('AB', AdaBoostClassifier()))
classifiers.append(('DT', DecisionTreeClassifier()))
classifiers.append(('LR', LogisticRegression(solver = 'lbfgs', max_iter = 500, multi_class = 'auto')))
classifiers.append(('NB', GaussianNB()))
classifiers.append(('SVR', SVC(gamma = 2, C = 1.0))) # gamma - коэффициент ядра для 'rbf' - radial basis function, 'poly' and 'sigmoid'
classifiers.append(('MLP', MLPClassifier(alpha = 0.01, max_iter = 200, solver = 'lbfgs', tol = 0.001)))
classifiers.append(('KNN', KNeighborsClassifier(3)))
classifiers.append(('QDA', QuadraticDiscriminantAnalysis()))
classifiers.append(('LDA', LinearDiscriminantAnalysis()))
"""

#model = HistGradientBoostingClassifier(max_iter=100)
#model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
#model = AdaBoostClassifier(n_estimators=100)
#model = RandomForestClassifier(n_estimators=100)
#model = KNeighborsClassifier(n_neighbors=5)
#model = SVC()
#model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')



models = []
# Артериальная гипертензия
model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1500)#RandomForestClassifier(n_estimators=100, max_depth=100, max_features=30, bootstrap=False)#LogisticRegression(random_state=0)#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)#KNeighborsClassifier(n_neighbors=5)
models.append(model)
# ОНМК Острое нарушение мозгового кровообращения (ОНМК, инсульт)
model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1500)#RandomForestClassifier(n_estimators=100, max_depth=100, max_features=30, bootstrap=False)#LogisticRegression(random_state=0)#RandomForestClassifier(n_estimators=100)#LogisticRegression(random_state=0)#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)#SVC(gamma = 2, C = 1.0)#SGDClassifier(max_iter = 1500, tol = 1e-4)#HistGradientBoostingClassifier(max_iter=100)#RandomForestClassifier(n_estimators=100)
models.append(model)
# Стенокардия, ИБС, инфаркт миокарда
model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1500)#RandomForestClassifier(n_estimators=100, max_depth=100, max_features=30, bootstrap=False)#LogisticRegression(random_state=0)#RandomForestClassifier(n_estimators=100)#LogisticRegression(random_state=0)#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)
models.append(model)
# Сердечная недостаточность
model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1500)#RandomForestClassifier(n_estimators=100, max_depth=100, max_features=30, bootstrap=False)#LogisticRegression(random_state=0)#RandomForestClassifier(n_estimators=100)#LogisticRegression(random_state=0)#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)
models.append(model)
# Прочие заболевания сердца
model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1500)#RandomForestClassifier(n_estimators=100, max_depth=100, max_features=30, bootstrap=False)#LogisticRegression(random_state=0)#RandomForestClassifier(n_estimators=100)#LogisticRegression(random_state=0)#RandomForestClassifier(max_depth = 10, n_estimators = 10, max_features = 1)#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)
models.append(model)

drop_always = [
    'Пол (1 - М; 2 - Ж)',
    'Семья (0 - никогда не был(а) в браке; 1 - в разводе; 2 - гражданский брак / проживание с партнером; 3 - в браке в настоящее время; 4 - вдовец / вдова)',
    'Этнос (1 - европейская; 2 - другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан))',
    'Национальность (0 - Азербайджанцы; 1 - Армяне; 2 - Башкиры; 3 - Белорусы; 4 - Буряты;  5 - Казахи; 6 - Киргизы; 7 - Молдаване; 8 - Немцы; 9 - Русские; 10 - Таджики; 11 - Татары; 12 - Украинцы; 13 - Чуваши; 14 - Эстонцы)',
    'Религия (0 - Атеист / агностик; 1 - Христианство; 2 - Ислам)',
    'Образование (0 - начальная школа; 1 - средняя школа / закон.среднее / выше среднего; 2 - профессиональное училище; 3 - ВУЗ)',
    'Профессия (0 - ведение домашнего хозяйства; 1 - вооруженные силы; 2 - дипломированные специалисты; 3 - квалифицированные работники сельского хозяйства и рыболовного; 4 - низкоквалифицированные работники; 5 - операторы и монтажники установок и машинного оборудования; 6 - представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры; 7 - работники.  занятые в сфере обслуживания. торговые работники магазинов и рынков; 8 - ремесленники и представители других отраслей промышленности; 9 - служащие; 10 - техники и младшие специалисты)',
    'Статус Курения (0 - Никогда не курил(а); 1 - Бросил(а); 2 - Курит)',
    'Возраст курения (0 - 0; 1 - 1-09; 2 - 10-15; 3 - 16-19; 4 - 20-29; 5 - 30-39; 6 - 40-49; 7 - > 50)',
    'Сигарет в день (0 - 0; 1 - 1-5; 2 - 6-9; 3 - 10-19; 4 - 20-29; 5 - > 30)',
    'Частота пасс кур (0 - Нет; 1 - 1-2 раза в неделю; 2 - 2-3 раза в день; 3 - 3-6 раз в неделю; 4 - не менее 1 раза в день; 5 - 4 и более раз в день)',
    'Алкоголь (0 - никогда не употреблял; 1 - ранее употреблял; 2 - употребляю в настоящее время)',
    'Возраст алког (0 - 0; 1 - 1-09; 2 - 10-15; 3 - 16-19; 4 - 20-29; 5 - 30-39; 6 - 40-49; 7 - > 50)',
    'Время засыпания (0 - 20:00:00-21:59:59; 1 - 22:00:00-23:59:59; 2 - 00:00:00-01:59:59; 3 - 02:00:00-03:59:59) - time',
    'Время пробуждения (0 - 00:00:00-01:59:59; 1 - 02:00:00-03:59:59; 2 - 04:00:00-05:59:59; 3 - 06:00:00-07:59:59; 4 - 08:00:00-09:59:59; 5 - 10:00:00-11:59:59) - time',
    ]

base_columns = [
    'Пол (1 - М; 2 - Ж) - value',
    'Семья (0 - никогда не был(а) в браке; 1 - в разводе; 2 - гражданский брак / проживание с партнером; 3 - в браке в настоящее время; 4 - вдовец / вдова) - value',
    'Этнос (1 - европейская; 2 - другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)) - value',
    'Национальность (0 - Азербайджанцы; 1 - Армяне; 2 - Башкиры; 3 - Белорусы; 4 - Буряты;  5 - Казахи; 6 - Киргизы; 7 - Молдаване; 8 - Немцы; 9 - Русские; 10 - Таджики; 11 - Татары; 12 - Украинцы; 13 - Чуваши; 14 - Эстонцы) - value',
    'Религия (0 - Атеист / агностик; 1 - Христианство; 2 - Ислам) - value',
    'Образование (0 - начальная школа; 1 - средняя школа / закон.среднее / выше среднего; 2 - профессиональное училище; 3 - ВУЗ) - value',
    'Профессия (0 - ведение домашнего хозяйства; 1 - вооруженные силы; 2 - дипломированные специалисты; 3 - квалифицированные работники сельского хозяйства и рыболовного; 4 - низкоквалифицированные работники; 5 - операторы и монтажники установок и машинного оборудования; 6 - представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры; 7 - работники.  занятые в сфере обслуживания. торговые работники магазинов и рынков; 8 - ремесленники и представители других отраслей промышленности; 9 - служащие; 10 - техники и младшие специалисты) - value',
    'Вы работаете?',
    'Выход на пенсию',
    'Прекращение работы по болезни',
    'Сахарный диабет',
    'Гепатит',
    'Онкология',
    'Хроническое заболевание легких',
    'Бронжиальная астма',
    'Туберкулез легких',
    'ВИЧ/СПИД',
    'Регулярный прим лекарственных средств',
    'Травмы за год',
    'Переломы',
    'Статус Курения (0 - Никогда не курил(а); 1 - Бросил(а); 2 - Курит) - value',
    'Возраст курения (0 - 0; 1 - 1-09; 2 - 10-15; 3 - 16-19; 4 - 20-29; 5 - 30-39; 6 - 40-49; 7 - > 50) - value',
    'Сигарет в день (0 - 0; 1 - 1-5; 2 - 6-9; 3 - 10-19; 4 - 20-29; 5 - > 30) - value',
    'Пассивное курение',
    'Частота пасс кур (0 - Нет; 1 - 1-2 раза в неделю; 2 - 2-3 раза в день; 3 - 3-6 раз в неделю; 4 - не менее 1 раза в день; 5 - 4 и более раз в день) - value',
    'Алкоголь (0 - никогда не употреблял; 1 - ранее употреблял; 2 - употребляю в настоящее время) - value',
    'Возраст алког (0 - 0; 1 - 1-09; 2 - 10-15; 3 - 16-19; 4 - 20-29; 5 - 30-39; 6 - 40-49; 7 - > 50) - value',
    'Время засыпания (0 - 20:00:00-21:59:59; 1 - 22:00:00-23:59:59; 2 - 00:00:00-01:59:59; 3 - 02:00:00-03:59:59) - value',
    'Время пробуждения (0 - 00:00:00-01:59:59; 1 - 02:00:00-03:59:59; 2 - 04:00:00-05:59:59; 3 - 06:00:00-07:59:59; 4 - 08:00:00-09:59:59; 5 - 10:00:00-11:59:59) - value',
    'Сон после обеда',
    'Спорт, клубы',
    'Религия, клубы'
    ]

base_columns_cut = [
    'Пол',
    'Семья',
    'Этнос',
    'Национальность',
    'Религия',
    'Образование',
    'Профессия',
    'Вы работаете?',
    'Выход на пенсию',
    'Прекращение работы по болезни',
    'Сахарный диабет',
    'Гепатит',
    'Онкология',
    'Хроническое заболевание легких',
    'Бронжиальная астма',
    'Туберкулез легких',
    'ВИЧ/СПИД',
    'Регулярный прим лекарственных средств',
    'Травмы за год',
    'Переломы',
    'Статус Курения',
    'Возраст курения',
    'Сигарет в день',
    'Пассивное курение',
    'Частота пасс кур',
    'Алкоголь',
    'Возраст алког',
    'Время засыпания',
    'Время пробуждения',
    'Сон после обеда',
    'Спорт, клубы',
    'Религия, клубы'
    ]


df = pd.DataFrame(base_columns)
#print("{:032b}".format(7))
g = list(map(lambda x: True if int(x) == 1 else False,"{:032b}".format(7)))
g.reverse()

#print(g)
df_v = df[g].values
print(len(df_v))
#print(df_v[0][0])

all_columns = []
# Артериальная гипертензия
columns = [
    #'Пол (1 - М; 2 - Ж) - value',
    #'Семья (0 - никогда не был(а) в браке; 1 - в разводе; 2 - гражданский брак / проживание с партнером; 3 - в браке в настоящее время; 4 - вдовец / вдова) - value',
    'Этнос (1 - европейская; 2 - другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)) - value',
    'Национальность (0 - Азербайджанцы; 1 - Армяне; 2 - Башкиры; 3 - Белорусы; 4 - Буряты;  5 - Казахи; 6 - Киргизы; 7 - Молдаване; 8 - Немцы; 9 - Русские; 10 - Таджики; 11 - Татары; 12 - Украинцы; 13 - Чуваши; 14 - Эстонцы) - value',
    #'Религия (0 - Атеист / агностик; 1 - Христианство; 2 - Ислам) - value',
    'Образование (0 - начальная школа; 1 - средняя школа / закон.среднее / выше среднего; 2 - профессиональное училище; 3 - ВУЗ) - value',
    #'Профессия (0 - ведение домашнего хозяйства; 1 - вооруженные силы; 2 - дипломированные специалисты; 3 - квалифицированные работники сельского хозяйства и рыболовного; 4 - низкоквалифицированные работники; 5 - операторы и монтажники установок и машинного оборудования; 6 - представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры; 7 - работники.  занятые в сфере обслуживания. торговые работники магазинов и рынков; 8 - ремесленники и представители других отраслей промышленности; 9 - служащие; 10 - техники и младшие специалисты) - value',
    'Вы работаете?',
    'Выход на пенсию',
    'Прекращение работы по болезни',
    #'Сахарный диабет',
    #'Гепатит',
    #'Онкология',
    #'Хроническое заболевание легких',
    #'Бронжиальная астма',
    #'Туберкулез легких',
    'ВИЧ/СПИД',
    #'Регулярный прим лекарственных средств',
    'Травмы за год',
    #'Переломы',
    #'Статус Курения (0 - Никогда не курил(а); 1 - Бросил(а); 2 - Курит) - value',
    #'Возраст курения',
    'Сигарет в день',
    'Пассивное курение',
    'Частота пасс кур (0 - Нет; 1 - 1-2 раза в неделю; 2 - 2-3 раза в день; 3 - 3-6 раз в неделю; 4 - не менее 1 раза в день; 5 - 4 и более раз в день) - value',
    #'Алкоголь (0 - никогда не употреблял; 1 - ранее употреблял; 2 - употребляю в настоящее время) - value',
    #'Возраст алког',
    #'Время засыпания (0 - 20:00:00-21:59:59; 1 - 22:00:00-23:59:59; 2 - 00:00:00-01:59:59; 3 - 02:00:00-03:59:59) - value',
    #'Время пробуждения (0 - 00:00:00-01:59:59; 1 - 02:00:00-03:59:59; 2 - 04:00:00-05:59:59; 3 - 06:00:00-07:59:59; 4 - 08:00:00-09:59:59; 5 - 10:00:00-11:59:59) - value',
    'Сон после обеда',
    #'Спорт, клубы',
    #'Религия, клубы'
    ]
all_columns.append(columns)
# ОНМК Острое нарушение мозгового кровообращения (ОНМК, инсульт)
#columns = []
columns = [
    'Пол (1 - М; 2 - Ж) - value',
    #'Семья (0 - никогда не был(а) в браке; 1 - в разводе; 2 - гражданский брак / проживание с партнером; 3 - в браке в настоящее время; 4 - вдовец / вдова) - value',
    'Этнос (1 - европейская; 2 - другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)) - value',
    #'Национальность (0 - Азербайджанцы; 1 - Армяне; 2 - Башкиры; 3 - Белорусы; 4 - Буряты;  5 - Казахи; 6 - Киргизы; 7 - Молдаване; 8 - Немцы; 9 - Русские; 10 - Таджики; 11 - Татары; 12 - Украинцы; 13 - Чуваши; 14 - Эстонцы) - value',
    #'Религия (0 - Атеист / агностик; 1 - Христианство; 2 - Ислам) - value',
    #'Образование (0 - начальная школа; 1 - средняя школа / закон.среднее / выше среднего; 2 - профессиональное училище; 3 - ВУЗ) - value',
    #'Профессия (0 - ведение домашнего хозяйства; 1 - вооруженные силы; 2 - дипломированные специалисты; 3 - квалифицированные работники сельского хозяйства и рыболовного; 4 - низкоквалифицированные работники; 5 - операторы и монтажники установок и машинного оборудования; 6 - представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры; 7 - работники.  занятые в сфере обслуживания. торговые работники магазинов и рынков; 8 - ремесленники и представители других отраслей промышленности; 9 - служащие; 10 - техники и младшие специалисты) - value',
    #'Вы работаете?',
    'Выход на пенсию',
    'Прекращение работы по болезни',
    'Сахарный диабет',
    'Гепатит',
    #'Онкология',
    'Хроническое заболевание легких',
    'Бронжиальная астма',
    'Туберкулез легких',
    'ВИЧ/СПИД',
    #'Регулярный прим лекарственных средств',
    #'Травмы за год',
    'Переломы',
    #'Статус Курения (0 - Никогда не курил(а); 1 - Бросил(а); 2 - Курит) - value',
    #'Возраст курения',
    #'Сигарет в день',
    #'Пассивное курение',
    #'Частота пасс кур (0 - Нет; 1 - 1-2 раза в неделю; 2 - 2-3 раза в день; 3 - 3-6 раз в неделю; 4 - не менее 1 раза в день; 5 - 4 и более раз в день) - value',
    #'Алкоголь (0 - никогда не употреблял; 1 - ранее употреблял; 2 - употребляю в настоящее время) - value',
    #'Возраст алког',
    #'Время засыпания (0 - 20:00:00-21:59:59; 1 - 22:00:00-23:59:59; 2 - 00:00:00-01:59:59; 3 - 02:00:00-03:59:59) - value',
    #'Время пробуждения (0 - 00:00:00-01:59:59; 1 - 02:00:00-03:59:59; 2 - 04:00:00-05:59:59; 3 - 06:00:00-07:59:59; 4 - 08:00:00-09:59:59; 5 - 10:00:00-11:59:59) - value',
    #'Сон после обеда',
    #'Спорт, клубы',
    'Религия, клубы'
    ]
all_columns.append(columns)
# Стенокардия, ИБС, инфаркт миокарда
columns = [
    'Пол (1 - М; 2 - Ж) - value',
    #'Семья (0 - никогда не был(а) в браке; 1 - в разводе; 2 - гражданский брак / проживание с партнером; 3 - в браке в настоящее время; 4 - вдовец / вдова) - value',
    'Этнос (1 - европейская; 2 - другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)) - value',
    #'Национальность (0 - Азербайджанцы; 1 - Армяне; 2 - Башкиры; 3 - Белорусы; 4 - Буряты;  5 - Казахи; 6 - Киргизы; 7 - Молдаване; 8 - Немцы; 9 - Русские; 10 - Таджики; 11 - Татары; 12 - Украинцы; 13 - Чуваши; 14 - Эстонцы) - value',
    #'Религия (0 - Атеист / агностик; 1 - Христианство; 2 - Ислам) - value',
    #'Образование (0 - начальная школа; 1 - средняя школа / закон.среднее / выше среднего; 2 - профессиональное училище; 3 - ВУЗ) - value',
    #'Профессия (0 - ведение домашнего хозяйства; 1 - вооруженные силы; 2 - дипломированные специалисты; 3 - квалифицированные работники сельского хозяйства и рыболовного; 4 - низкоквалифицированные работники; 5 - операторы и монтажники установок и машинного оборудования; 6 - представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры; 7 - работники.  занятые в сфере обслуживания. торговые работники магазинов и рынков; 8 - ремесленники и представители других отраслей промышленности; 9 - служащие; 10 - техники и младшие специалисты) - value',
    #'Вы работаете?',
    'Выход на пенсию',
    'Прекращение работы по болезни',
    'Сахарный диабет',
    'Гепатит',
    #'Онкология',
    'Хроническое заболевание легких',
    'Бронжиальная астма',
    'Туберкулез легких',
    'ВИЧ/СПИД',
    #'Регулярный прим лекарственных средств',
    #'Травмы за год',
    'Переломы',
    #'Статус Курения (0 - Никогда не курил(а); 1 - Бросил(а); 2 - Курит) - value',
    #'Возраст курения',
    #'Сигарет в день',
    #'Пассивное курение',
    #'Частота пасс кур (0 - Нет; 1 - 1-2 раза в неделю; 2 - 2-3 раза в день; 3 - 3-6 раз в неделю; 4 - не менее 1 раза в день; 5 - 4 и более раз в день) - value',
    #'Алкоголь (0 - никогда не употреблял; 1 - ранее употреблял; 2 - употребляю в настоящее время) - value',
    #'Возраст алког',
    #'Время засыпания (0 - 20:00:00-21:59:59; 1 - 22:00:00-23:59:59; 2 - 00:00:00-01:59:59; 3 - 02:00:00-03:59:59) - value',
    #'Время пробуждения (0 - 00:00:00-01:59:59; 1 - 02:00:00-03:59:59; 2 - 04:00:00-05:59:59; 3 - 06:00:00-07:59:59; 4 - 08:00:00-09:59:59; 5 - 10:00:00-11:59:59) - value',
    #'Сон после обеда',
    #'Спорт, клубы',
    'Религия, клубы'
    ]
all_columns.append(columns)
# Сердечная недостаточность
columns = [
    'Пол (1 - М; 2 - Ж) - value',
    #'Семья (0 - никогда не был(а) в браке; 1 - в разводе; 2 - гражданский брак / проживание с партнером; 3 - в браке в настоящее время; 4 - вдовец / вдова) - value',
    'Этнос (1 - европейская; 2 - другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)) - value',
    #'Национальность (0 - Азербайджанцы; 1 - Армяне; 2 - Башкиры; 3 - Белорусы; 4 - Буряты;  5 - Казахи; 6 - Киргизы; 7 - Молдаване; 8 - Немцы; 9 - Русские; 10 - Таджики; 11 - Татары; 12 - Украинцы; 13 - Чуваши; 14 - Эстонцы) - value',
    #'Религия (0 - Атеист / агностик; 1 - Христианство; 2 - Ислам) - value',
    #'Образование (0 - начальная школа; 1 - средняя школа / закон.среднее / выше среднего; 2 - профессиональное училище; 3 - ВУЗ) - value',
    #'Профессия (0 - ведение домашнего хозяйства; 1 - вооруженные силы; 2 - дипломированные специалисты; 3 - квалифицированные работники сельского хозяйства и рыболовного; 4 - низкоквалифицированные работники; 5 - операторы и монтажники установок и машинного оборудования; 6 - представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры; 7 - работники.  занятые в сфере обслуживания. торговые работники магазинов и рынков; 8 - ремесленники и представители других отраслей промышленности; 9 - служащие; 10 - техники и младшие специалисты) - value',
    #'Вы работаете?',
    'Выход на пенсию',
    'Прекращение работы по болезни',
    'Сахарный диабет',
    'Гепатит',
    #'Онкология',
    'Хроническое заболевание легких',
    'Бронжиальная астма',
    'Туберкулез легких',
    'ВИЧ/СПИД',
    #'Регулярный прим лекарственных средств',
    #'Травмы за год',
    'Переломы',
    #'Статус Курения (0 - Никогда не курил(а); 1 - Бросил(а); 2 - Курит) - value',
    #'Возраст курения',
    #'Сигарет в день',
    #'Пассивное курение',
    #'Частота пасс кур (0 - Нет; 1 - 1-2 раза в неделю; 2 - 2-3 раза в день; 3 - 3-6 раз в неделю; 4 - не менее 1 раза в день; 5 - 4 и более раз в день) - value',
    #'Алкоголь (0 - никогда не употреблял; 1 - ранее употреблял; 2 - употребляю в настоящее время) - value',
    #'Возраст алког',
    #'Время засыпания (0 - 20:00:00-21:59:59; 1 - 22:00:00-23:59:59; 2 - 00:00:00-01:59:59; 3 - 02:00:00-03:59:59) - value',
    #'Время пробуждения (0 - 00:00:00-01:59:59; 1 - 02:00:00-03:59:59; 2 - 04:00:00-05:59:59; 3 - 06:00:00-07:59:59; 4 - 08:00:00-09:59:59; 5 - 10:00:00-11:59:59) - value',
    #'Сон после обеда',
    #'Спорт, клубы',
    'Религия, клубы'
    ]
all_columns.append(columns)
# Прочие заболевания сердца
columns = [
    'Пол (1 - М; 2 - Ж) - value',
    #'Семья (0 - никогда не был(а) в браке; 1 - в разводе; 2 - гражданский брак / проживание с партнером; 3 - в браке в настоящее время; 4 - вдовец / вдова) - value',
    'Этнос (1 - европейская; 2 - другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)) - value',
    #'Национальность (0 - Азербайджанцы; 1 - Армяне; 2 - Башкиры; 3 - Белорусы; 4 - Буряты;  5 - Казахи; 6 - Киргизы; 7 - Молдаване; 8 - Немцы; 9 - Русские; 10 - Таджики; 11 - Татары; 12 - Украинцы; 13 - Чуваши; 14 - Эстонцы) - value',
    #'Религия (0 - Атеист / агностик; 1 - Христианство; 2 - Ислам) - value',
    #'Образование (0 - начальная школа; 1 - средняя школа / закон.среднее / выше среднего; 2 - профессиональное училище; 3 - ВУЗ) - value',
    #'Профессия (0 - ведение домашнего хозяйства; 1 - вооруженные силы; 2 - дипломированные специалисты; 3 - квалифицированные работники сельского хозяйства и рыболовного; 4 - низкоквалифицированные работники; 5 - операторы и монтажники установок и машинного оборудования; 6 - представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры; 7 - работники.  занятые в сфере обслуживания. торговые работники магазинов и рынков; 8 - ремесленники и представители других отраслей промышленности; 9 - служащие; 10 - техники и младшие специалисты) - value',
    'Вы работаете?',
    'Выход на пенсию',
    'Прекращение работы по болезни',
    'Сахарный диабет',
    'Гепатит',
    'Онкология',
    'Хроническое заболевание легких',
    'Бронжиальная астма',
    'Туберкулез легких',
    'ВИЧ/СПИД',
    #'Регулярный прим лекарственных средств',
    #'Травмы за год',
    'Переломы',
    #'Статус Курения (0 - Никогда не курил(а); 1 - Бросил(а); 2 - Курит) - value',
    #'Возраст курения',
    #'Сигарет в день',
    'Пассивное курение',
    #'Частота пасс кур (0 - Нет; 1 - 1-2 раза в неделю; 2 - 2-3 раза в день; 3 - 3-6 раз в неделю; 4 - не менее 1 раза в день; 5 - 4 и более раз в день) - value',
    'Алкоголь (0 - никогда не употреблял; 1 - ранее употреблял; 2 - употребляю в настоящее время) - value',
    #'Возраст алког',
    #'Время засыпания (0 - 20:00:00-21:59:59; 1 - 22:00:00-23:59:59; 2 - 00:00:00-01:59:59; 3 - 02:00:00-03:59:59) - value',
    #'Время пробуждения (0 - 00:00:00-01:59:59; 1 - 02:00:00-03:59:59; 2 - 04:00:00-05:59:59; 3 - 06:00:00-07:59:59; 4 - 08:00:00-09:59:59; 5 - 10:00:00-11:59:59) - value',
    #'Сон после обеда',
    #'Спорт, клубы',
    'Религия, клубы'
    ]
all_columns.append(columns)

"""
# Рабочее на 2022.06.29
data = []

good_level = []
good_level.append(0.70)
good_level.append(0.70)
good_level.append(0.70)
good_level.append(0.70)
good_level.append(0.60)

count_ch = 0

for i in range(len(illness_list)):
    print(f'Заболевание = {illness_list[i]}')
    max_score = 0;
    max_g = []
    rand_data = []
    good_rand_val = 0
    #k = 0
    #for k in range(4294967295):
    good_model = []
    while len(rand_data) < 4294967295:
        rand_val = random.randint(0, 4294967295)
        if rand_val not in rand_data:
            rand_data.append(rand_val)
            
            g = list(map(lambda x: True if int(x) == 1 else False,"{:032b}".format(rand_val+1)))
            g.reverse()
            df_v = df[g].values
            if i == 1:
                data_train = pd.read_excel("train_dataset_train/train_2(cut_ONMK).xls")
            elif i == 2:
                data_train = pd.read_excel("train_dataset_train/train_2(cut_Stenokardiya).xls")
            elif i == 3:
                data_train = pd.read_excel("train_dataset_train/train_2(cut_Serdechnaya).xls")
            elif i == 4:
                data_train = pd.read_excel("train_dataset_train/train_2(cut_Prochee).xls")
            else:
                data_train = pd.read_excel("train_dataset_train/train_2(cut).xls")
            Xd = data_train.drop(['ID', 'ID_y',
                                 'Артериальная гипертензия',
                                 'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда',
                                 'Сердечная недостаточность', 'Прочие заболевания сердца',
                                 ], axis=1)
            Xd = Xd.drop(drop_always, axis=1)
            X = Xd.drop(df_v[0], axis=1)

            # data pre-processing
            X = preprocessing.StandardScaler().fit_transform(X)
            y = data_train[illness_list[i]]

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.70, test_size = 0.30)
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train,y_train)
            predict_y = model.predict(X_test)
            score = recall_score(y_test, predict_y, average='macro')
            if score > max_score:
                max_score = score
                good_rand_val = rand_val
                max_g = g
                max_list = df[max_g].values
                good_model = model
            if len(rand_data) % 100 == 0:
                print(f'Заболевание = {illness_list[i]}')
                print(g)
                print(f'{datetime.strftime(datetime.now(),"%d.%m.%Y_%H.%M.%S.%f")} - Перебрали вариантов = {len(rand_data)}, текущий вариант = {rand_val}, лучший вариант = {good_rand_val}, max_score = {max_score}')
                print(max_g)
                print('Не учитываются поля:')
                print(max_list)
                print('------------------------------------------------------------------------------------')
            #k += 1
            if score >= good_level[i]:
                print(f'Заболевание = {illness_list[i]}')
                print(f'Перебрали вариантов = {len(rand_data)}, текущий вариант = {rand_val}, лучший вариант = {good_rand_val}, max_score = {max_score}')
                print(max_g)
                print('Не учитываются поля:')
                print(max_list)

                df_v = df[max_g].values

                data_test = pd.read_excel("test_1.xls")
                X_final_test = data_test.drop(['ID'], axis=1)
                X_final_test = X_final_test.drop(drop_always, axis=1)
                X_final_test = X_final_test.drop(df_v[0], axis=1)
                X_final_ID = data_test['ID']
                # data pre-processing
                X_final_test = preprocessing.StandardScaler().fit_transform(X_final_test) #fit(X_final_test).transform(X_final_test)
    
                predict_y = good_model.predict(X_final_test)

                if i == 0:
                    data = np.column_stack((X_final_ID, predict_y))
                else:
                    data = np.column_stack((data, predict_y))

                print('------------------------------------------------------------------------------------')
                print(f'Конец расчёта заболевания = {illness_list[i]}')
                print('------------------------------------------------------------------------------------')
                
                break


fileName = 'submission_gritsenko.csv'
with open(fileName, "w", encoding = "utf8", newline = "") as csvFile:
  csvWriter = writer(csvFile, delimiter = ",")
  csvWriter.writerow(["ID", "Артериальная гипертензия", "ОНМК", "Стенокардия, ИБС, инфаркт миокарда", "Сердечная недостаточность", "Прочие заболевания сердца"])
  for row in data:
    csvWriter.writerow(row)
print('------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------')
"""






    

"""
results_score = 0
count_results_score = 0
for i in range(len(illness_list)):

    #print(all_columns[i])
    if i == 1:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_ONMK).xls")
    elif i == 2:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Stenokardiya).xls")
    elif i == 3:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Serdechnaya).xls")
    elif i == 4:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Prochee).xls")
    else:
        data_train = pd.read_excel("train_dataset_train/train_2(cut).xls")
    Xd = data_train.drop(['ID', 'ID_y',
                         'Артериальная гипертензия',
                         'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда',
                         'Сердечная недостаточность', 'Прочие заболевания сердца',
                         ], axis=1)
    Xd = Xd.drop(drop_always, axis=1)
    X = Xd.drop(all_columns[i], axis=1)

    #print(X)

    # data pre-processing
    X = preprocessing.StandardScaler().fit_transform(X) #fit(X).transform(X)

    
    
    y = data_train[illness_list[i]]
    

    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20)#, random_state = seed)
    #print(len(X_test))

    sss = StratifiedShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        models[i].fit(X_train,y_train)
        predict_y = models[i].predict(X_test)
    
        #score = accuracy_score(y_test,predict_y)
        score = recall_score(y_test, predict_y, average='macro')
        results_score += score
        count_results_score += 1
        print(f'score = {score}')
print(f'result score = {results_score/count_results_score}')
"""
"""
for i in range(len(illness_list)):
    #y = data_train[illness_list[i]]
    #models[i].fit(X,y)

    data_test = pd.read_excel("test_1.xls")
    X_final_test = data_test.drop(['ID'], axis=1)
    X_final_test = X_final_test.drop(drop_always, axis=1)
    X_final_test = X_final_test.drop(all_columns[i], axis=1)
    X_final_ID = data_test['ID']
    # data pre-processing
    X_final_test = preprocessing.StandardScaler().fit_transform(X_final_test) #fit(X_final_test).transform(X_final_test)
    
    predict_y = models[i].predict(X_final_test)

    if i == 0:
        data = np.column_stack((X_final_ID, predict_y))
    else:
        data = np.column_stack((data, predict_y))


fileName = 'submission_gritsenko.csv'
with open(fileName, "w", encoding = "utf8", newline = "") as csvFile:
  csvWriter = writer(csvFile, delimiter = ",")
  csvWriter.writerow(["ID", "Артериальная гипертензия", "ОНМК", "Стенокардия, ИБС, инфаркт миокарда", "Сердечная недостаточность", "Прочие заболевания сердца"])
  for row in data:
    csvWriter.writerow(row)
"""



# Рабочее на 2022.06.30
#seed = 348
#np.random.seed(seed)

data = np.array([])

good_level = []
good_level.append(0.72)
good_level.append(0.72)
good_level.append(0.72)
good_level.append(0.72)
good_level.append(0.72)

test_size_i = []
test_size_i.append(0.5)
test_size_i.append(0.4)
test_size_i.append(0.4)
test_size_i.append(0.4)
test_size_i.append(0.4)

count_ch = 0

middle_recall_score = 0

results_score = 0
#i = 4
for i in range(len(illness_list)):
#if i == 4:
    print_s(f'Заболевание = {illness_list[i]}')
    max_score = 0
    max_score_one = 0
    max_score_zero = 0
    max_middle_score_one_zero = 0
    last_max_score_one = 0
    last_max_score_zero = 0
    last_max_middle_score_one_zero = 0
    
    max_g = []
    rand_data = []
    good_rand_val = 0
    #k = 0
    #for k in range(4294967295):
    good_model = []
    find_good_level = False
    all_change_score_list = []
    m_i = -1
    max_accuration = 0
    max_classification_report = []
    max_predict = []
    max_test = []
    while len(rand_data) < 4294967295:
        rand_val = random.randint(0, 4294967295)+1
        #rand_val = np.random.randint(0, 4294967295, size=10)
        if rand_val not in rand_data:
            rand_data.append(rand_val)
            
            g = list(map(lambda x: True if int(x) == 1 else False,"{:032b}".format(rand_val)))
            g_str = "{:032b}".format(rand_val)
            g_str_revers = g_str[::-1]
            g.reverse()
            df_v = df[g].values
            df_v_v = df_v[0]
            
            #for m in range(len(g)+1):
            m = -1
            m_k = 0
            if len(rand_data) > 1:
                all_change_score_list.append(",".join(change_score_list))
            change_score_list = ['0.000']*33
            change_score_list[0] = "{:07.0f}".format(rand_val)
            
            m_i += 1
            if m_i % 10 == 0:
                #print(len(all_change_score_list))
                #print(all_change_score_list[m_i-2])
                fileName = 'submission_gritsenko_'+str(i)+'_(new)_param.csv'
                with open(fileName, "w", encoding = "utf8", newline = "") as f:
                    f.write(f'value,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n')
                    f.write('\n'.join(all_change_score_list))
                    f.close()
                
            while m < (len(g)) and m_k < 10:
                m += 1
                df_v_v_last  = df_v_v
                if m > 0:
                    #print(f'Пытаемся улучшить = {m}')
                    g[m-1] = not g[m-1]
                    if g[m-1] == True:
                        #g_str_revers[m-1] = '1'
                        #g_str_revers = g_str_revers[:(m-1)] + '1' + g_str_revers[(m-1)+1:]
                        temp = list(g_str_revers)
                        temp[m-1] = '1'
                        g_str_revers = "".join(temp)
                    else:
                        #g_str_revers[m-1] = '0'
                        #g_str_revers = g_str_revers[:(m-1)] + '0' + g_str_revers[(m-1)+1:]
                        temp = list(g_str_revers)
                        temp[m-1] = '0'
                        g_str_revers = "".join(temp)
                    rand_val = int(g_str_revers[::-1], 2)
                    df_v = df[g].values
                    df_v_v = df_v[0]
                    df_v_v_last = df_v_v
                    g[m-1] = not g[m-1]
                    
                if rand_val not in rand_data:
                    if m > 0:
                        rand_data.append(rand_val)

                    if i == 1:
                        data_train = pd.read_excel("train_dataset_train/train_2(cut_ONMK)_3.xls")
                        #data_train = pd.read_excel("train_dataset_train/train_2(cut)_3.xls")
                    elif i == 2:
                        data_train = pd.read_excel("train_dataset_train/train_2(cut_Stenokardiya)_3.xls")
                        #data_train = pd.read_excel("train_dataset_train/train_2(cut)_3.xls")
                    elif i == 3:
                        data_train = pd.read_excel("train_dataset_train/train_2(cut_Serdechnaya)_3.xls")
                        #data_train = pd.read_excel("train_dataset_train/train_2(cut)_3.xls")
                    elif i == 4:
                        data_train = pd.read_excel("train_dataset_train/train_2(cut_Prochee)_3.xls")
                        #data_train = pd.read_excel("train_dataset_train/train_2(cut)_3.xls")
                    else:
                        data_train = pd.read_excel("train_dataset_train/train_2(cut)_3.xls")
                    X = data_train.drop(['ID', 'ID_y',
                                         'Артериальная гипертензия',
                                         'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда',
                                         'Сердечная недостаточность', 'Прочие заболевания сердца',
                                         ], axis=1)
                    X = X.drop(drop_always, axis=1)
                    X = X.drop(df_v_v_last, axis=1)
                
                    #print(f'Длинна Х = {X.shape[1]}')
                    len_X = X.shape[1]
                
                    # data pre-processing
                    X = preprocessing.StandardScaler().fit_transform(X) #fit(X).transform(X)
                    y = data_train[illness_list[i]]*1.0
                
                    # Выделяем обучающую и тестовую выборки
                    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = test_size_i[i])#, random_state = seed)
                    
                    """
                    y_train = keras.utils.to_categorical(y_train,2) # 3 - число классов; необязательный параметр
                    y_test_cat = keras.utils.to_categorical(y_test,2)
                
                    x_val = X_train[-10000:]
                    y_val = y_train[-10000:]
                
                    # Создаем НС
                    #model_2 = Sequential()
                    #model_2.add(Dense(units = X.shape[1], input_dim = X.shape[1], activation = 'relu'))
                    # softmax или sigmoid - функция активация нейрона последнего слоя
                    #model_2.add(Dense(2, activation = 'softmax')) # softmax, sigmoid
                
                    #----------------------------------------------------------
                    # рабочий вариант 30.06.2022
                    #inputs = keras.Input(shape=(len_X,), name='digits')
                    #x_l = layers.Dense(len_X, activation='relu', name='dense_1')(inputs)
                    #x_l = layers.Dense(len_X, activation='relu', name='dense_2')(x_l)
                    #x_l = layers.Dense(math.trunc(len_X*4), activation='softmax', name='dense_3')(x_l)
                    #x_l = layers.Dense(math.trunc(len_X*16), activation='relu', name='dense_4')(x_l)
                    #-----------------------------------------------------------
               
                    
                    inputs = keras.Input(shape=(len_X,), name='digits')
                    x_l = layers.Dense(len_X, activation='relu', name='dense_1')(inputs)
                    x_l = layers.Dense(math.trunc(len_X*4), activation='relu', name='dense_2')(x_l)
                    x_l = layers.Dense(math.trunc(len_X*16), activation='softmax', name='dense_3')(x_l)
                    x_l = layers.Dense(math.trunc(len_X*32), activation='relu', name='dense_4')(x_l)
                    x_l = layers.Dense(math.trunc(len_X*16), activation='softmax', name='dense_5')(x_l)
                    x_l = layers.Dense(math.trunc(len_X*4), activation='relu', name='dense_6')(x_l)
                    x_l = layers.Dense(len_X, activation='relu', name='dense_7')(inputs)
                    # softmax или sigmoid - функция активация нейрона последнего слоя
                    outputs = layers.Dense(2, activation='softmax', name='predictions')(x_l)
                    model_2 = keras.Model(inputs=inputs, outputs=outputs)
                
                    # Указываем конфигурацию обучения (оптимизатор, функция потерь, метрики)
                    model_2.compile(optimizer = 'adam', # Оптимизатор
                                    loss = myLoss, # Минимизируемая функция потерь
                                    metrics = [tf.keras.metrics.Recall()] # Список метрик для мониторинга
                                    ) #['accuracy']) # mean_squared_error, Adam
                    #model_2.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = 'adam', metrics = [tf.keras.metrics.Recall()])
                    #model_2.summary()
                    
                    #
                    # Обучение нейронной сети
                    print('\nОбучение')
                    history = model_2.fit(X_train, y_train, validation_data=(x_val,y_val), batch_size = 32, shuffle = True, epochs = 100, verbose = 0)#epochs = 60, batch_size = 10, verbose = 0)
                    #verbose: Целое. 0, 1 или 2. Вербозный режим. 0 = бесшумный, 1 = шкала прогресса, 2 = одна строка в эпоху.
                    
                    # Возвращаемый объект "history" содержит записи
                    #  мониторинга потерь и метрик на этих данных
                    #  в конце каждой эпохи
                    print('\nhistory dict:',history.history)
                    #
                    print(f'')
                    print(f'Заболевание = {illness_list[i]}')
                    print(f'{datetime.strftime(datetime.now(),"%d.%m.%Y_%H.%M.%S.%f")} - Перебрали вариантов = {len(rand_data)}, текущий вариант = {rand_val}, лучший вариант = {good_rand_val}, max_score = {max_score}')
                    # Оценка результата
                    print('\nТестирование')
                    scores = model_2.evaluate(X_test, y_test_cat, batch_size=32, verbose = 0)
                    #
                    print("%s: %.2f%%" % (model_2.metrics_names[0], scores[0]*100)) # loss (потери)
                    print("%s: %.2f%%" % (model_2.metrics_names[1], scores[1]*100)) # acc (точность)
                    #print("%s: %.2f%%" % (model_2.metrics_names[2], scores[2]*100)) # acc (точность)
                    #
                    # Прогноз
                    classes = model_2.predict(X_test)#_classes(X_test) # , batch_size = 10
                    #predict_y = np.round(classes[0:,1],decimals = 0)
                    predict_y = np.round(classes[0:,1],decimals = 0).astype(np.uint8)
                    """
                    
                    #model_2 = RandomForestClassifier(n_estimators=100)
                    model_2 = models[i] #KNeighborsClassifier(n_neighbors=5) #VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
                    model_2.fit(X_train,y_train)
                    predict_y = model_2.predict(X_test) # zero_division ?
                    #score = recall_score(y_test, predict_y, average='macro')

                    # np.sum(classes == y_test) вернет сумму случаев, когда classes[i] = y_test[i]
                    accuration = np.sum(predict_y == y_test)/(len(X_test))*100# / 30.0 * 100

                    current_predict_y = np.round(list(predict_y),decimals = 0).astype(np.uint8)
                    current_y_test = np.round(list(y_test[0:]),decimals = 0).astype(np.uint8)
                    score_one = 0
                    for sc_i in range(len(current_y_test)):
                        if current_y_test[sc_i] == 1 and current_y_test[sc_i] == current_predict_y[sc_i]:
                            score_one += 1
                    if sum(current_y_test) > 0:
                        score_one = score_one/sum(current_y_test)
                    score_zero = 0
                    for sc_i in range(len(current_y_test)):
                        if current_y_test[sc_i] == 0 and current_y_test[sc_i] == current_predict_y[sc_i]:
                            score_zero += 1
                    score_zero = score_zero/(len(current_y_test)-sum(current_y_test))
                    middle_score_one_zero = (score_one+score_zero)/2

                    current_classification_report = classification_report(y_test,predict_y,zero_division=0)

                    """
                    print('')
                    print(f'Заболевание = {illness_list[i]}')
                    print(f'{datetime.strftime(datetime.now(),"%d.%m.%Y_%H.%M.%S.%f")} - Перебрали вариантов = {len(rand_data)}, текущий вариант = {rand_val}, лучший вариант = {good_rand_val}, max_score = {max_score}')
                    print("%s: %.2f%%" % ('recall_score',recall_score(y_test, predict_y, average='macro')*100))
                    print('')
                    #middle_recall_score += recall_score(y_test, np.round(classes[0:,1],decimals = 0), average='macro')*100
                    
                    print("Точность прогнозирования: " + str(accuration) + '%')
                    print(classification_report(y_test,predict_y))
                    print("Прогноз:")
                    print(np.round(list(predict_y),decimals = 0).astype(np.uint8))
                    #print(f'сумма - {sum(np.round(classes[0:,1]))}')
                    print("На самом деле:")
                    print(np.round(list(y_test[0:]),decimals = 0).astype(np.uint8))
                    print(f'сумма - {sum(y_test)}')
                    """
                
                    score = recall_score(y_test, predict_y, average='macro')
                    if g[m-1] == True:
                        # Если убрали параметр из процесса обучения:
                        change_score_list[m] = "{:.3f}".format((-1)*score)
                    else:
                        # Если добавили параметр в процесса обучения:
                        change_score_list[m] = "{:.3f}".format(score)
                    if score < 0.6 and m == 0:
                        break
                    
                    if ((score >= max_score and score_one >= max_score_one and score_zero >= max_score_zero) or
                       (score_one >= max_score_one and score_zero >= good_level[i]) or # and score >= max_score) or
                       (score_zero >= max_score_zero and score_one >= good_level[i])): # and score >= max_score)):
                        max_score = score
                        max_score_one = score_one
                        max_score_zero = score_zero
                        max_middle_score_one_zero = middle_score_one_zero
                        
                        good_rand_val = rand_val
                        if m > 0:
                            g[m-1] = not g[m-1]
                        max_g = g
                        
                        if m > 0:
                            g[m-1] = not g[m-1]
                        m = 0
                        m_k += 1
                        max_list = df[max_g].values
                        good_model = model_2
                        df_v_v_max = df_v_v_last

                        max_accuration = accuration
                        max_classification_report = current_classification_report
                        max_predict = np.round(list(predict_y),decimals = 0).astype(np.uint8)
                        max_test = np.round(list(y_test[0:]),decimals = 0).astype(np.uint8)
                        
                    if (len(rand_data) % 100 == 0) or (score >= good_level[i] and score_one >= good_level[i] and score_zero >= good_level[i]):
                        #print(f'Заболевание = {illness_list[i]}')
                        #print(g)
                        #print(f'{datetime.strftime(datetime.now(),"%d.%m.%Y_%H.%M.%S.%f")} - Перебрали вариантов = {len(rand_data)}, текущий вариант = {rand_val}, лучший вариант = {good_rand_val}, max_score = {max_score}')
                        #print(max_g)
                        #print('Не учитываются поля:')
                        #print(max_list)
                        #print('------------------------------------------------------------------------------------')
                        print_s('')
                        print_s(f'Заболевание = {illness_list[i]}')
                        print_s(f'{datetime.strftime(datetime.now(),"%d.%m.%Y_%H.%M.%S.%f")} - Перебрали вариантов = {len(rand_data)}, текущий вариант = {rand_val}, лучший вариант = {good_rand_val}, max_score = {max_score}')
                        print_s("%s: %.2f%%" % ('recall_score',max_score*100))
                        print_s('')
                        print_s("Точность прогнозирования: " + str(max_accuration) + '%')
                        print_s(max_classification_report)
                        print_s(f'Точность прогнозирования единиц: {max_score_one*100:.3f}%')
                        print_s(f'Точность прогнозирования нулей: {max_score_zero*100:.3f}%')
                        print_s(f'Средняя Точность прогнозирования единиц и нулей: {max_middle_score_one_zero*100:.3f}%')
                        print_s('')
                        print_s("Прогноз:")
                        print_s(f'{max_predict}')
                        print_s(f'сумма - {sum(max_predict)}')
                        print_s("На самом деле:")
                        print_s(f'{max_test}')
                        print_s(f'сумма - {sum(max_test)}')
                        #k += 1
                    if score >= good_level[i] and score_one >= good_level[i] and score_zero >= good_level[i]:
                        find_good_level = True
                        middle_recall_score += recall_score(y_test, predict_y, average='macro')*100
                        print_s(f'Заболевание = {illness_list[i]}')
                        print_s(f'Перебрали вариантов = {len(rand_data)}, текущий вариант = {rand_val}, лучший вариант = {good_rand_val}, max_score = {max_score}')
                        print_s(f'{max_g}')
                        print_s('Не учитываются поля:')
                        print_s(f'{max_list}')

                        
                
                        #df_v = df[max_g].values
                        df_v_v = df_v_v_max#df_v[0]
                
                        data_test = pd.read_excel("test_1_2.xls")
                        X_final_test = data_test.drop(['ID'], axis=1)
                        X_final_test = X_final_test.drop(drop_always, axis=1)
                        X_final_test = X_final_test.drop(df_v_v, axis=1)
                        X_final_ID = data_test['ID']
                        # data pre-processing
                        X_final_test = preprocessing.StandardScaler().fit_transform(X_final_test) #fit(X_final_test).transform(X_final_test)
               
                
                        #classes_final = good_model.predict(X_final_test)
                        #predict_y = np.round(classes_final[0:,1],decimals = 0).astype(np.uint8)
                        predict_y = np.round(good_model.predict(X_final_test),decimals = 0).astype(np.uint8)
                
                        if i == 0:
                            data = np.column_stack((X_final_ID, predict_y))
                        else:
                            data = np.column_stack((data, predict_y))
               
                        fileName = 'submission_gritsenko_'+str(i)+'_(new).csv'
                        with open(fileName, "w", encoding = "utf8", newline = "") as csvFile:
                            csvWriter = writer(csvFile, delimiter = ",")
                            csvWriter.writerow(["ID", "Артериальная гипертензия", "ОНМК", "Стенокардия, ИБС, инфаркт миокарда", "Сердечная недостаточность", "Прочие заболевания сердца"])
                            for row in data:
                                csvWriter.writerow(row)
                
                        print_s('------------------------------------------------------------------------------------')
                        print_s(f'Конец расчёта заболевания = {illness_list[i]}')
                        print_s('------------------------------------------------------------------------------------')
                        
                        break
            if find_good_level == True:
                break

middle_recall_score = middle_recall_score / 5
print_s("%s: %.2f%%" % ('middle_recall_score',middle_recall_score))

fileName = 'submission_gritsenko_(new).csv'
with open(fileName, "w", encoding = "utf8", newline = "") as csvFile:
    csvWriter = writer(csvFile, delimiter = ",")
    csvWriter.writerow(["ID", "Артериальная гипертензия", "ОНМК", "Стенокардия, ИБС, инфаркт миокарда", "Сердечная недостаточность", "Прочие заболевания сердца"])
    for row in data:
      csvWriter.writerow(row)




"""

seed = 348
np.random.seed(seed)

data = np.array([])

middle_recall_score = 0

results_score = 0
for i in range(len(illness_list)):
    
    if i == 1:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_ONMK).xls")
    elif i == 2:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Stenokardiya).xls")
    elif i == 3:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Serdechnaya).xls")
    elif i == 4:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Prochee).xls")
    else:
        data_train = pd.read_excel("train_dataset_train/train_2(cut).xls")
    X = data_train.drop(['ID', 'ID_y',
                         'Артериальная гипертензия',
                         'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда',
                         'Сердечная недостаточность', 'Прочие заболевания сердца',
                         ], axis=1)
    X = X.drop(drop_always, axis=1)
    X = X.drop(all_columns[i], axis=1)

    print(f'Длинна Х = {X.shape[1]}')
    len_X = X.shape[1]

    # data pre-processing
    X = preprocessing.StandardScaler().fit_transform(X) #fit(X).transform(X)

    data_test = pd.read_excel("test_1.xls")
    X_final_test = data_test.drop(['ID'], axis=1)
    X_final_test = X_final_test.drop(drop_always, axis=1)
    X_final_test = X_final_test.drop(all_columns[i], axis=1)
    X_final_ID = data_test['ID']

    # data pre-processing
    X_final_test = preprocessing.StandardScaler().fit_transform(X_final_test) #fit(X_final_test).transform(X_final_test)
    
    y = data_train[illness_list[i]]*1.0
    
    print(y)
    

    #X_train, X_test = X[:400], X[400:]
    #y_train, y_test = y[:400], y[400:]

    # Выделяем обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20)#, random_state = seed)
    y_train = keras.utils.to_categorical(y_train,2) # 3 - число классов; необязательный параметр
    y_test_cat = keras.utils.to_categorical(y_test,2)

    x_val = X_train[-10000:]
    y_val = y_train[-10000:]

    # Создаем НС
    #model_2 = Sequential()
    #model_2.add(Dense(units = X.shape[1], input_dim = X.shape[1], activation = 'relu'))
    #model_2.add(Dense(units = X.shape[1], input_dim = X.shape[1], activation = 'relu'))
    #model_2.add(Dense(units = X.shape[1], input_dim = X.shape[1], activation = 'relu'))
    #model_2.add(Dense(units = X.shape[1], input_dim = X.shape[1], activation = 'relu'))
    #model_2.add(Dense(units = X.shape[1], input_dim = X.shape[1], activation = 'relu'))
    # softmax или sigmoid - функция активация нейрона последнего слоя
    #model_2.add(Dense(2, activation = 'softmax')) # softmax, sigmoid
    inputs = keras.Input(shape=(len_X,), name='digits')
    x_l = layers.Dense(len_X, activation='relu', name='dense_1')(inputs)
    x_l = layers.Dense(len_X, activation='relu', name='dense_2')(x_l)
    x_l = layers.Dense(math.trunc(len_X*4), activation='softmax', name='dense_3')(x_l)
    x_l = layers.Dense(math.trunc(len_X*16), activation='relu', name='dense_4')(x_l)
    # softmax или sigmoid - функция активация нейрона последнего слоя
    outputs = layers.Dense(2, activation='softmax', name='predictions')(x_l)
    model_2 = keras.Model(inputs=inputs, outputs=outputs)
    
    # Указываем конфигурацию обучения (оптимизатор, функция потерь, метрики)
    model_2.compile(optimizer = 'adam', # Оптимизатор
                    loss = myLoss, # Минимизируемая функция потерь
                    metrics = [tf.keras.metrics.Recall()] # Список метрик для мониторинга
                    ) #['accuracy']) # mean_squared_error, Adam
    #model_2.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = 'adam', metrics = [tf.keras.metrics.Recall()])
    model_2.summary()
    #
    # Обучение нейронной сети
    print('\nОбучение')
    history = model_2.fit(X_train, y_train, validation_data=(x_val,y_val), batch_size = 512, epochs = 100, verbose = 2)#epochs = 60, batch_size = 10, verbose = 0)
    # Возвращаемый объект "history" содержит записи
    #  мониторинга потерь и метрик на этих данных
    #  в конце каждой эпохи
    print('\nhistory dict:',history.history)
    #
    # Оценка результата
    print('\nТестирование')
    scores = model_2.evaluate(X_test, y_test_cat, batch_size=512, verbose = 2)
    #
    print("%s: %.2f%%" % (model_2.metrics_names[0], scores[0]*100)) # loss (потери)
    print("%s: %.2f%%" % (model_2.metrics_names[1], scores[1]*100)) # acc (точность)
    #print("%s: %.2f%%" % (model_2.metrics_names[2], scores[2]*100)) # acc (точность)
    #
    # Прогноз
    classes = model_2.predict(X_test)#_classes(X_test) # , batch_size = 10
    print("%s: %.2f%%" % ('recall_score',recall_score(y_test, np.round(classes[0:,1],decimals = 0), average='macro')*100))
    middle_recall_score += recall_score(y_test, np.round(classes[0:,1],decimals = 0), average='macro')*100
    # np.sum(classes == y_test) вернет сумму случаев, когда classes[i] = y_test[i]
    # Поскольку в тестовой выборке 30 элементов (150 * 0.2), то для получения % пишем '/ 30.0 * 100'
    accuration = np.sum(np.round(classes[0:,1],decimals = 0) == y_test)/(len(X_test))*100# / 30.0 * 100
    print("Точность прогнозирования: " + str(accuration) + '%')
    print("Прогноз:")
    print(np.round(classes[0:,1],decimals = 0))
    print(f'сумма - {sum(np.round(classes[0:,1]))}')
    print("На самом деле:")
    print(list(y_test[0:]))
    print(f'сумма - {sum(y_test)}')


    classes_final = model_2.predict(X_final_test)

    predict_y = np.round(classes_final[0:,1],decimals = 0).astype(np.uint8)

    if i == 0:
        data = np.column_stack((X_final_ID, predict_y))
    else:
        data = np.column_stack((data, predict_y))

middle_recall_score = middle_recall_score / 5
print("%s: %.2f%%" % ('middle_recall_score',middle_recall_score))

fileName = 'submission_gritsenko.csv'
with open(fileName, "w", encoding = "utf8", newline = "") as csvFile:
  csvWriter = writer(csvFile, delimiter = ",")
  csvWriter.writerow(["ID", "Артериальная гипертензия", "ОНМК", "Стенокардия, ИБС, инфаркт миокарда", "Сердечная недостаточность", "Прочие заболевания сердца"])
  for row in data:
    csvWriter.writerow(row)

"""
