import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestRegressor


models = []
# Артериальная гипертензия
model = RandomForestClassifier(max_depth=300,n_estimators=540, random_state=107)#KNeighborsClassifier(n_neighbors=10,weights='distance')#SVC(kernel='poly', degree=2)#GaussianNB()#LogisticRegression(solver='liblinear', C=1)#GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)#DecisionTreeClassifier(random_state=2018)#QuadraticDiscriminantAnalysis()#MLPClassifier(alpha = 0.01, max_iter = 200, solver = 'lbfgs', tol = 0.001)#KNeighborsClassifier(n_neighbors=3)#SGDClassifier(loss="hinge", penalty="l2", max_iter=5)#SVC(gamma = 'auto')#RandomForestClassifier(max_depth = 10, n_estimators = 30, max_features = 1)#GradientBoostingClassifier()#RandomForestClassifier(max_depth = 10, n_estimators = 30, max_features = 1)#LogisticRegression(random_state=0)#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)#KNeighborsClassifier(n_neighbors=5)
models.append(model)
# ОНМК Острое нарушение мозгового кровообращения (ОНМК, инсульт)
model = RandomForestClassifier(max_depth=300,n_estimators=540, random_state=17)#KNeighborsClassifier(n_neighbors=3)#SGDClassifier(loss="hinge", penalty="l2", max_iter=5)#RandomForestClassifier(max_depth = 10, n_estimators = 30, max_features = 1)#GradientBoostingClassifier()#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)#SVC(gamma = 2, C = 1.0)#SGDClassifier(max_iter = 1500, tol = 1e-4)#HistGradientBoostingClassifier(max_iter=100)#RandomForestClassifier(n_estimators=100)
models.append(model)
# Стенокардия, ИБС, инфаркт миокарда
model = RandomForestClassifier(max_depth=300,n_estimators=540, random_state=17)#KNeighborsClassifier(n_neighbors=3)#SGDClassifier(loss="hinge", penalty="l2", max_iter=5)#RandomForestClassifier(max_depth = 10, n_estimators = 30, max_features = 1)#GradientBoostingClassifier()#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)
models.append(model)
# Сердечная недостаточность
model = RandomForestClassifier(max_depth=300,n_estimators=540, random_state=17)#KNeighborsClassifier(n_neighbors=3)#SGDClassifier(loss="hinge", penalty="l2", max_iter=5)#RandomForestClassifier(max_depth = 10, n_estimators = 30, max_features = 1)#GradientBoostingClassifier()#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)
models.append(model)
# Прочие заболевания сердца
model = RandomForestClassifier(max_depth=300,n_estimators=540, random_state=17)#KNeighborsClassifier(n_neighbors=3)#SGDClassifier(loss="hinge", penalty="l2", max_iter=5)#RandomForestClassifier(max_depth = 10, n_estimators = 30, max_features = 1)#GradientBoostingClassifier()#RandomForestClassifier(max_depth = 10, n_estimators = 30, max_features = 8)#SVC(gamma = 'auto')#SVC(gamma = 2, C = 1.0)#RandomForestClassifier(n_estimators=100)
models.append(model)

illness_list = ['Артериальная гипертензия', 'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда', 'Сердечная недостаточность', 'Прочие заболевания сердца']

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


i = 0
for i in range(len(illness_list)):
    if i == 1:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_ONMK)_3.xls")
    elif i == 2:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Stenokardiya)_3.xls")
    elif i == 3:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Serdechnaya)_3.xls")
    elif i == 4:
        data_train = pd.read_excel("train_dataset_train/train_2(cut_Prochee)_3.xls")
    else:
        data_train = pd.read_excel("train_dataset_train/train_2(cut)_3.xls")

    #X = data_train.drop(['ID', 'ID_y',
    #                    'Артериальная гипертензия',
    #                    'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда',
    #                    'Сердечная недостаточность', 'Прочие заболевания сердца',
    #                    ], axis=1)
    X2 = data_train.drop(drop_always, axis=1)
    X3 = X2.copy()

    corr_X3 = X3.corr()
    list_corr_X3 = corr_X3[illness_list[i]].sort_values(ascending=False)
    #print(type(list_corr_X2))

    list_true_axes = []
    for k in range(len(list_corr_X3)):
        #print(list_corr_X2.values[k])
        #print(list_corr_X2.axes[0][k])
        if k > 0 and list_corr_X3.values[k] >= 0.01:
            list_true_axes.append(list_corr_X3.axes[0][k])
            print(f'{list_corr_X3.values[k]} = {list_corr_X3.axes[0][k]}')
        
    print('---------------------------------------------------------------------------------------------')
    #print(X.get(list_true_axes))

    X = X2.get(list_true_axes)
    len_X = X.shape[1]
                
    # data pre-processing
    X = preprocessing.StandardScaler().fit_transform(X) #fit(X).transform(X)
    y = data_train[illness_list[i]]*1.0

    # Выделяем обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.50)#, random_state = seed)



    score = 0
    score_zero = 0
    score_one = 0
    middle_score_one_zero = 0
    max_score = score
    max_score_one = score_one
    max_score_zero = score_zero
    max_middle_score_one_zero = middle_score_one_zero

    good_level = []
    good_level.append(0.75)
    good_level.append(0.75)
    good_level.append(0.75)
    good_level.append(0.75)
    good_level.append(0.75)

    m = 0
#while (score_zero <= 0.75 and score_one <= 0.75):
#for m_i in range(1000000000):
    m += 1
    #print(m)
    #model_2 = RandomForestClassifier(n_estimators=100)
    #GradientBoostingClassifier()
    model_2 = models[i] #KNeighborsClassifier(n_neighbors=5) #VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')

    model_2.fit(X_train,y_train)
    predict_y = model_2.predict(X_test)
    
    score = recall_score(y_test, predict_y, average='macro')
    #print(f'score = {score}')

    # np.sum(classes == y_test) вернет сумму случаев, когда classes[i] = y_test[i]
    accuration = np.sum(predict_y == y_test)/(len(X_test))*100# / 30.0 * 100

    current_predict_y = np.round(list(predict_y),decimals = 0).astype(np.uint8)
    current_y_test = np.round(list(y_test[0:]),decimals = 0).astype(np.uint8)
    score_one = 0
    for sc_i in range(len(current_y_test)):
        if current_y_test[sc_i] == 1 and current_y_test[sc_i] == current_predict_y[sc_i]:
            score_one += 1
    score_one = score_one/sum(current_y_test)
    #print(f'score_one = {score_one}')
    score_zero = 0
    for sc_i in range(len(current_y_test)):
        if current_y_test[sc_i] == 0 and current_y_test[sc_i] == current_predict_y[sc_i]:
            score_zero += 1
    score_zero = score_zero/(len(current_y_test)-sum(current_y_test))
    #print(f'score_zero = {score_zero}')
    middle_score_one_zero = (score_one+score_zero)/2

    current_classification_report = classification_report(y_test,predict_y)

    print("Точность прогнозирования: " + str(accuration) + '%')
    print(f'Точность прогнозирования единиц: {score_one*100:.3f}%')
    print(f'Точность прогнозирования нулей: {score_zero*100:.3f}%')
    print(f'Средняя Точность прогнозирования единиц и нулей: {middle_score_one_zero*100:.3f}%')
    print('')


    #if ((score >= max_score and score_one >= max_score_one and score_zero >= max_score_zero) or
    #    (score_one >= max_score_one and score_zero >= good_level[i]) or # and score >= max_score) or
    #    (score_zero >= max_score_zero and score_one >= good_level[i])): # and score >= max_score)):
    if i > -1:
        max_score = score
        max_score_one = score_one
        max_score_zero = score_zero
        max_middle_score_one_zero = middle_score_one_zero

        max_accuration = accuration
        max_classification_report = current_classification_report
        max_predict = np.round(list(predict_y),decimals = 0).astype(np.uint8)
        max_test = np.round(list(y_test[0:]),decimals = 0).astype(np.uint8)

        #print(f'max_score = {max_score}')

    #if (m % 100 == 0):
    if i > -1:
        #print(m)
        print("Точность прогнозирования: " + str(accuration) + '%')
        print(f'Точность прогнозирования единиц: {score_one*100:.3f}%')
        print(f'Точность прогнозирования нулей: {score_zero*100:.3f}%')
        print(f'Средняя Точность прогнозирования единиц и нулей: {middle_score_one_zero*100:.3f}%')
        print('')
        print("Максимальная Точность прогнозирования: " + str(max_accuration) + '%')
        print(f'Максимальная Точность прогнозирования единиц: {max_score_one*100:.3f}%')
        print(f'Максимальная Точность прогнозирования нулей: {max_score_zero*100:.3f}%')
        print(f'Максимальная Средняя Точность прогнозирования единиц и нулей: {max_middle_score_one_zero*100:.3f}%')
        print(current_classification_report)
        print('')
        print("Прогноз:")
        print(max_predict)
        print(f'сумма - {sum(max_predict)}')
        print("На самом деле:")
        print(max_test)
        print(f'сумма - {sum(max_test)}')

    #if  (score >= good_level[i] and score_one >= good_level[i] and score_zero >= good_level[i]):
    #    break
    







"""
fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
#ax1.scatter(X[base_columns[0]], base_columns[1])
x_i = np.array(list(X[base_columns[10]][0:]))
#x_i = x_i/max(x_i)
y_i = np.array(list(X[base_columns[21]][0:]))
#y_i = y_i/max(y_i)
c_i = list(X[illness_list[0]][0:])
for i in range(len(X[base_columns[6]])):
    c = 'black'
    if c_i[i] == 0:
        c = 'blue'
    else:
        c = 'red'
    circ = plt.Circle((x_i[i]*10+random.randint(0, 100)/100*5-2.5, y_i[i]*10+random.randint(0, 100)/100)*5, 0.05, color=c, alpha=1)
    ax2.add_patch(circ)
    #print(f'{x_i[i]}, {y_i[i]}')
    plt.xlim([-10,110])
    plt.ylim([-10,70])
plt.show()
"""
