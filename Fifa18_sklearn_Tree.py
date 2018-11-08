#coding=utf-8

import pymysql
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sql_select(str_fields, str_conditions):
    connection = pymysql.connect(
                                 host='127.0.0.1',
                                 user='root',
                                 passwd='123456',
                                 db='FIFA18',
                                 charset='latin1'
                                )
    cursor = connection.cursor()

    cmd = "SELECT "+str_fields+" FROM complete WHERE "+str_conditions+" ;"

    cursor.execute(cmd)
    ret = cursor.fetchall()

    connection.commit()
    cursor.close()
    connection.close()

    return ret

def sk_labels(data, cluster):
    kmeans = KMeans(n_clusters=cluster).fit(data)
    labels = kmeans.labels_
    return labels

def sk_regressor(x_data, y_data, x_test, y_test):
    x_data = x_data.reshape(-1,1)
    x_test = x_test.reshape(-1,1)
    #reg = DecisionTreeRegressor(criterion='mse')
    reg = RandomForestRegressor(criterion='mse')
    dt = reg.fit(x_data, y_data)
    y_pred = dt.predict(x_test)
    return y_pred

def sk_regressor_2(x_data, y_data, x_test, y_test):
    x_data = x_data.reshape(-1,2)
    x_test = x_test.reshape(-1,2)
    #reg = DecisionTreeRegressor(criterion='mse')
    reg = RandomForestRegressor(criterion='mse')
    dt = reg.fit(x_data, y_data)
    y_pred = dt.predict(x_test)
    return y_pred

def plt_draw_1D(data, str_t, str_x):
    plt.figure(figsize=(7,7))
    plt.hist(data, bins=100, color='steelblue')
    plt.title(str_t)
    plt.xlabel(str_x)
    plt.ylabel('Ratio')
    plt.show()

def plt_draw_1D_s(data, str_t, str_x):
    fig, axes = plt.subplots(1, 2)
    sns.distplot(data, ax=axes[0], bins=20, kde=True, rug=True) #kde密度曲线 rug边际毛毯
    sns.kdeplot(data, ax=axes[1], shade=True) #shade阴影
    plt.title(str_t)
    plt.xlabel(str_x)
    plt.ylabel('Ratio')
    plt.show()

def plt_draw_1D_c(data_0, data_1, str_t, str_x):
    sns.kdeplot(data_0, color='r', shade=False) #shade阴影
    sns.kdeplot(data_1, color='b', shade=False) #shade阴影
    plt.title(str_t)
    plt.xlabel(str_x)
    plt.ylabel('Ratio')
    plt.show()

def plt_draw_2D(data, str_t, str_x, str_y):
    index = [str_x, str_y]    
    diction = {index[i]:data[:,i] for i in range(2)}
    sns.JointGrid(str_x, str_y, diction).plot(sns.regplot, sns.distplot)
    plt.xlim((None, None))
    plt.ylim((None, None))
    plt.show()

def plt_draw_2D_c(data_0, data_1):
    #print(data_0)
    #print(data_1)
    plt.scatter(x=data_0[:,0], y=data_0[:,1], c='r')
    plt.scatter(x=data_1[:,0], y=data_1[:,1], c='b')
    plt.show()

def plt_draw_3D(data, labels, str_t, str_x, str_y, str_z):
    colors = ['#E4846C', '#19548E', '#E44B4E', '#197D7F', '#0282C9']
    c_list = [colors[labels[i]] for i in range(data.shape[0])]

    plt.figure(figsize=(12,7))
    ax1 = plt.subplot(111,projection='3d')
    x,y,z = data[:,0], data[:,1], data[:,2]
    ax1.scatter(x,y,z,s=15,color=c_list)
    ax1.set_title(str_t)
    ax1.set_xlabel(str_x)
    ax1.set_ylabel(str_y)
    ax1.set_zlabel(str_z)
    plt.show()

def plt_draw_3D_c(data_0, data_1):
    plt.figure(figsize=(12,7))
    ax1 = plt.subplot(111,projection='3d')
    x,y,z = data_0[:,0], data_0[:,1], data_0[:,2]
    ax1.scatter(x,y,z,s=15,color='r')
    x,y,z = data_1[:,0], data_1[:,1], data_1[:,2]
    ax1.scatter(x,y,z,s=15,color='b')
    plt.show()

def analysis_1D():
    content = sql_select('weight_kg', 'league LIKE "%English Premier%" ') 
    content = np.array(content).flatten()
    print content.shape[0]
    plt_draw_1D_s(content, 'English', 'Weight')

def analysis_1D_c():
    compare = 'eur_value'
    objects = ['English Premier', 'Spanish Primera', 'German Bundesliga', 'French Ligue 1', 'Italian Serie A']
    objects_a = objects[0]
    objects_b = objects[3]

    content_0 = sql_select(compare, 'league LIKE "%'+objects_a+'%" ') 
    content_0 = np.array(content_0).flatten()
    print content_0.shape[0]
    content_1 = sql_select(compare, 'league LIKE "%'+objects_b+'%" ') 
    content_1 = np.array(content_1).flatten()
    print content_1.shape[0]
    plt_draw_1D_c(content_0, content_1, objects_a+' vs '+objects_b, compare)

def analysis_2D():
    content = sql_select('potential, eur_wage', 'league LIKE "%English Premier%" AND prefers_st = "True"') 
    content = np.array(content)
    print content.shape[0]
    plt_draw_2D(content, 'English ST', 'Potential', 'Wage')

def analysis_2D_c():
    #content = sql_select('overall, eur_value', 'league LIKE "%English Premier%" AND prefers_cm = "True"') 
    content = sql_select('potential, eur_wage', 'age < 23') 
    content = np.array(content)
    #print content.shape
    num = content.shape[0]
    np.random.shuffle(content)
    x_train = content[num/10:, 0]
    y_train = content[num/10:, 1]
    x_test = content[:num/10, 0]
    y_test = content[:num/10, 1]
    y_pred = sk_regressor(x_train, y_train, x_test, y_test)
    data_0 = np.zeros((num/10, 2))
    data_1 = np.zeros((num/10, 2))
    data_0[:,0] = x_test[:]
    data_0[:,1] = y_test[:]
    data_1[:,0] = x_test[:]
    data_1[:,1] = y_pred[:]
    plt_draw_2D_c(data_0, data_1)
    

def analysis_3D():
    content = sql_select('height_cm, weight_kg, eur_value', 'league LIKE "%English Premier%" AND prefers_cb = "True"') 
    content = np.array(content)
    print content.shape[0]
    labels = sk_labels(content, 5)
    plt_draw_3D(content, labels, 'English CB', 'Height', 'Weight', 'Value')

def analysis_3D_c():
    #content = sql_select('overall, age, eur_value', 'league LIKE "%English Premier%" AND prefers_cb = "True"') 
    content = sql_select('age, overall, potential', 'prefers_st = "True"') 
    content = np.array(content)
    #print content.shape
    num = content.shape[0]
    print(num)
    np.random.shuffle(content)
    x_train = content[num/10:, 0:2]
    y_train = content[num/10:, 2]
    x_test = content[:num/10, 0:2]
    y_test = content[:num/10, 2]
    y_pred = sk_regressor_2(x_train, y_train, x_test, y_test)
    data_0 = np.zeros((num/10, 3))
    data_1 = np.zeros((num/10, 3))
    data_0[:,0:2] = x_test[:]
    data_0[:,2] = y_test[:]
    data_1[:,0:2] = x_test[:]
    data_1[:,2] = y_pred[:]
    plt_draw_3D_c(data_0, data_1)
    
if __name__ == '__main__':
    #analysis_1D()
    #analysis_1D_c()
    #analysis_2D()
    #analysis_2D_c()
    #analysis_3D()
    analysis_3D_c()
