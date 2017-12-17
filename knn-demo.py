# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import datetime
from sklearn.ensemble import ExtraTreesRegressor

#function to load the datasets
def prodata():
    train = pd.read_csv("train.csv")
    feature = pd.read_csv('features.csv')
    test = pd.read_csv('test.csv')
    feature = del_unemployment(feature)
    train = del_train_markdown(train)
    return (train,test,feature)

#function to store the features column values
def del_unemployment(feature):
    feature = feature[['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','IsHoliday']]
    return feature
#function to store required date data
def del_train_markdown(train):
    train = train[train.Date >= '2011-11-11']
    return train
#logic to handle holiday wrt sales
def combi_train_feature(train,test,feature,markdown):
    train = np.array(train)
    test = np.array(test)
    feature = np.array(feature)
    train_x,train_y,test_x,dates=[],[],[],[]
    j = 0
    for i in range(len(train)):
        train_x.append([])
        store,dept,date,sales,isholiday = train[i]
        f = find_from_feature(store,date,feature,markdown)
        train_y.append(sales)
        train_x[j] =list(f)
        temp = date.split('-')
        y,m,d =int(temp[0]),int(temp[1]),int(temp[2])
        ymd = datetime.date(y,m,d)
        week = datetime.timedelta(days=7)
        preweek = ymd-week
        preweek = str(preweek)
        pre2week = ymd-week-week
        pre2week = str(pre2week)
        nextweek = ymd+week
        nextweek = str(nextweek)
        next2week = ymd+week+week
        next2week = str(next2week)
        next3week = ymd+week+week+week
        next3week = str(next3week)
        preweek = get_holiday_feature(preweek)
        pre2week = get_holiday_feature(pre2week)
        thisweek = get_holiday_feature(date)
        nextweek = get_holiday_feature(nextweek)
        next2week = get_holiday_feature(next2week)
        next3week = get_holiday_feature(next3week)
        train_x[j] =train_x[j]+preweek+thisweek+nextweek+pre2week+next2week+next3week
        j += 1
    j = 0
    for i in range(len(test)):
        test_x.append([])
        store,dept,date,isholiday = test[i]
        f = find_from_feature(store,date,feature,markdown)
        test_x[j] = list(f)
        temp = date.split('-')
        y,m,d = int(temp[0]),int(temp[1]),int(temp[2])
        ymd = datetime.date(y,m,d)
        week = datetime.timedelta(days=7)
        preweek = ymd-week
        preweek = str(preweek)
        nextweek = ymd+week
        nextweek = str(nextweek)
        preweek = get_holiday_feature(preweek)
        thisweek = get_holiday_feature(date)
        nextweek = get_holiday_feature(nextweek)
        pre2week = ymd-week-week
        pre2week = str(pre2week)
        next2week = ymd+week+week
        next2week = str(next2week)
        next3week = ymd+week+week+week
        next3week = str(next3week)
        pre2week = get_holiday_feature(pre2week)
        next2week = get_holiday_feature(next2week)
        next3week = get_holiday_feature(next3week)
        test_x[j] =test_x[j]+ preweek+thisweek+nextweek+pre2week+next2week+next3week
        dates.append(date)
        j += 1
    return (train_x,train_y,test_x,dates)

def find_from_feature(store,date,feature,markdown):
    for i in range(len(feature)):
        if feature[i][0] == store and feature[i][1] == date:
            for j in range(4,9):
                if pd.isnull(feature[i][j]):
                    feature[i][j] = markdown[j-4]
            return feature[i][2:-1]
#model the datasets
def linear_r(train_x,train_y,test_x):
    clf = LinearRegression()
    clf.fit(train_x,train_y)
    test_y = clf.predict(test_x)
    return test_y
def knn_model(train_x,train_y,test_x,k):
    clf = ExtraTreesRegressor(n_estimators=200,max_features='log2')
    clf.fit(train_x,train_y)
    test_y = clf.predict(test_x)
    return test_y
#handle missing values
def nan_rep(trains):
    md = []
    md.append(list(trains.MarkDown1))
    md.append(list(trains.MarkDown2))
    md.append(list(trains.MarkDown3))
    md.append(list(trains.MarkDown4))
    md.append(list(trains.MarkDown5))
    result = []
    for m in md:
        temp = np.array([i for i in m if pd.notnull(i)])
        result.append(temp.mean())
    return result
#handle holidays
def get_holiday_feature(date):
    easter = ['2010-04-02', '2011-04-22', '2012-04-06', '2013-03-29']
    super_bowl = ['2010-02-12','2011-02-11','2012-02-10','2013-02-08']
    labor = ['2010-09-10','2011-09-09','2012-09-07','2013-09-06']
    thx = ['2010-11-26','2011-11-25','2012-11-23','2013-11-29']
    chris = ['2010-12-24','2011-12-23','2012-12-28','2013-12-27']
    Halloween = ['2010-10-28', '2011-11-04', '2012-11-02', '2013-11-01']
    if date in super_bowl:
        return [0,0,0,0,0,1]
    elif date in labor:
        return [0,0,0,0,1,0]
    elif date in thx:
        return [0,0,0,1,0,0]
    elif date in chris:
        return [0,0,1,0,0,0]
    elif date in easter:
        return [0,1,0,0,0,0]
    elif date in Halloween:
        return [1,0,0,0,0,0]
    else:
        return [0,0,0,0,0,0]
#save the output
def write(y,store,dept,dates):
    f = open('result.csv','a')
    for i in range(len(y)):
        Id = str(store)+'_'+str(dept)+'_'+str(dates[i])
        sales = y[i]
        f.write('%s,%s\n'%(Id,sales))
    f.close()
if __name__=="__main__":
    f = open('result.csv','w')
    f.write('Id,Weekly_Sales\n')
    f.close()
    train,test,feature = prodata()
    for i in range(1,46):
        traindata = train[train.Store == i]
        testdata = test[test.Store == i]
        featuredata = feature[feature.Store == i]
        dept_train = list(set(traindata.Dept.values))
        dept_test = list(set(testdata.Dept.values))
        for dept in dept_test:
            if dept not in dept_train:
                tests = testdata[testdata.Dept == dept]
                dates = list(tests.Date)
                y=[0 for j in range(len(tests))]
                write(y,i,dept,dates)
                print(i,dept)
        
        for dept in dept_train:
            trains = traindata[traindata.Dept == dept]
            tests = testdata[testdata.Dept == dept]
            markdown = nan_rep(featuredata)
            train_x,train_y,test_x,dates = combi_train_feature(trains,tests,featuredata,markdown)
            k = 3
            if len(test_x) > 0:
                if len(train_x) <k:
                    test_y = knn_model(train_x,train_y,test_x,len(train_x))
#                    test_y = linear_r(train_x,train_y,test_x)
                    write(test_y,i,dept,dates)
                else:
                    test_y = knn_model(train_x,train_y,test_x,k)
#                    test_y = linear_r(train_x,train_y,test_x)
                    write(test_y,i,dept,dates)
#            test_y = linear_r(train_x,train_y,test_x)
#            write(test_y,i,dept,dates)
