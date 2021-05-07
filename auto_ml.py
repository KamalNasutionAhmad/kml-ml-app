# package
import streamlit as st

import pandas as pd 
import numpy as np


import matplotlib.pyplot as plt 
import matplotlib 
import seaborn as sns
matplotlib.use("Agg")

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
        X = data.data
        y = data.target
    elif name == 'Wine':
        data = datasets.load_wine()
        X = data.data
        y = data.target
    elif name == 'Titanic':
        X = pd.read_csv('train_titanic.csv')
        y = pd.read_csv('test_titanic.csv')
    else:
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
    return X, y

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", 'poly'])
        params['kernel'] = kernel
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'],kernel = params['kernel'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf



st.title("ML aplication")
st.text("Using streamlit framework")

activities = ["EDA", "Plot", "Model Building", "Do more predicting", "About"]
choise = st.sidebar.selectbox("Select activities", activities)

if choise == "EDA":
	st.subheader("Exploratory Data Analysis")

	data = st.file_uploader("upload dataset", type=["csv", "txt", 'xls'])
	if data is not None:
		df = pd.read_csv(data)
		st.dataframe(df.head())


		if st.checkbox("show shape"):
			st.write(df.shape)

		if st.checkbox("show column"):
			all_columns = df.columns.to_list()
			st.write(all_columns)
		
		if st.checkbox("Select columns to show"):
			selected_columns = st.multiselect("select columns ", all_columns)
			new_data = df[selected_columns]
			st.dataframe(new_data)

		if st.checkbox("show summary"):
			st.write(df.describe())

		if st.checkbox("show value counts"):
			st.write(df.iloc[:, -1]. value_counts())

elif choise == "Plot":
	st.subheader("Exploratory Data Analysis")

	data = st.file_uploader("upload dataset", type=["csv", "txt", 'xls'])
	if data is not None:
		df = pd.read_csv(data)
		st.dataframe(df.head())

	if st.checkbox("Correlation with seaborn"):
		st.write(sns.heatmap(df.corr(), annot=True))
		st.pyplot()

	if st.checkbox("Pie Chart"):
		all_columns = df.columns.to_list()
		columns_to_plot = st.selectbox("Select 1 columns", all_columns)
		pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
		st.write(pie_plot)
		st.pyplot()

	all_columns_names = df.columns.to_list()
	type_of_plot = st.selectbox("Select type of plot", ["area", "bar","line","hist", "box","kde"])
	selected_columns_names = st.multiselect("Select columns to plot", all_columns_names)

	if st.button("Generate Plot"):
		st.success("Generating costumization plot of {} for {}". format(type_of_plot, selected_columns_names))

		if type_of_plot == "area":
			cust_data = df[selected_columns_names]
			st.area_chart(cust_data)
		
		elif type_of_plot == "bar":
			cust_data = df[selected_columns_names]
			st.bar_chart(cust_data)
		
		elif type_of_plot == "line":
			cust_data = df[selected_columns_names]
			st.line_chart(cust_data)
		
		elif type_of_plot: 
			cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
			st.write(cust_plot)
			st.pyplot()

elif choise == "Model Building":
	st.subheader("Screning performance usesy 30 Machine Learning")

	data = st.file_uploader("upload dataset", type=["csv", "txt", 'xls'])
	if data is not None:
		df = pd.read_csv(data)
		st.dataframe(df.head())

		# model bulding 
		X = df.iloc[:,0:-1]
		Y = df.iloc[:, -1]
		seed = 7
		X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
		clf = LazyClassifier(verbose=0,ignore_warnings=True)
		models, predictions = clf.fit(X_train, X_test, y_train, y_test)
		hasil = pd.DataFrame(models)
	
		if st.button("predicting"):
			st.progress(100)
			st.success("30 Machine Learning")
			st.dataframe(hasil)

elif choise == "Do more predicting":
	st.subheader("Do more predicting")

	data = st.file_uploader("upload dataset", type=["csv", "txt", 'xls'])
	if data is not None:
		df = pd.read_csv(data)
		st.dataframe(df.head())
		X = df.iloc[:,0:-1]
		Y = df.iloc[:, -1]
		classifier_name = st.sidebar.selectbox('Select classifier',('KNN', 'SVM', 'Random Forest'))

		st.write('Shape of dataset:', X.shape)
		st.write('number of classes:', len(np.unique(Y)))
		params = add_parameter_ui(classifier_name)

		clf = get_classifier(classifier_name, params)
			#### CLASSIFICATION ####

		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		acc = accuracy_score(y_test, y_pred)

		st.write(f'Classifier = {classifier_name}')
		st.write(f'Accuracy =', acc)
		

if choise == "About":
	st.subheader("Exploratory Data Analysis")