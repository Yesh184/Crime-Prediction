import csv
from flask import Flask, render_template, flash, redirect, url_for, session, request, logging , jsonify
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
import re 
from patsy import dmatrices
from sklearn.metrics import accuracy_score,classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from keras.models import load_model
import tensorflow as tf


app = Flask(__name__,static_url_path='/static')
app.debug = True


def init_tf():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('model.h5')
    graph = tf.get_default_graph()

# MODEL_PATH = 'model.h5'
# model = tf.keras.models.load_model(MODEL_PATH)
# graph = tf.get_default_graph()


@app.route('/')
def index():
		district=[]
		day=[]
		with open('Book1.csv','r') as f:
			reader = csv.reader(f,delimiter=',')
			for row in reader:
				district.append(row[8])
				day.append(row[9])
		lat = request.args.get('lat')
		longi = request.args.get('long')
		return render_template('/index.html',district=district,day=day,lat=lat,longi=longi)

@app.route('/graphs')
def graphs():
    return render_template('/graphs.html')

#FOR GRAPH PAGE
@app.route('/plot')
def plot():
	crime_type = request.args.get('crime_category')
	df = pd.read_csv('C:\\Crime_rate\\flask\\app\\chicago_crime_2016.csv')
	crime_graph1 = df[['Primary Type', 'Arrest']].groupby('Primary Type').count()['Arrest']
	crime_graph1.plot(kind = 'barh', figsize= (10,6), fontsize= 8, title = 'Arrested')
	plt.show()
	plt.savefig('/static/images/new_plot.png')
	return jsonify(result='/static/images/new_plot.png')

# def normalizeX(data): 
#     data = (data - (-122.511293798596)) / (-120.5 - (-122.511293798596))
#     return data

def normalize(data): 
    data = (data - data.min()) / (data.max() - data.min())
    return data

def parse_time(x):
    DD=datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S")
    time=DD.hour
    day=DD.day
    month=DD.month
    year=DD.year
    return time, day, month, year

#getting season : summer, fall, winter, spring from months column
def get_season(x):
    summer=0
    fall=0
    winter=0
    spring=0
    if (x in [5, 6, 7]):
        summer=1
    if (x in [8, 9, 10]):
        fall=1
    if (x in [11, 0, 1]):
        winter=1
    if (x in [2, 3, 4]):
        spring=1
    return summer, fall, winter, spring

#getting season : summer, fall, winter, spring from months column
def get_season(x):
    summer=0
    fall=0
    winter=0
    spring=0
    if (x in [5, 6, 7]):
        summer=1
    if (x in [8, 9, 10]):
        fall=1
    if (x in [11, 0, 1]):
        winter=1
    if (x in [2, 3, 4]):
        spring=1
    return summer, fall, winter, spring

def preprocess_data(df):
    
    feature_list=df.columns.tolist()
    
    if "Id" in feature_list:
        feature_list.remove("Id")
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    cleanData=df[feature_list]
    cleanData.index=range(len(df))
    print ("Parsing dates...")
    cleanData["Time"], cleanData["Day"], cleanData["Month"], cleanData["Year"]=zip(*cleanData["Dates"].apply(parse_time))
    
    print ("Creating season features...")
    cleanData["Summer"], cleanData["Fall"], cleanData["Winter"], cleanData["Spring"]=zip(*cleanData["Month"].apply(get_season))
    print("Creating Lat/Long feature...")
    xy_scaler = preprocessing.StandardScaler()
    xy_scaler.fit(cleanData[["X","Y"]])
    cleanData[["X","Y"]] = xy_scaler.transform(cleanData[["X","Y"]])
    #set outliers to 0
    cleanData["X"]=cleanData["X"].apply(lambda x: 0 if abs(x)>5 else x)
    cleanData["Y"]=cleanData["Y"].apply(lambda y: 0 if abs(y)>5 else y)
    print ("Creating address features...")
    #recoding address as 0: if no interaction , 1: if interaction
    cleanData["Addr"]=cleanData["Address"].apply(lambda x: 1 if "/" in x else 0)
    print ("Creating dummy variables...")
    PD = pd.get_dummies(cleanData['PdDistrict'], prefix='PD')
    #DAYOfWeek = pd.get_dummies(cleanData["DayOfWeek"], prefix='WEEK')
    TIME = pd.get_dummies(cleanData['Time'],prefix='HOUR')
    Day = pd.get_dummies(cleanData['Day'],prefix='DAY')
    Month = pd.get_dummies(cleanData['Month'],prefix='MONTH')
    Year = pd.get_dummies(cleanData['Year'],prefix='YEAR')
    
    feature_list=cleanData.columns.tolist()
    
    print ("Joining features...")
    features = pd.concat([cleanData[feature_list],PD,TIME,Day,Month,Year],axis=1)
    
    print ("Droping processed columns...")
    cleanFeatures=features.drop(["PdDistrict","Address","Dates","Time","Day","Month","Year"],\
                                axis=1,inplace=False)
    
    print('Done!')
    
    return cleanFeatures


@app.route('/analysis')
def analysis():
    return render_template('/analysis.html')

#actual prediction
@app.route('/predict',methods=['POST'])
def predict():
	#season = request.form['season']
	date = request.form['date']
	dist = request.form['dist']
	addr = request.form['address']
	lat = request.form['lat']
	longi = request.form['long']

	actual_dt = pd.read_excel("crime_and_day.xlsx")
	actual_dt = actual_dt.iloc[:, 0:7]
	print("Printing actual df .... ",actual_dt.head(5))
	SNF1 = pd.DataFrame({'Dates': date,'Descript':'null', 'PdDistrict': dist,'Resolution': 'null', 'Address': addr,'X': lat,'Y': longi}, index=[0])
	actual_dt.append(SNF1)
	print("tail Printing..." , actual_dt.tail(2))

	#normalize
	scaler = preprocessing.StandardScaler()
	scaler.fit(actual_dt[["X","Y"]])
	actual_dt[["X","Y"]] = scaler.transform(actual_dt[["X","Y"]])
	actual_dt=actual_dt[abs(actual_dt["Y"])<100]
	actual_dt.index=range(len(actual_dt))
	actual_dt['X'] = normalize(actual_dt['X'])
	actual_dt['Y'] = normalize(actual_dt['Y'])
	print("normalize df location=>..."  , actual_dt.tail(2))


	features = preprocess_data(actual_dt)
	features = features.iloc[:,0:84]
	print("normalized features =>"  , features.tail(2))

	with graph.as_default():
		res = model.predict(features.tail(1))
		res = (res > 0.1)
		dataset = pd.DataFrame({'Prediction':res[:,0]})
		dataset = dataset*1
		print("this is result ==> ",res ,"dataset==>",dataset)
	return render_template('/ans.html',lat=lat,longi=longi, addr=addr, res=res[0][0]*1, dataset=dataset)

@app.route('/index')
def dashboard():
		district=[]
		day=[]
		with open('Book1.csv','r') as f:
			reader = csv.reader(f,delimiter=',')
			for row in reader:
				district.append(row[8])
				day.append(row[9])
		return render_template('/index.html',district=district,day=day)

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server...please wait until server has fully started"))
	init_tf()
	app.run(threaded=True)

