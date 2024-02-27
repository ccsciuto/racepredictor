import pandas as pd
import warnings
from datetime import date
import datetime
warnings.filterwarnings('ignore')
from garminconnect import (
    Garmin
)
import statsmodels.api as sm
import pickle
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import csv
import codecs
import urllib.request
import urllib.error
import sys
import Passwords


#pulling in missing dates
dates = pd.read_csv("Calendar.csv", sep=",")
dates = dates[["dt"]]
dates["dt"] = pd.to_datetime(dates["dt"]).dt.date

# #pulling in weather
#
# BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
#
# ApiKey='RH2FMWBES6WRHYVWR4XA33D86'
# #UnitGroup sets the units of the output - us or metric
# UnitGroup='us'
#
# #Location for the weather data
# Location='Austin,TX'
#
# #Start and end dates
#
# StartDate = '2021-08-28'
# EndDate='2024-02-06'
#
# #CSV format requires an 'include' parameter below to indicate which table section is required
# ContentType="csv"
#
# #include sections
# #values include days,hours,current,alerts
# Include="days"
#
# #basic query including location
# ApiQuery=BaseURL + Location
#
# #append the start and end date if present
# if (len(StartDate)):
#     ApiQuery+="/"+StartDate
#     if (len(EndDate)):
#         ApiQuery+="/"+EndDate
#
# #Url is completed. Now add query parameters (could be passed as GET or POST)
# ApiQuery+="?"
#
# #append each parameter as necessary
# if (len(UnitGroup)):
#     ApiQuery+="&unitGroup="+UnitGroup
#
# if (len(ContentType)):
#     ApiQuery+="&contentType="+ContentType
#
# if (len(Include)):
#     ApiQuery+="&include="+Include
#
# ApiQuery+="&key="+ApiKey
#
# print(' - Running query URL: ', ApiQuery)
# print()
#
# try:
#     CSVBytes = urllib.request.urlopen(ApiQuery)
# except urllib.error.HTTPError  as e:
#     ErrorInfo= e.read().decode()
#     print('Error code: ', e.code, ErrorInfo)
#     sys.exit()
# except urllib.error.URLError as e:
#     ErrorInfo= e.read().decode()
#     print('Error code: ', e.code,ErrorInfo)
#     sys.exit()
#
# CSVText = csv.reader(codecs.iterdecode(CSVBytes, 'utf-8'))
file = pd.read_csv("weather.csv", sep=",")
wthr = file[["datetime","dew"]]
wthr["Date"] = pd.to_datetime(wthr["datetime"]).dt.date

#Filtering dataset to aug/28 to todays date
start_date = pd.to_datetime('08-28-2021').date()
end_date = date.today()
dates = dates[dates["dt"] >= start_date]
dates = dates[dates["dt"] <= end_date]
dates["Date"] = dates["dt"]
dates = dates[["Date"]]

#Pulling in running data


def init_api():
    api = Garmin(Passwords.garminusername, Passwords.garminpassword)
    api.login()

    return api

api = init_api()

start_date = datetime.date(2021, 8, 28)
end_date = datetime.date.today()

activities = api.get_activities_by_date(
                start_date.isoformat(), end_date.isoformat(), 'running')

for activity in activities:
    activity_id = activity["activityId"]
    gpx_data = api.download_activity(
                        activity_id, dl_fmt=api.ActivityDownloadFormat.GPX
                    )
    output_file = f"/Users/ceceliasciuto/PycharmProjects/racepredictor/garmindata/{str(activity_id)}.KML"
    with open(output_file, "wb") as fb:
        fb.write(gpx_data)
print(gpx_data)
data = pd.read_csv("garmindata.csv", sep=",")
data.drop(['Activity Type', 'Favorite', 'Title','Moving Time','Max HR','Max Run Cadence',
       'Avg Pace', 'Best Pace','Avg Vertical Ratio', 'Avg Vertical Oscillation',
       'Training Stress Score®', 'Grit', 'Flow', 'Dive Time', 'Min Temp',
       'Surface Interval', 'Decompression', 'Best Lap Time', 'Number of Laps',
       'Max Temp',  'Elapsed Time', 'Min Elevation',
       'Max Elevation'], axis=1, inplace=True)

#Cleaning running data
times = pd.to_timedelta(data['Time'])
dist = data["Distance"]
data["Date"] = pd.to_datetime(data["Date"]).dt.date
data["Avg Speed"] = dist / (times / pd.Timedelta('1 hour'))
data = data.replace(to_replace="--",value=0)
data["Calories"] = data["Calories"].str.replace(',','').astype(float)
#combing weather data
data = pd.merge(data,wthr, how='left')
data["Power"] = ((data["Avg Speed"]/data["Avg HR"])*(data["dew"]+50))*10
data["Power"] = data["Power"].astype(float)
data.sort_values(by="Date",ascending=True,inplace=True)

pickle_in = open("run_model.pickle", "rb")
model2 = pickle.load(pickle_in)
t_date = date.today()
l_date = date(2024, 2, 19)
delta = l_date - t_date
index_future_dates = pd.date_range(start=date.today(),end='2024-02-19')
pred2=model2.predict(start=len(data),end=len(data)+delta.days,typ='levels').rename('ARIMA Predictions')
pred2.index=index_future_dates
AVGHR = float(input("What is your expected average heart rate? "))
dew=(10.25+50)
mph = ((pred2[-1]/10)/dew)*AVGHR
if (((60/mph)-10)*60).astype(int) >= 120:
    print(round(mph,2), "MPH or",str((60/mph).astype(int))+":"+str(f"{((((60/mph)-10)*60).astype(int)-120):02d}")+"/Mile")
elif (((60/mph)-10)*60).astype(int) > 60:
    print(round(mph,2), "MPH or",str((60/mph).astype(int))+":"+str(f"{((((60/mph)-10)*60).astype(int)-60):02d}")+"/Mile")
else:
    print(round(mph,2), "MPH or",str((60/mph).astype(int))+":"+str(f"{abs(((((60/mph)-10)*60).astype(int))):02d}")+"/Mile")