#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from operator import eq
from matplotlib.lines import Line2D

#Load Data Sets
cabData = pd.read_csv("D:/Internship/Data Glacier/Week 2/DataSets/Cab_Data.csv")
transactions = pd.read_csv('D:/Internship/Data Glacier/Week 2/DataSets/Transaction_ID.csv')
tripsData = pd.merge(left = cabData, right = transactions, how = 'left').sort_values(by=['Company', 'Date of Travel'])
#Data sets for each city and customer
city = pd.read_csv('D:/Internship/Data Glacier/Week 2/DataSets/City.csv')
customer = pd.read_csv('D:/Internship/Data Glacier/Week 2/DataSets/Customer_ID.csv')
customer.set_index('Customer ID', inplace=True)

#Convert dates to days starting from 31/01/2016
tripsData['Date of Travel'] = tripsData['Date of Travel'].apply(lambda x : xlrd.xldate_as_datetime(x, 0))

#Seasonal data
monthlyData = tripsData[['Company', 'Date of Travel', 'Price Charged']]
monthlyData = monthlyData.groupby(['Company', pd.Grouper(key='Date of Travel', freq= 'M')]).sum()
yearlyData = tripsData[['Company', 'Customer ID', 'Transaction ID', 'KM Travelled', 'Price Charged']]
yearlyData = pd.concat([yearlyData, pd.Series(tripsData['Date of Travel'].apply(func=lambda x: datetime(x.year, 12, 31)))], join='inner', axis=1)
#Plot monthly profits for each company
fig, ax1 = plt.subplots()
ax1.plot(monthlyData.loc['Pink Cab', 'Price Charged'], label='Pink Cab')
ax1.plot(monthlyData.loc['Yellow Cab', 'Price Charged'], label='Yellow Cab')
ax1.set_xlabel('Date')
ax1.set_ylabel('Profit (Millions USD)')
ax1.set_title('Monthly profits')
"""ax1.axvline(x= '2017-01-01', linestyle= 'dashed', linewidth = 0.5, color= 'black')
ax1.axvline(x= '2018-01-01', linestyle= 'dashed', linewidth = 0.5, color= 'black')
ax1.axvline(x= '2019-01-01', linestyle= 'dashed', linewidth = 0.5, color= 'black')"""
avgYellow = monthlyData.loc['Yellow Cab', 'Price Charged'].mean()
avgPink = monthlyData.loc['Pink Cab', 'Price Charged'].mean()
ax1.axhline(y= avgYellow, linestyle= '-.', linewidth= 0.7, color='orange')
ax1.axhline(y= avgPink, linestyle= '-.', linewidth= 0.7, color='blue')
plt.legend(loc='best')

#Exploring the customer base
custTrips = yearlyData.pivot_table(columns=['Company', 'Date of Travel'], index=['Customer ID'], aggfunc= {'Transaction ID': 'count', 'KM Travelled': 'sum', 'Price Charged': 'sum'})
custTrips.rename(columns={'Transaction ID': 'NumOfTrips'}, inplace=True)
"""custTrips.fillna({('NumOfTrips', 'Pink Cab', '2016-12-31'): 0,
                  ('NumOfTrips', 'Pink Cab', '2017-12-31'): 0,
                  ('NumOfTrips', 'Pink Cab', '2018-12-31'): 0,
                  ('NumOfTrips', 'Yellow Cab', '2016-12-31'): 0,
                  ('NumOfTrips', 'Yellow Cab', '2017-12-31'): 0,
                  ('NumOfTrips', 'Yellow Cab', '2018-12-31'): 0}, inplace=True)"""
custTrips.fillna(0, inplace=True)
#Which customer uses which company more:
yellow_customers = custTrips.loc[:,('NumOfTrips', 'Yellow Cab')] > custTrips.loc[:,('NumOfTrips', 'Pink Cab')]  #A boolean pivot_table with columns as years
pink_customers = ~yellow_customers
yellow_genders = pd.DataFrame(index=['Male', 'Female'], columns=pd.date_range(start=yearlyData['Date of Travel'].min(), end=yearlyData['Date of Travel'].max(), freq="Y"))  #To store gender count in this DF
pink_genders = pd.DataFrame(index=['Male', 'Female'], columns=yellow_genders.columns)

#Assuming yellow and pink customers change with years
for year in yellow_genders.columns:
    #Retrieve customer info for yellow and pink customers of that year
    y_cust = custTrips.index.values[yellow_customers[year]]
    y_cust = customer.loc[y_cust,'Gender']

    p_cust = custTrips.index.values[pink_customers[year]]
    p_cust = customer.loc[p_cust, 'Gender']

    #Count males and females and assign them accordignly
    yellow_genders.loc['Male', year] = y_cust.value_counts()[0]
    yellow_genders.loc['Female', year] = y_cust.value_counts()[1]

    pink_genders.loc['Male', year] = p_cust.value_counts()[0]
    pink_genders.loc['Female', year] = p_cust.value_counts()[1]

fig, ax2 = plt.subplots()
x= np.arange(len(pink_genders.columns))
#Histogram of male customers
ax2.bar(x= x-0.35/2, height=pink_genders.loc['Male'], width= 0.2, label='Pink Cab', color= 'deeppink')
ax2.bar(x= x+0.35/2, height=yellow_genders.loc['Male'], width= 0.2, label='Yellow Cab', color='darkorange')
#Histogram of female customers
ax2.bar(x=x-0.35/2, height=pink_genders.loc['Female'], bottom=pink_genders.loc['Male'], width= 0.2, color= 'lightpink')
ax2.bar(x=x+0.35/2, height=yellow_genders.loc['Female'], bottom=yellow_genders.loc['Male'], width= 0.2, color='gold')

ax2.set_title('Number of Customers Classified by Gender (Bottom: Male, Top: Female)')
ax2.set_xticks(x)
ax2.set_xticklabels(['2016', '2017', '2018'])
plt.legend(loc='best')

rel_pink_genders = pink_genders / pink_genders.apply(axis=0, func= 'sum')
rel_yellow_genders = yellow_genders / yellow_genders.apply(axis=0, func= 'sum')
relative_values = pd.concat([rel_pink_genders.unstack(), rel_yellow_genders.unstack()]) * 100

for i, rect in enumerate(ax2.patches):
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        label_x = x + width / 2
        label_y = y + height / 2
        ax2.text(label_x, label_y, str(np.round(relative_values[i], 1)) + '%', ha='center', va='center')

#Activity of the company
activity = tripsData.groupby(['Company', pd.Grouper(key='Date of Travel', freq='Y')]).agg({'Transaction ID': 'count', 'KM Travelled': 'sum'})
activity.rename(columns={'Transaction ID': 'NumOfTrips'}, inplace=True)
activity.reset_index(level='Date of Travel', inplace=True)

#Barplot of Activity in each city
x=np.arange(len(tripsData.City.unique()))
fig, ax5 = plt.subplots()
ax5.bar(x=x - .35 / 2,
        height=tripsData.loc[tripsData['Company'] == 'Pink Cab', 'City'].value_counts(), width=.35,
        label='Pink Cab')

ax5.bar(x=x + .35 / 2,
        height=tripsData.loc[tripsData['Company'] == 'Yellow Cab', 'City'].value_counts(), width=0.35,
        label='Yellow Cab')

ax5.set_title('Activity in each city')
ax5.set_xticks(x)
ax5.set_ylabel('Num Of Trips')
ax5.set_xticklabels(tripsData.City.unique())
ax5.tick_params(axis='x', labelrotation = 45)
plt.legend(loc='best')

#According to KM Travelled
firstYear = tripsData['Date of Travel'].min()
lastYear = tripsData['Date of Travel'].max()
x = np.arange(int(lastYear.strftime('%Y'))-int(firstYear.strftime('%Y')) + 1)
fig, ax6 = plt.subplots()
ax6.bar(x= x - 0.35 / 2,
        height= activity.loc['Pink Cab', 'KM Travelled'],
        width= 0.35,
        label= 'Pink Cab')

ax6.bar(x=x + 0.35 / 2,
        height=activity.loc['Yellow Cab', 'KM Travelled'],
        width= 0.35,
        label= 'Yellow Cab')

ax6.set_title('KM Travelled each year')
ax6.set_xticks(x)
ax6.set_xticklabels(pd.date_range(firstYear, lastYear, freq='Y').strftime('%Y'))
plt.legend()

#According to number of trips
fig, ax8 = plt.subplots()
ax8.bar(x= x - 0.35 / 2,
        height= activity.loc['Pink Cab', 'NumOfTrips'],
        width= 0.35,
        label= 'Pink Cab')

ax8.bar(x=x + 0.35 / 2,
        height=activity.loc['Yellow Cab', 'NumOfTrips'],
        width= 0.35,
        label= 'Yellow Cab')

ax8.set_title('Number of trips each year')
ax8.set_xticks(x)
ax8.set_xticklabels(pd.date_range(firstYear, lastYear, freq='Y').strftime('%Y'))
plt.legend()

#Seasonal Activity
monthlyData = pd.concat(
        [
                monthlyData,
                tripsData[['Transaction ID', 'Date of Travel', 'Company']].groupby(['Company', pd.Grouper(key='Date of Travel', freq='M')]).count()
         ], join='inner', axis=1)
monthlyData.rename(columns={'Transaction ID':'NumOfRides'}, inplace=True)
fig, ax9 = plt.subplots()
ax9.plot(monthlyData.loc['Pink Cab', 'NumOfRides'], label= 'Pink Cab')
ax9.plot(monthlyData.loc['Yellow Cab', 'NumOfRides'], label= 'Yellow Cab')
ax9.axvline(x= '2017-01-01', linestyle= 'dashed', linewidth = 0.5, color= 'black')
ax9.axvline(x= '2018-01-01', linestyle= 'dashed', linewidth = 0.5, color= 'black')
ax9.axvline(x= '2019-01-01', linestyle= 'dashed', linewidth = 0.5, color= 'black')
ax9.set_title('Comapany\'s Seasonal Activity')
plt.ylabel('Num. of Rides')
plt.legend(loc='best')

#Daily activity
dailyRides = tripsData[['Transaction ID', 'Date of Travel', 'Company']].groupby(['Company', pd.Grouper(key='Date of Travel', freq='D')]).count()
dailyRides.reset_index(level='Date of Travel', inplace=True)
dailyRides['Date of Travel'] = dailyRides['Date of Travel'].dt.day_name()
#Sort according to name of weekday
weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dailyRides['Date of Travel'] = pd.Categorical(dailyRides['Date of Travel'], categories=weekDays, ordered=True)
dailyRides.sort_values(by='Date of Travel', inplace=True)
dailyRides = dailyRides.groupby(['Company', pd.Grouper(key='Date of Travel')]).mean()
dailyRides.rename(columns={'Transaction ID': 'Number of Rides'}, inplace= True)

dailyRides.unstack(level='Company').plot.bar(subplots=True, rot= 0)

#Customer Retention
repeatPurchase = yearlyData.groupby(['Company', 'Date of Travel', pd.Grouper(key='Customer ID')])['Transaction ID'].count()
#repeatPurchase.rename(columns={'Transaction ID': 'NumOfTrips'}, inplace=True)

fig, ax10 = plt.subplots()
ax10.boxplot(repeatPurchase.loc[('Pink Cab', '2016-12-31')], positions= [-0.4], showfliers= False, patch_artist=True, boxprops=dict(facecolor='pink'))
ax10.boxplot(repeatPurchase.loc[('Pink Cab', '2017-12-31')], positions= [1.6], showfliers= False, patch_artist=True, boxprops=dict(facecolor='pink'))
ax10.boxplot(repeatPurchase.loc[('Pink Cab', '2018-12-31')], positions= [3.6], showfliers= False, patch_artist=True, boxprops=dict(facecolor='pink'))

ax10.boxplot(repeatPurchase.loc[('Yellow Cab', '2016-12-31')], positions= [0], showfliers= False, patch_artist=True, boxprops=dict(facecolor='yellow'))
ax10.boxplot(repeatPurchase.loc[('Yellow Cab', '2017-12-31')], positions= [2], showfliers= False, patch_artist=True, boxprops=dict(facecolor='yellow'))
ax10.boxplot(repeatPurchase.loc[('Yellow Cab', '2018-12-31')], positions= [4], showfliers= False, patch_artist=True, boxprops=dict(facecolor='yellow'))

ax10.set_xticks([-0.4+(0+0.4)/2, 1.6+0.5*(2-1.6), 3.6+0.5*(4-3.6)])
ax10.set_xticklabels(['2016', '2017', '2018'])
ax10.set_title('Number of Repeated Customer purchases')
ax10.set_xlabel('Year')
ax10.legend([Line2D([0], [0], color='pink', lw=4), Line2D([0], [0], color='yellow', lw=4)], ['Pink Cab', 'Yellow Cab'])
