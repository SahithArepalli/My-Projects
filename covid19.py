# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:18:46 2020

@author: HP
"""

import pandas as pd
import numpy as np
import requests
import re
from urllib.request import Request, urlopen

req = Request('https://www.worldometers.info/coronavirus/', headers={'User-Agent': 'Firefox/75.0'})
webpage = re.sub(r'<.*?>', lambda g: g.group(0).upper(), urlopen(req).read().decode('utf-8') )
tables = pd.read_html(webpage)

df = tables[1]
df1.info()
df.isnull().sum()
df = df.drop('#',axis=1)

from pandas.api.types import is_string_dtype
for col in df.columns:
        if is_string_dtype(df[col]):
            df[col] = df[col].str.replace('[^A-Za-z0-9-\s]+', '')             

df1 = df.drop(df.index[-8:])

df1 = df1.fillna(0)

df1['NewCases'] = df1['NewCases'].astype(int)
df1['NewDeaths'] = df1['NewDeaths'].astype(int)
df1['NewRecovered'] = df1['NewRecovered'].astype(int)
df1['ActiveCases'] = df1['ActiveCases'].astype(int)
df1['Serious,Critical'] = df1['Serious,Critical'].astype(int)
df1['TotalDeaths'] = df1['TotalDeaths'].astype(int)
df1['TotalRecovered'] = df1['TotalRecovered'].astype(int)
df1['TotalTests'] = df1['TotalTests'].astype(int) 

df1[['TotalCases','TotalDeaths','TotalRecovered', 'ActiveCases','Serious,Critical','TotalTests']].sum()

df2 = df1[['Country,Other','NewCases', 'NewDeaths']]

df2['Date_Time'] = pd.to_datetime('today')+ pd.DateOffset(-1)

df3 = pd.read_csv('https://raw.githubusercontent.com/valmetisrinivas/Covid19_Worldometers/master/worldometers_covid19_uptoMay2nd2020.csv')
df3['Date_Time'] = pd.to_datetime(df3['Date_Time'])

if df3['Date_Time'].dt.date.max() < df2['Date_Time'].dt.date.min():
    daily_data = df2.append(df3)
    daily_data.to_csv('data', sep =',', index=False)
else:
    daily_data = df3.copy()

daily_data['Date'] = pd.to_datetime(daily_data['Date_Time'].dt.date)
display(daily_data.shape)

daily_data['NewCases'] = daily_data['NewCases'].fillna(0).astype(int)
daily_data['NewDeaths'] = pd.to_numeric(daily_data['NewDeaths']).fillna(0).astype(int)
daily_data['NewRecovered'] = pd.to_numeric(daily_data['NewRecovered']).fillna(0).astype(int)

date_newcase = daily_data.sort_values(['Date', 'NewCases'], ascending=False).drop('Date_Time', axis=1)

date_newdeaths = daily_data.sort_values(['Date', 'NewDeaths'], ascending=False).drop('Date_Time', axis=1).head(30)

cum_data = df1.drop(columns=['NewCases','NewDeaths'])
cum_data['Dead_to_Recovered'] = 100*cum_data['TotalDeaths']/cum_data['TotalRecovered']
cum_data = cum_data.sort_values('TotalCases', ascending=False)
cum_data['TotalCases_Per'] = 100*cum_data['TotalCases']/cum_data['TotalCases'].sum()
cum_data['TotalDeaths_Per'] = 100*cum_data['TotalDeaths']/cum_data['TotalDeaths'].sum()
cum_data['TotalRecovered_Per'] = 100*cum_data['TotalRecovered']/cum_data['TotalRecovered'].sum()
cum_data['TotalActive_Per'] = 100*cum_data['ActiveCases']/cum_data['ActiveCases'].sum()
cum_data['TotalTests_Per'] = 100*cum_data['TotalTests']/cum_data['TotalTests'].sum()

countries = cum_data['Country,Other'][0:30]

select_con = cum_data[cum_data['Country,Other'].isin(countries)]

select_daily = daily_data[daily_data['Country,Other'].isin(countries)].drop(columns = 'Date_Time')

con_per=select_con[['Country,Other','TotalCases_Per','TotalDeaths_Per','TotalRecovered_Per','TotalActive_Per','TotalTests_Per']]

sps=pd.melt(con_per, id_vars='Country,Other',value_name='Percentage', var_name='Type')
sps['Type'] = sps['Type'].str.replace("_Percent","")

import matplotlib.pyplot as plt
from pylab import rcParams

#Total number of cases

rcParams['figure.figsize'] = 15, 5
fig, ax = plt.subplots()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.bar("Confimred", cum_data['TotalCases'].sum())
plt.text(-.1, cum_data['TotalCases'].sum() + 50000, str(cum_data['TotalCases'].sum()),fontweight='bold')

ax.bar("ActiveCases", cum_data['ActiveCases'].sum())
plt.text(-.1+1, cum_data['ActiveCases'].sum() + 50000, str(cum_data['ActiveCases'].sum()),fontweight='bold')

ax.bar("Recovered", cum_data['TotalRecovered'].sum())
plt.text(-.1+2, cum_data['TotalRecovered'].sum() + 50000, str(cum_data['TotalRecovered'].sum()),fontweight='bold')

ax.bar("Deaths", cum_data['TotalDeaths'].sum())
plt.text(-.1+3, cum_data['TotalDeaths'].sum() + 50000, str(cum_data['TotalDeaths'].sum()),fontweight='bold')

ax.bar("Serious,Critical", cum_data['Serious,Critical'].sum())
plt.text(-.1+4, cum_data['Serious,Critical'].sum() + 50000, str(cum_data['Serious,Critical'].sum()),fontweight='bold')

ax.set_ylabel("Total Numbers")
plt.show()

#No.of cases as per standard deviation

fig, ax = plt.subplots()

ax.bar("Confimred", cum_data['TotalCases'].mean(), yerr=cum_data['TotalCases'].std())

ax.bar("ActiveCases", cum_data['ActiveCases'].mean(), yerr=cum_data['ActiveCases'].std())

ax.bar("Recovered", cum_data['TotalRecovered'].mean(), yerr=cum_data['TotalRecovered'].std())

ax.bar("Deaths", cum_data['TotalDeaths'].mean(), yerr=cum_data['TotalDeaths'].std())

ax.bar("Serious,Critical", cum_data['Serious,Critical'].mean(), yerr=cum_data['Serious,Critical'].std())

ax.set_ylabel("Numbers")
plt.show()

#Tests conducted

fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5

test_data = cum_data.sort_values('Tests/ 1M pop', ascending=False)
test_data = test_data.head(30).set_index('Country,Other').sort_values('Tests/ 1M pop', ascending=False).fillna(0)

ax.bar(test_data.index,test_data['Tests/ 1M pop'])

ax.set_xticklabels(test_data.index, rotation = 100)

ax.set_ylabel("Tests Conducted/ Million")
plt.show()

#Top 30 countries

select_con1 = select_con.sort_values('TotalCases', ascending=False).set_index('Country,Other').fillna(0)

rcParams['figure.figsize'] = 15, 5
fig, ax = plt.subplots()

ax.bar(select_con1.index,select_con1['TotalCases'])

ax.set_xticklabels(select_con1.index, rotation = 90)

ax.set_ylabel("Total Confirmed Cases")
plt.show()

#Deaths occured in top 30 countries

fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5
select_con1 = select_con1.sort_values('TotalDeaths', ascending=False).fillna(0)

ax.bar(select_con1.index,select_con1['TotalDeaths'])

ax.set_xticklabels(select_con1.index, rotation = 90)

ax.set_ylabel("Total Deaths")
plt.show()

#Recovered cases of top 30 countries

fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5
select_con1 = select_con1.sort_values('TotalRecovered', ascending=False).fillna(0)

ax.bar(select_con1.index,select_con1['TotalRecovered'])

ax.set_xticklabels(select_con1.index, rotation = 90)

ax.set_ylabel("Total Recovered Cases")
plt.show()


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5
select_con1 = select_con1.sort_values('Tests/ 1M pop', ascending=False).fillna(0)

ax.bar(select_con1.index,select_con1['Tests/ 1M pop'],label='Tests Conducted')
ax.bar(select_con1.index,select_con1['Tests/ 1M pop'],bottom=select_con1['Tests/ 1M pop'],label='Confirmed')
ax.bar(select_con1.index,select_con1['Tests/ 1M pop'],bottom= select_con1['Tests/ 1M pop']+select_con1['Tot\xa0Cases/1M pop'],label='Dead')

ax.set_xticklabels(select_con1.index, rotation = 90)
ax.set_ylabel("Number of cases")

plt.legend()
plt.show()

#Number people dead for 100 people recovered - top 30 hit countries

select_con1 = select_con.set_index('Country,Other').sort_values('Dead_to_Recovered',ascending=False)

rcParams['figure.figsize'] = 15, 5

fig, ax = plt.subplots()

ax.bar(select_con1.index,select_con1['Dead_to_Recovered'])

ax.set_xticklabels(select_con1.index, rotation = 90)

ax.set_ylabel("% dead against recovered")
plt.show()

daily_data1 = daily_data.set_index('Date').fillna(0).groupby('Date').sum()
daily_data2 = daily_data1.cumsum()

def plot_timeseries(axes, x, y, color, xlabel, ylabel):
  axes.plot(x, y, color=color)
  axes.set_xlabel(xlabel)
  axes.set_ylabel(ylabel, color=color)
  axes.tick_params('y', colors=color)
  
fig, ax = plt.subplots()
plot_timeseries(ax, daily_data1.index, daily_data1['NewCases'], "blue", "Date" , "Number of confirmed cases")
plt.scatter(daily_data1.index, daily_data1['NewCases'], color='b')

ax2 = ax.twinx()

plot_timeseries(ax2, daily_data1.index, daily_data1['NewDeaths'], "red", "Date" , "Number of deaths")
plt.scatter(daily_data1.index, daily_data1['NewDeaths'], color='r')
plt.show()  
  