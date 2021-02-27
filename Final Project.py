#!/usr/bin/env python
# coding: utf-8

# # Inequality in house prices by racial profile in USA

# Author: Michael Gorjaltsan 321031890 gorjalts@post.bgu.ac.il
# 
# ## Abstract
# 
# ### Motivation
#  
# 
# ### Goal
# 
# 
# 
# 
# ## Methodology
# 
# ### Datasets
# 
# 
# ### Assumptions
# 
# 
# ### Research Flow
# 
# 

# # The Code

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path
import json
from urllib.request import urlopen
import seaborn as sns
import matplotlib.pyplot as plt
# import missingno as msno
from scipy import stats

# import plotly
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# # Preprocess

# In[2]:


api_token = {"username":"michaelgorjaltsan","key":"e4a23a3c05c3399b70161684e079b2bd"}
get_ipython().system('mkdir ~/.kaggle')

with open(Path(f'{str(Path.home())}/.kaggle/kaggle.json'), 'w') as file:
    json.dump(api_token, file)
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# ## Census Data

# לקחתי את המידע הגולמי מהצנזוס של ארהב  בין השנים 2010-2019 ברמת הקאונטי

# In[3]:


# for more details about ACS https://www.census.gov/programs-surveys/acs


# ### Census (ACS) 2000-2010

# In[4]:


#! wget https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-sexracehisp.csv
acs = pd.read_csv('co-est00int-sexracehisp.csv' ,sep=',', encoding='latin-1')
acs.head(3)


# A description of the data can be found here: <br>
# https://www2.census.gov/programs-surveys/popest/datasets/2010-2011/counties/asrh/cc-est2011-6race.pdf

# In[5]:


acs['SEX'] = acs['SEX'].replace({0:'Total',1:'Male',2:'Female'})
acs['ORIGIN'] = acs['ORIGIN'].replace({0:'Total',1:'Not Hispanic',2:'Hispanic'})
acs['RACE'] = acs['RACE'].replace({0:'Total',1:'White',2:'Black',})
year_cols = [col for col in acs.columns if 'POPESTIMATE' in col]
for col in year_cols:
    acs.rename(columns={col:int(col[-4:])},  inplace=True)
acs.head(3)


# In[6]:


#נסנן מה שרלוונטי
# נשאיר רק לבן שחור והשאר 
acs = acs[(acs['ORIGIN']=='Total')&(acs['SEX']=='Total')&(acs['RACE'].isin(['Total','White','Black']))]

acs['CTYNAME'] = acs['CTYNAME'].str[:-7]
acs.reset_index(drop=True, inplace=True)

acs['State FIPS Code'] = acs['STATE'].apply(lambda x: str(x).zfill(2))
acs['County FIPS Code'] = acs['COUNTY'].apply(lambda x: str(x).zfill(3))
acs['FIPS'] = acs['State FIPS Code'] + acs['County FIPS Code']

acs.rename(columns={'STNAME':'state','CTYNAME':'county',}, inplace=True)
acs_00_10 = acs[['FIPS','state','county','RACE']+list(range(2000,2011))]
acs_00_10.head(3)


# In[7]:


# info about FIPS https://en.wikipedia.org/wiki/FIPS_county_code


# In[ ]:





# In[8]:


acs_00_10_piv = acs_00_10.pivot(
               values=list(range(2000,2011)),
               index='FIPS',
                    columns='RACE').T.unstack(level=0).T.reset_index().rename(columns={'level_1':'year'})

acs_00_10_piv['white_ratio']= (acs_00_10_piv['White'] / acs_00_10_piv['Total']).round(4)
acs_00_10_piv['black_ratio']= (acs_00_10_piv['Black'] / acs_00_10_piv['Total']).round(4)
acs_00_10_piv['others']= 1-(acs_00_10_piv['white_ratio']+acs_00_10_piv['black_ratio'])

acs_00_10_final = acs_00_10.groupby('FIPS').first()[['state','county']].reset_index().merge(acs_00_10_piv,on='FIPS')
acs_00_10_final.head()


# ## Census (ACS) 2010-2019

# In[9]:


# ! wget https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/asrh/cc-est2019-alldata.csv
acs = pd.read_csv('cc-est2019-alldata.csv' ,sep=',', encoding='latin-1')
acs.head(3)


# A description of the data can be found here: <br>
# https://www2.census.gov/programs-surveys/popest/datasets/2010-2017/counties/asrh/cc-est2017-alldata.pdf

# In[10]:


acs['AGEGRP'] = acs['AGEGRP'].replace(0, 'Total')
acs['YEAR'] = acs['YEAR'].replace({3: 2010, 4: 2011,
                    5: 2012, 6: 2013,
                    7: 2014, 8: 2015,
                    9: 2016, 10: 2017,
                    11: 2018, 12: 2019,
                    })

#נסתכל על כולם ולא לפי קבוצות גלאים
acs = acs[(acs['AGEGRP']=='Total')&(acs['YEAR'].isin(range(2010,2020)))]
acs['CTYNAME'] = acs['CTYNAME'].str[:-7]
acs.reset_index(drop=True, inplace=True)

acs['State FIPS Code'] = acs['STATE'].apply(lambda x: str(x).zfill(2))
acs['County FIPS Code'] = acs['COUNTY'].apply(lambda x: str(x).zfill(3))
acs['FIPS'] = acs['State FIPS Code'] + acs['County FIPS Code']

acs.rename(columns={'STNAME':'state','CTYNAME':'county', 'YEAR':'year','TOT_POP':'Total'}, inplace=True)
acs.head(3)


# In[11]:


acs['White'] = acs['WA_MALE'] + acs['WA_FEMALE']
acs['white_ratio']= (acs['White'] / acs['Total']).round(4)

acs['Black']= acs['BA_MALE'] + acs['BA_FEMALE']
acs['black_ratio']= (acs['Black'] / acs['Total']).round(4)

acs['others']= 1-(acs['white_ratio']+acs['black_ratio'])

for col in ['white_ratio','black_ratio','others']:
    acs[col] = acs[col].round(4)

acs_10_19= acs[['FIPS','state','county','year','Total','White','Black','white_ratio','black_ratio','others']]
acs_10_19.head(3)


# ### Census (ACS) 2000-2019

# In[12]:


acs_00_09 =acs_00_10_final[acs_00_10_final['year']!=2010]
acs_full = pd.concat([acs_10_19,acs_00_09]).sort_values(['FIPS','year']).reset_index(drop=True)
acs_full.iloc[8:22]


# In[13]:


acs_by_year = acs_full.groupby('year')
us_white_ratio_by_year = acs_by_year['White'].sum()/ acs_by_year['Total'].sum() *100 

fig1=px.bar(x=us_white_ratio_by_year.index, y =us_white_ratio_by_year, range_y=[70,82],
       title='Toatl US White Ratio by Year' ,
      labels={'x':'Year', 'y':"White Ratio %"},color_discrete_sequence=px.colors.qualitative.Pastel)


us_b_ratio_by_year = acs_by_year['Black'].sum()/ acs_by_year['Total'].sum() *100 

fig2  = px.bar(x=us_b_ratio_by_year.index, y =us_b_ratio_by_year,range_y=[10,14], 
       title='Toatl US Black Ratio by Year' ,
      labels={'x':'Year', 'y':"Black Ratio %"},color_discrete_sequence=px.colors.qualitative.Antique)


 
fig1.show()
fig2.show()


# In[14]:


fig1 = px.histogram(acs_full, x='state', y='white_ratio' ,
             color='state', animation_frame='year',histfunc='avg', color_discrete_sequence=px.colors.qualitative.Dark24,
             title='White Ratio by State', 
             labels={'white_ratio':'White Ratio',},)

fig2 = px.histogram(acs_full, x='state', y='black_ratio' ,
             color='state', animation_frame='year',histfunc='avg',color_discrete_sequence=px.colors.qualitative.Dark24, 
             title='Black Ratio by State', 
             labels={'black_ratio':'Black Ratio',},)

fig1.show()
fig2.show()


# In[15]:


acs_piv = (acs_full.pivot(index='FIPS', columns='year', values=['white_ratio','black_ratio',])*100).round(2)
acs_piv.reset_index(inplace=True)
acs_piv=acs_piv.merge(acs_full[['FIPS','state','county']], on='FIPS')
acs_piv.head(3)


# In[16]:


with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(
                    geojson=counties, locations=acs_piv['FIPS'], color=acs_piv[('white_ratio',2019)],
                           color_continuous_scale=px.colors.sequential.Plasma,
                            labels={'color': 'White Ratio 2019'},
                           range_color=(0, 100),
                           scope="usa",
                            hover_data={'state':acs_piv['state'],'county':acs_piv['county'],
                                       'White Ratio 2000':acs_piv[('white_ratio',2000)].map('{:.2f}%'.format),
                                        'White Ratio 2019':acs_piv[('white_ratio',2019)].map('{:.2f}%'.format),
                                       'Black Ratio 2000':acs_piv[('black_ratio',2000)].map('{:.2f}%'.format),
                                       'Black Ratio 2019':acs_piv[('black_ratio',2019)].map('{:.2f}%'.format),},
                    title='White Ratio by County (2019)',
                          )
fig.update_layout(margin={"r":0,"l":0,"b":0})
fig.show()


# ## Zillow Data 

# ### Preprocess

# In[17]:


# ! kaggle datasets download -d zillow/zecon
# !unzip zecon.zip


# ZHVI methodology <br>
# https://www.zillow.com/research/zhvi-methodology-2019-deep-26226/<br>
# <br>
# ZRI methodology <br>
# https://www.zillow.com/research/zillow-rent-index-methodology-2393/

# In[18]:


zillow = pd.read_csv('County_time_series.csv')
zillow['Date'] = pd.to_datetime(zillow['Date'])
zillow.rename(columns={'RegionName':'FIPS'}, inplace=True)
zillow['FIPS'] = zillow['FIPS'].apply(lambda x: str(x).zfill(5))
zillow.tail(3)


# as we can see there are a lot of missing values, let's check it in more depth

# In[19]:


zillow_isna = zillow.isna().mean().sort_values()*100
plt.figure(figsize=(15,15))
sns.barplot(x=zillow_isna.values, y=zillow_isna.index, color='blue',)
plt.grid()
plt.xlabel('%')
plt.title('Zillow Is nan dist.', size=20)


# ### Handeling missing values

# In[20]:


pd.Series(zillow['Date'].unique()).head(10)


# In[21]:


# אפשר לראות שיש המידע מוגש בסוף כל חודש, אז אלי על ידי איחוד של מספר חודשים נצליח להשלים את הדאטה


# In[22]:


pd.Series(zillow['Date'].unique()).dt.year.value_counts(sort=False)


# In[23]:


pd.Series(zillow['Date'].unique()).dt.month.value_counts(sort=False)


# אפשר לראות הדאטהסט הזה מדווחים כל חודש , אנסה למלא את הערכים החסרים על ידי הסתכלות על ממוצע הדיווחים לכל שנה.

# In[24]:


zillow['year'] = zillow['Date'].dt.year
zillow_by_year = zillow.groupby(['year','FIPS']).mean().reset_index()
zillow_by_year.head()


# In[25]:


#sanity check
zillow_by_year['FIPS'].nunique() == zillow['FIPS'].nunique()


# In[26]:


zillow_by_year_isna = zillow_by_year.isna().mean().sort_values()*100
plt.figure(figsize=(15,15))
sns.barplot(x=zillow_isna.values, y=zillow_isna.index, color='red',label='Before')
sns.barplot(x=zillow_by_year_isna.values, y=zillow_by_year_isna.index, color='blue',label='After')
plt.grid()
plt.xlabel('%')
plt.title('Zillow Is nan dist.', size=20)
plt.legend()


# נראה שעדיין יש מידע חסר
# בואו נסתכל ברמת הקאונטי לראות כמה מידע חסר בכל קאונטי לאורך השנים 
# כמו כן נסתכל על השנים הרלוונטיות לעבודה (אחרי שנת 2000( 

# In[27]:



notna_table = zillow_by_year[zillow_by_year['year']>=2000].drop('FIPS', 1).notna().groupby(zillow_by_year['FIPS'], sort=False).mean().reset_index()
plt.figure(figsize=(20,10))

sns.heatmap(notna_table.drop(columns='FIPS'), )
plt.title('County NotNa 2000-2017 Heatmap', size=15)


# ניתן לראות שיש מחוזות שבעבורם כנראה לא נאסף מידע מסויים כמו מדיאן רנטל פרייס
# מנגד הזי אר אי נאסף טוב באותם הקאונטיז 
# כמו כן ניתן לראות שה זי אטץ׳ וי אי נאסף טוב ברוב המדינות.

# ### additional Zillow dataset

# בדאטה הזה אנחנו מציגים את מחיר הממוצע של דירות לפי רגל מרובעת.

# In[28]:


# ZHVI = Zillow Home Value Index 


# In[29]:


# from https://data.world/zillow-data/median-home-value-per-sq-ft
zillow2 = pd.read_csv('https://query.data.world/s/c6r5cc373k6edachvjqsvk4ur4ue5d')
zillow2['MunicipalCodeFIPS'] = zillow2['MunicipalCodeFIPS'].apply(lambda x: str(x).zfill(3))
zillow2['FIPS'] = (zillow2['StateCodeFIPS'].astype(str) + zillow2['MunicipalCodeFIPS']).apply(lambda x: str(x).zfill(5))
zillow2 = zillow2.drop(columns = ['RegionName','State','Metro','StateCodeFIPS','MunicipalCodeFIPS','SizeRank','RegionID'])
zillow2.head(3)


# In[35]:


zillow2_piv = zillow2.set_index('FIPS').unstack(level=1).to_frame().reset_index().rename(columns={0:'ZHVI_per_sqft', 'level_0':'date'})
zillow2_piv['year'] = pd.to_datetime(zillow2_piv['date']).dt.year
zillow2_per_year = zillow2_piv.groupby(['year','FIPS']).mean().reset_index()
zillow2_per_year.head(3)


# In[52]:


isna_table['ZHVI_per_sqft'].map('{:.1%}'.format)


# In[56]:


isna_table = zillow2_per_year.drop('year',axis=1).isna().groupby(zillow2_per_year['year'], sort=False).mean().drop(columns='FIPS')
px.bar(x=isna_table.index, y =(isna_table['ZHVI_per_sqft']*100).round(2),
       title='Zillow Is NaN by Year' ,
      labels={'x':'Year', 'y':"Isna %"},color_discrete_sequence=px.colors.qualitative.Antique)


# In[68]:


# isna_table = zillow2_per_year.drop('FIPS',axis=1).isna().groupby(zillow2_per_year['FIPS'], sort=False).mean().drop(columns='year')
# missing features
sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

plt.figure(figsize=(20,10))
sns.heatmap(zillow2.notna(), cbar=False)
plt.title('Notna Heatmap', size=20)


# אפשר לראות שיש קאונטי שלא מדווח עליהם לאורך שנים.. יש פה ושם נקודות שחורות אבל שוב אם אסתכל על הממוצע של על השנה זה יעלם (כלומר ברגע שהתחילו לדווח/לאסוף מידע לא היה שנה רצופה ללא נתונים)

# ### Zillow 1+2

# קודמת נעשה סנטי צ׳ק שזה בכלל דומה

# In[69]:


zillow_by_year_merged = zillow_by_year.merge(zillow2_per_year.rename(columns={'ZHVI_per_sqft':'ZHVI_per_sqft(other data)'}), on=['year','FIPS'], how='left')
zillow_by_year_merged[['year','FIPS','ZHVIPerSqft_AllHomes','ZHVI_per_sqft(other data)']].sample(5)


# In[70]:


zillow_by_year_merged.shape, zillow_by_year.shape, zillow2_per_year.shape,


# #### sanity check

# In[71]:


# drop Fips with both Nan values
merged_zillow_w_ZHVI = zillow_by_year_merged[zillow_by_year_merged[['ZHVIPerSqft_AllHomes','ZHVI_per_sqft(other data)']].notna().any(axis=1)].reset_index(drop=True)

# check if the value are close, 
# checks if out of existing values (in both DataFrames) the relative diff is greater than {diff}%.
rel_diff = (abs(merged_zillow_w_ZHVI['ZHVIPerSqft_AllHomes'] - merged_zillow_w_ZHVI['ZHVI_per_sqft(other data)']) 
 / merged_zillow_w_ZHVI['ZHVIPerSqft_AllHomes'] )

for diff in [0.01, 0.02, 0.05]:
    print(f"only {round((rel_diff> diff).mean()*100,2)}% have more than {diff*100}% diff")


# נסתכל על החריגים

# In[72]:


rel_diff.sort_values(ascending=False).head()


# בגלל שאני הולך להשתמש במידע הראשון של זילו כמקור ובשני רק להשלמות נתרכז באותם קאונטי שאנחנו הולכים להשיג
# ננסה להבין האם יש פערים חריגים בין 2 מקורות הדאטה עבור הקרונטיס האלה.. כיוון שאם יש פער באחד השנים זה יכול להשפיע על ההשלמה של הערכים

# In[73]:


fips_of_miss_val = merged_zillow_w_ZHVI[(merged_zillow_w_ZHVI['ZHVIPerSqft_AllHomes'].isna())&
(merged_zillow_w_ZHVI['ZHVI_per_sqft(other data)']).notna()]['FIPS'].unique()

org_zillow_wo_ZHVI = merged_zillow_w_ZHVI[merged_zillow_w_ZHVI['FIPS'].isin(fips_of_miss_val)]


# In[74]:


((org_zillow_wo_ZHVI['ZHVIPerSqft_AllHomes'] - org_zillow_wo_ZHVI['ZHVI_per_sqft(other data)'])
/org_zillow_wo_ZHVI['ZHVIPerSqft_AllHomes']).abs().sort_values(ascending=False).head()


# ניתן לראות שלא קיים הבדל משמעותי לכן אפשר להשלים את המידע

# In[75]:


# אחרי שראינו שאין בעיה נוכל להשלים את הדאטה
zillow_by_year_merged['ZHVI_per_Sqft'] = zillow_by_year_merged['ZHVIPerSqft_AllHomes'].fillna(zillow_by_year_merged['ZHVI_per_sqft(other data)'])


# In[76]:


zillow_by_year_merged.groupby('year')['ZHVI_per_Sqft'].mean().plot(kind='bar', figsize=(10, 6), grid=True)
plt.title('Avg (not weighted) ZHVI per Sqft',size=20)
plt.ylabel('Avg ZHVI Per Sqft')


# # Zillow + ACS

# יוצא מנקודת הנחה שכמות הבתים פרופורציונלית לכמות האנשים

# In[77]:


#נבחר את המחיר הממוצע ואת המחיר הכללי כיוון שאחוז הנאנים בשניהם יחסית קטן
zillow_relevant = zillow_by_year_merged[zillow_by_year_merged['year']>=2000][['FIPS','year','ZHVI_per_Sqft','ZHVI_AllHomes']].rename(columns={'ZHVI_AllHomes':'ZHVI'})
zillow_relevant.head()


# In[78]:


acs_zillow = acs_full.merge(zillow_relevant, on=['FIPS','year'], how='left')
acs_zillow.head()


# In[79]:


px.histogram(acs_zillow[acs_zillow['year']<=2017], x='state', y='ZHVI_per_Sqft' ,
             color='state', animation_frame='year',histfunc='avg', color_discrete_sequence=px.colors.qualitative.Dark24,
             title='ZHVI per Sqft by State',range_y=[0,425] ,
             labels={'ZHVI_per_Sqft':'ZHVI per Sqft',},)


# In[80]:


px.histogram(acs_zillow[acs_zillow['year']<=2017], x='state', y='ZHVI' ,
             color='state', animation_frame='year',histfunc='avg', color_discrete_sequence=px.colors.qualitative.Dark24,
             title='ZHVI by State',range_y=[0,5.5e5])


# In[90]:


acs_zillow_piv = (acs_zillow.pivot(index='FIPS', columns='year', values=['white_ratio','black_ratio','ZHVI_per_Sqft','ZHVI', 'Total'])).round(4)
acs_zillow_piv.reset_index(inplace=True)
acs_zillow_piv=acs_zillow_piv.merge(acs_zillow.groupby('FIPS').first().reset_index()[['FIPS','state','county']], on='FIPS')
acs_zillow_piv=acs_zillow_piv[(acs_zillow_piv[('ZHVI_per_Sqft',2017)].notna())|(acs_zillow_piv[('ZHVI_per_Sqft',2010)].notna())]


# In[91]:


with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(
                    geojson=counties, locations=acs_zillow_piv['FIPS'], color=acs_zillow_piv[('white_ratio',2017)]*100,
                           color_continuous_scale=px.colors.sequential.Plasma,
                            labels={'color': 'White Ratio 2017'},
                           range_color=(0, 100),
                           scope="usa",
                            hover_data={'state':acs_zillow_piv['state'],
                                        'county':acs_zillow_piv['county'],
                                       'White Ratio 2000':acs_zillow_piv[('white_ratio',2000)].map('{:.2%}'.format),
                                        'White Ratio 2017':acs_zillow_piv[('white_ratio',2017)].map('{:.2%}'.format),
                                       'Black Ratio 2000': acs_zillow_piv[('black_ratio',2000)].map('{:.2%}'.format),
                                       'Black Ratio 2017': acs_zillow_piv[('black_ratio',2017)].map('{:.2%}'.format),
                                       'ZHVI per Sqft 2000':acs_zillow_piv[('ZHVI_per_Sqft',2000)].fillna(' X'),
                                       'ZHVI per Sqft 2017':acs_zillow_piv[('ZHVI_per_Sqft',2017)].fillna(' X'),}
    ,
                    title='ZHVI and Recial dist by County (2017)',
                          )
fig.update_layout(margin={"r":0,"l":0,"b":0})
fig.show()


# In[92]:


# האם חש קשר בין הדיווח לרמת הצבע?


# נראה שיש לחלק ניכר מהקאונטי אין דיווח, בואו נראה האם יש קשר לריכוז לבנים בשביל זה נחזור צעד אחורה ונמרגג׳ את המידע שוב
# 

# # גזענות

# In[93]:


def relative_change(x,y):
    return ((x-y)/y).round(4)


# In[94]:


acs_zillow_piv['white_change'] = relative_change(acs_zillow_piv[('white_ratio',2017)], acs_zillow_piv[('white_ratio',2000)])
acs_zillow_piv['black_change'] = relative_change(acs_zillow_piv[('black_ratio',2017)], acs_zillow_piv[('black_ratio',2000)])
acs_zillow_piv['zhvi_change'] = relative_change(acs_zillow_piv[('ZHVI_per_Sqft',2017)], acs_zillow_piv[('ZHVI_per_Sqft',2000)])


# In[95]:


fig = px.choropleth(
                    geojson=counties, locations=acs_zillow_piv['FIPS'], color=acs_zillow_piv['white_change']*100,
                           color_continuous_scale=px.colors.sequential.Plasma,
                            labels={'color': 'White Change since 2000'},
                           range_color=(-20, 20),
                           scope="usa",
                            hover_data={'state':acs_zillow_piv['state'],
                                        'county':acs_zillow_piv['county'],
                                       'White Change since 2000':acs_zillow_piv['white_change'].map('{:.2%}'.format),
                                       'Black Change since 2000': acs_zillow_piv['black_change'].map('{:.2%}'.format),
                                       'ZHVI Change since 2000':acs_zillow_piv['zhvi_change'].map('{:.2%}'.format).fillna(' X'),}
    ,
                    title='ZHVI and Recial change by County (2017)',
                          )
fig.update_layout(margin={"r":0,"l":0,"b":0})
fig.show()


# In[96]:


fig = px.choropleth(
                    geojson=counties, locations=acs_zillow_piv['FIPS'], color=acs_zillow_piv['black_change']*100,
                           color_continuous_scale=px.colors.sequential.Plasma,
                            labels={'color': 'Black Change since 2000'},
                           range_color=(-20, 20),
                           scope="usa",
                            hover_data={'state':acs_zillow_piv['state'],
                                        'county':acs_zillow_piv['county'],
                                       'White Change since 2000':acs_zillow_piv['white_change'].map('{:.1%}'.format),
                                       'Black Change since 2000': acs_zillow_piv['black_change'].map('{:.1%}'.format),
                                       'ZHVI Change since 2000':acs_zillow_piv['zhvi_change'].map('{:.1%}'.format).fillna(' X'),}
    ,
                    title='ZHVI and Recial change by County (2017)',
                          )
fig.update_layout(margin={"r":0,"l":0,"b":0})
fig.show()


# ניתן לראות שקיימים שינויים באוכלוסיית ארה״ב, בואו נבדוק האם יש שוני

# תחילה נדבדוק האם יש קשר בין בעיית הדיווח לתמהיל האנשים בקאונטי

# In[97]:


report_df =pd.DataFrame()

reported_fips = acs_full[acs_full['FIPS'].isin(acs_zillow_piv['FIPS'])]
miss_fips = acs_full[~acs_full['FIPS'].isin(acs_zillow_piv['FIPS'])]

for race in ['Black','White']:
    for df, df_name in zip([reported_fips, miss_fips], ['Reported', 'Miss']):
        for year in range(2000,2018):
            rel_df = df[df['year']==year]
            report_df.loc[year, f'{race} % {df_name} Fips'] = '{:.2%}'.format(rel_df[race].sum()/rel_df['Total'].sum())
report_df


# ניתן לראות בבירור כי אין אפליה בדיווח ואולי אפילו אפשר להסיק כי קיימת אפליה נגד לבנים

# In[ ]:





# In[ ]:





# In[98]:


#חישוב הממוצע של כל ארהב לפי משקול של האוכלוסיה לכל קאונטי
for year in range(2000,2018):
    selected_year = acs_zillow[acs_zillow['year']==year]
    for col in ['white_ratio','black_ratio','ZHVI_per_Sqft','ZHVI']:
        acs_zillow_piv.loc['usa_avg',(col,year)] = round((selected_year[col]*selected_year['Total']).sum() / (selected_year[col].notna() * selected_year['Total']).sum(),4)   
        


# In[99]:


for delta in range(1,18):
    for year in range(2000,2018-delta):
        acs_zillow_piv[f'white_change {year}-{year+delta}'] = relative_change(acs_zillow_piv[('white_ratio',year+delta)], acs_zillow_piv[('white_ratio',year)])
        acs_zillow_piv[f'black_change {year}-{year+delta}'] = relative_change(acs_zillow_piv[('black_ratio',year+delta)], acs_zillow_piv[('black_ratio',year)])
        acs_zillow_piv[f'zhvi_change {year}-{year+delta}'] = relative_change(acs_zillow_piv[('ZHVI_per_Sqft',year+delta)], acs_zillow_piv[('ZHVI_per_Sqft',year)])


# In[108]:


def impute_to_impact_table(impact_df, change_col, change_th,impact_col, result_df,location,weighted_col=None):
            
        bigger_change = impact_df[impact_df[change_col]>change_th ]
        smaller_change= impact_df[impact_df[change_col]<change_th ]
        
        if weighted_col is not None:
            df.loc[location, 'bigger_change'] = (bigger_change[impact_col]*bigger_change[weighted_col]).sum() / (bigger_change[impact_col].notna()*bigger_change[weighted_col]).sum() 
            df.loc[location, 'smaller_change'] = (smaller_change[impact_col]*smaller_change[weighted_col]).sum() / (smaller_change[impact_col].notna()*smaller_change[weighted_col]).sum() 
        else:   
            df.loc[location, 'bigger_change'] = bigger_change[impact_col].mean()
            df.loc[location, 'smaller_change'] = smaller_change[impact_col].mean()
        df.loc[location, 'pvalue'] = stats.ttest_ind( bigger_change[impact_col], smaller_change[impact_col], nan_policy='omit')[1]
        df.loc[location, 'sagnificant'] =  df.loc[location, 'pvalue'] <=0.05


# In[109]:


df = pd.DataFrame()
for delta in range(1,18):
    for year in range(2000,2018-delta):
        change_col = f'white_change {year}-{year+delta}'
        
        impute_to_impact_table(impact_df=acs_zillow_piv,
                     change_col= change_col,
                     change_th = acs_zillow_piv.loc['usa_avg',change_col] ,
                     impact_col=('ZHVI',year+delta),
                     result_df= df,
                     location=f'{year}-{year+delta}',
                     weighted_col =('Total',year+delta),)

df


# In[116]:


df = pd.DataFrame()
for delta in range(1,18):
    for year in range(2000,2018-delta):
        change_col = f'white_change {year}-{year+delta}'
        
        impute_to_impact_table(impact_df=acs_zillow_piv,
                     change_col= change_col,
                     change_th = acs_zillow_piv.loc['usa_avg',change_col] ,
                     impact_col=f'zhvi_change {year}-{year+delta}',
                     result_df= df,
                     location=f'{year}-{year+delta}',
                     weighted_col =('Total',year+delta),)

df


# In[967]:


df = pd.DataFrame()
for k in range(3):
    for j in range(2):
        for i in range(1,17-j-k):
            for x in range(2000,2018-i-j-k):
                change_col = f'white_change {x}-{x+i}'

                impute_to_impact_table(impact_df=acs_zillow_piv,
                             change_col= change_col,
                             change_th = acs_zillow_piv.loc['usa_avg',change_col] ,
                             impact_col=f'zhvi_change {x+i+k-1}-{x+i+j+k}',
                             result_df= df,
                             location=f'{x}-{x+i} for {x+i+k-1}-{x+i+j+k}')

df


# In[733]:


#נסתכל אם שינוי בגזע השפיע על השינוי במחירי הדיור בין
# 2000-2017


# ניתן לראות שקאונטיז עם צמיחה בשיעור הלבנים מתומחרות גבוה יותר
# אבל נראה שלא ניתן להשליח באופן מובהק של שינוי המחירים לא משנה כמה שנים מסתכלים

# In[117]:


acs_zillow_piv.head()


# ננסה לבנות מודל משין לירנינג שינסה לחזות את השינוי בין 2016 ל 2017 לפי המידע של ה 5 שנים לפניו

# In[ ]:


for year in range(2005,2016):
    label_col = f'zhvi_change {year}-{year+1}'
    


# In[402]:


l =[]
max_y=5
for year in range(2000+max_y,2017):
    df = pd.DataFrame()
    for ago in range(max_y,0,-1):
        for delta in range(1,ago):
#             white_change_usa_avg = acs_zillow_piv.loc['usa_avg',f'white_change {year-ago}-{year-ago+delta}']
            df[f'white_change {ago}y-{ago-delta}y ago'] = acs_zillow_piv[f'white_change {year-ago}-{year-ago+delta}']#-white_change_usa_avg
            zhvi_change_usa_avg = acs_zillow_piv.loc['usa_avg',f'zhvi_change {year-ago}-{year-ago+delta}']
            df[f'zhvi_change {ago}y-{ago-delta}y ago'] = (acs_zillow_piv[f'zhvi_change {year-ago}-{year-ago+delta}']-zhvi_change_usa_avg).replace(np.nan, 0)
    
    df['label'] = acs_zillow_piv[f'zhvi_change {year}-{year+1}']# - acs_zillow_piv.loc['usa_avg',f'zhvi_change {year}-{year+1}']
    l.append(df)


# In[403]:


train_5y_hist = pd.concat(l[:-1])
train_5y_hist = train_5y_hist[train_5y_hist['label'].notna()]
test_5y_hist = l[-1]
test_5y_hist = test_5y_hist[test_5y_hist['label'].notna()]
train_5y_hist.shape, test_5y_hist.shape


# In[404]:


train_5y_hist.head()


# In[424]:


import turicreate as tc

# Load the data
train_data =  tc.SFrame(train_5y_hist)
test_data =  tc.SFrame(test_5y_hist)

# Automatically picks the right model based on your data.
model = tc.boosted_trees_regression.create(train_data, target='label',max_iterations=500,max_depth=15,step_size=0.1)


# In[425]:


# Save predictions to an SArray
predictions = model.predict(test_data)
# Evaluate the model and save the results into a dictionary
results = model.evaluate(test_data)


# In[426]:


results


# In[427]:


model.get_feature_importance().print_rows(30)


# # Classifier

# In[ ]:


# Load the data
train_5y_hist['is_increased'] = (train_5y_hist['label']>0).astype(int)
test_5y_hist['is_increased'] = (test_5y_hist['label']>0).astype(int)

train_data_class =  tc.SFrame(train_5y_hist.drop(columns='label'))
test_data_class =  tc.SFrame(test_5y_hist.drop(columns='label'))

# Automatically picks the right model based on your data.
cls_model = tc.boosted_trees_classifier.create(train_data_class, target='is_increased',
                                               max_iterations=200,   max_depth=15, step_size=0.15)


# In[429]:


# Save predictions to an SArray
predictions = cls_model.predict(test_data_class)

# Evaluate the model and save the results into a dictionary
cls_results = cls_model.evaluate(test_data_class)
cls_results


# In[430]:


plt.style.use('ggplot')
plt.plot(cls_results['roc_curve']['fpr'],
         cls_results['roc_curve']['tpr'],
         label='ROC Curve (AUC = %0.5f)' %cls_results['auc'])

plt.xlim([-0.02, 1.0])
plt.ylim([0., 1.02])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend()
plt.show()


# In[411]:


cls_model.get_feature_importance().print_rows(30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




