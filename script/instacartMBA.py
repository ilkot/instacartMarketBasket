import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import plotly.express as px
from plotly.offline import plot
#%%getData
localPath = '/Users/ilketopak/Documents/GitHub/ilkot/kaggle/instacartMarketBasket/'

#find the names
files = glob.glob(localPath+'data/*.csv')
print(files)

#read it
products = pd.read_csv(localPath+'data/products.csv')
orders = pd.read_csv(localPath+'data/orders.csv')
order_products__train = pd.read_csv(localPath+'data/order_products__train.csv')
departments = pd.read_csv(localPath+'data/departments.csv')
aisles = pd.read_csv(localPath+'data/aisles.csv')
order_products__prior = pd.read_csv(localPath+'data/order_products__prior.csv')
sample_submission = pd.read_csv(localPath+'data/sample_submission.csv')

#%%
aisles.head()
departments.head()
order_products__prior.head()
order_products__train.head()
orders.head()
products.head()
#%%basics
orders.info()
orders.isna().sum()
orders['order_id'].is_unique
#there are nan values in days_since_prior_order meaning it has a single order yet.
nanOrders = orders[orders['days_since_prior_order'].isna()]
orderSample = orders.head(100)
prioerSample = order_products__prior.head(100)
trainSample = order_products__train.head(100)

#%% questions
#1 - how many products in basket for each user?
userMaxOrder = orders.groupby("user_id")["order_number"].aggregate(np.max).reset_index()
maxOrderFreq = userMaxOrder['order_number'].value_counts().reset_index()
maxOrderFreq.columns = ['orderQnty','count']

#plot histogram
figDist = px.bar(maxOrderFreq,x='orderQnty',y='count',title='maxOrderHist')
plot(figDist,filename=localPath+'viz/maxOrderFreq.html')


#2 - which days are dominant
weekDays = orders['order_dow'].value_counts().reset_index().sort_values(by='index')
weekDays.columns = ['weekDay','count']
#plot
figDist = px.bar(weekDays,x='weekDay',y='count',title='weekDay vs orderedProductQty')
plot(figDist,filename=localPath+'viz/weekDaysVsOrderedProductsQty.html')

#3 - which hours are dominant
dayHours = orders['order_hour_of_day'].value_counts().reset_index().sort_values(by='index')
dayHours.columns = ['dayHours','count']
#plot
figDist = px.bar(dayHours,x='dayHours',y='count',title='dayHours vs orderedProductQty')
plot(figDist,filename=localPath+'viz/dayHoursVsOrderedProductsQty.html')

#4 - combine weekdays and hours
weekHours = orders.groupby(['order_dow','order_hour_of_day'])['order_id'].aggregate('count').reset_index()
#weekHours = weekHours.pivot('order_hour_of_day','order_dow', 'order_id')
weekHours = weekHours.pivot('order_dow','order_hour_of_day', 'order_id')
#plot
figHeat = px.imshow(weekHours,title='Days&Hours',color_continuous_scale='sunsetdark',text_auto=True,labels={'x':'Hours','y':'Days'})
plot(figHeat,filename=localPath+'viz/daysVsHours.html')
#it seems like saturday-sunday 10am to 15.pm are prime times

#5 - days between orders
orderInterval = orders['days_since_prior_order'].value_counts().reset_index().sort_values(by='index')
orderInterval.columns = ['intervalDays','count']
#plot
figDist = px.bar(orderInterval,x='intervalDays',y='count',title='Days between orders')
plot(figDist,filename=localPath+'/viz/daysBetweenOrders.html')

#6 - average days per user between orders
avgOrderInterval = orders.groupby('user_id')['days_since_prior_order'].aggregate(np.mean).reset_index()
avgOrderInterval = pd.cut(avgOrderInterval['days_since_prior_order'],bins=30,include_lowest=True,right=False)
avgOrderInterval = pd.DataFrame(avgOrderInterval.value_counts())
avgOrderInterval.reset_index(inplace=True)
avgOrderInterval = avgOrderInterval.sort_values(by='index')
avgOrderInterval['bins'] = avgOrderInterval['index'].astype(str)

figDist = px.bar(avgOrderInterval,x='bins', y='days_since_prior_order', title='Average Order Interval by User')
plot(figDist, filename=localPath+'viz/avgOrderIntervalByUser.html')

#%%prior df
#merge classes
prior = pd.merge(order_products__prior,products,left_on='product_id',right_on='product_id',how='left')
prior = pd.merge(prior,aisles,left_on='aisle_id',right_on='aisle_id',how='left')
prior = pd.merge(prior,departments, left_on = 'department_id', right_on='department_id', how='left')
#drop id columns
prior.drop(['aisle_id','department_id'],axis=1,inplace=True)
priorSample = prior.head(100)

#split df to ordered and non-ordered
ordered = prior[prior['reordered']==1]
nonOrdered = prior[prior['reordered']==0]
#top&bottom reordered products all and by category
orderedProducts = ordered['product_name'].value_counts().reset_index()
orderedProducts.columns=['product_name','count']
#select 20 of them
topOrderedProducts = orderedProducts.head(20)
bottomOrderedProducts = orderedProducts.tail(20)
#bottom has only count of 1 which makes sense but not give a much information. so let's visualize only top products
#plot
figBar = px.bar(topOrderedProducts,x='product_name',y='count')
plot(figBar, filename=localPath+'/viz/topReorderedProducts.html')

#func for visuals
def valCountVisual(df,colName,figName,rowLimit=100):
    tempValCount = df[colName].value_counts().reset_index()
    tempValCount.columns = [colName,'count']
    tempValCount = tempValCount.head(rowLimit)
    tempFig = px.bar(tempValCount,x=colName,y='count')
    plot(tempFig,filename=localPath+'/viz/{}.html'.format(figName))
    
    return print('Value counts for column : {} successfully done and visualized as {}'.format(colName,figName))

valCountVisual(ordered,'product_name','orderedTopProducts',rowLimit=25)
valCountVisual(ordered, 'aisle', 'topAisleProducts', rowLimit=25)
valCountVisual(ordered, 'department', 'topDepartmentProducts')

#for each department
depList = list(departments['department'])

topProductsByDep = pd.DataFrame()
for d in depList:
    tempDf = ordered[ordered['department']==d]
    tempDf = tempDf['product_name'].value_counts().reset_index()
    tempDf.columns = ['product_name','count']
    tempDf = tempDf.head(10)
    tempDf['department'] = d
    tempDf['percentage'] = tempDf['count'] / tempDf['count'].sum()
    topProductsByDep = topProductsByDep.append(tempDf)
    
#visualize
figBar = px.bar(topProductsByDep,x='product_name',y='percentage',color='department')
plot(figBar)


#products that have the lowest frequency of reordering all and by category




