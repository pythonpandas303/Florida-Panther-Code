### MSDS 692 Project
### Tom Teasdale
### Christy Pearson (Instructor)
### Regression Analysis and Predictions



import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from PIL import Image


os.chdir("C:/Users/megal/Desktop/MSDS692Project")

df = pd.read_csv("Florida_Panther_Telemetry.csv")
df = df.drop(['OBJECTID', 'AGENCY', 'TIME', 'last_edited_date'], axis=1)



df.rename(columns={'FLGTDATE':'Flight_Date'}, inplace=True)
df = df.dropna()
df = df.reset_index(drop=True)
df['Year'] = pd.DatetimeIndex(df['Flight_Date']).year
Q1 = np.percentile(df['UTM83NORTH'], 25,
                   interpolation = 'midpoint')
 
Q3 = np.percentile(df['UTM83NORTH'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1

print("Old Shape: ", df.shape)
 
# Upper bound
upper = np.where(df['UTM83NORTH'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df['UTM83NORTH'] <= (Q1-1.5*IQR))
 
''' Removing the Outliers '''
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
 
print("New Shape: ", df.shape)
# Minimum Convex Hull to show denning range by year
df2 = df[['UTM83EAST', 'UTM83NORTH']].copy()
df2 = df2.replace(0, np.nan).dropna(axis=0, how='any').fillna(0).astype(int)




points = df2.values   

hull = ConvexHull(points)
print(hull.area)

plt.rcParams["figure.figsize"] = (10,10)

plt.plot(points[:,0], points[:,1], 'o')

for simplex in hull.simplices:

    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.suptitle("Florida Panther Historical Range 1981-2020", size='xx-large')
plt.title('Area = 438507 HA, ~1693 sq mi')
plt.xlabel('Longitude (UTM)')
plt.ylabel('Latitude (UTM)')



plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')

plt.show()

hist = hull.area






y5 = df.loc[df['Year'].isin([1985])]
y5 = y5[['UTM83EAST', 'UTM83NORTH']].copy()
y6 = df.loc[df['Year'].isin([1986])]
y6 = y6[['UTM83EAST', 'UTM83NORTH']].copy()
y7 = df.loc[df['Year'].isin([1987])]
y7 = y7[['UTM83EAST', 'UTM83NORTH']].copy()
y8 = df.loc[df['Year'].isin([1988])]
y8 = y8[['UTM83EAST', 'UTM83NORTH']].copy()
y9 = df.loc[df['Year'].isin([1989])]
y9 = y9[['UTM83EAST', 'UTM83NORTH']].copy()
y10 = df.loc[df['Year'].isin([1990])]
y10 = y10[['UTM83EAST', 'UTM83NORTH']].copy()
y11 = df.loc[df['Year'].isin([1991])]
y11 = y11[['UTM83EAST', 'UTM83NORTH']].copy()
y12 = df.loc[df['Year'].isin([1992])]
y12 = y12[['UTM83EAST', 'UTM83NORTH']].copy()
y13 = df.loc[df['Year'].isin([1993])]
y13 = y13[['UTM83EAST', 'UTM83NORTH']].copy()
y14 = df.loc[df['Year'].isin([1994])]
y14 = y14[['UTM83EAST', 'UTM83NORTH']].copy()
y15 = df.loc[df['Year'].isin([1995])]
y15 = y15[['UTM83EAST', 'UTM83NORTH']].copy()
y16 = df.loc[df['Year'].isin([1996])]
y16 = y16[['UTM83EAST', 'UTM83NORTH']].copy()
y17 = df.loc[df['Year'].isin([1997])]
y17 = y17[['UTM83EAST', 'UTM83NORTH']].copy()
y18 = df.loc[df['Year'].isin([1998])]
y18 = y18[['UTM83EAST', 'UTM83NORTH']].copy()
y19 = df.loc[df['Year'].isin([1999])]
y19 = y19[['UTM83EAST', 'UTM83NORTH']].copy()
y20 = df.loc[df['Year'].isin([2000])]
y20 = y20[['UTM83EAST', 'UTM83NORTH']].copy()
y21 = df.loc[df['Year'].isin([2001])]
y21 = y21[['UTM83EAST', 'UTM83NORTH']].copy()
y22 = df.loc[df['Year'].isin([2002])]
y22 = y22[['UTM83EAST', 'UTM83NORTH']].copy()
y23 = df.loc[df['Year'].isin([2003])]
y23 = y23[['UTM83EAST', 'UTM83NORTH']].copy()
y24 = df.loc[df['Year'].isin([2004])]
y24 = y24[['UTM83EAST', 'UTM83NORTH']].copy()
y25 = df.loc[df['Year'].isin([2005])]
y25 = y25[['UTM83EAST', 'UTM83NORTH']].copy()
y26 = df.loc[df['Year'].isin([2006])]
y26 = y26[['UTM83EAST', 'UTM83NORTH']].copy()
y27 = df.loc[df['Year'].isin([2007])]
y27 = y27[['UTM83EAST', 'UTM83NORTH']].copy()
y28 = df.loc[df['Year'].isin([2008])]
y28 = y28[['UTM83EAST', 'UTM83NORTH']].copy()
y29 = df.loc[df['Year'].isin([2009])]
y29 = y29[['UTM83EAST', 'UTM83NORTH']].copy()
y30 = df.loc[df['Year'].isin([2010])]
y30 = y30[['UTM83EAST', 'UTM83NORTH']].copy()
y31 = df.loc[df['Year'].isin([2011])]
y31 = y31[['UTM83EAST', 'UTM83NORTH']].copy()
y32 = df.loc[df['Year'].isin([2012])]
y32 = y32[['UTM83EAST', 'UTM83NORTH']].copy()
y33 = df.loc[df['Year'].isin([2013])]
y33 = y33[['UTM83EAST', 'UTM83NORTH']].copy()
y34 = df.loc[df['Year'].isin([2014])]
y34 = y34[['UTM83EAST', 'UTM83NORTH']].copy()
y35 = df.loc[df['Year'].isin([2015])]
y35 = y35[['UTM83EAST', 'UTM83NORTH']].copy()
y36 = df.loc[df['Year'].isin([2016])]
y36 = y36[['UTM83EAST', 'UTM83NORTH']].copy()
y37 = df.loc[df['Year'].isin([2017])]
y37 = y37[['UTM83EAST', 'UTM83NORTH']].copy()
y38 = df.loc[df['Year'].isin([2018])]
y38 = y38[['UTM83EAST', 'UTM83NORTH']].copy()
y39 = df.loc[df['Year'].isin([2019])]
y39 = y39[['UTM83EAST', 'UTM83NORTH']].copy()




points5 = y5.values   
hull5 = ConvexHull(points5)
eighty5 = hull5.area

points6 = y6.values   
hull6 = ConvexHull(points6)
eighty6 = hull6.area

points7 = y7.values   
hull7 = ConvexHull(points7)
eighty7 = hull7.area

points8 = y8.values   
hull8 = ConvexHull(points8)
eighty8 = hull8.area

points9 = y9.values   
hull9 = ConvexHull(points9)
eighty9 = hull9.area

points10 = y10.values   
hull10 = ConvexHull(points10)
ninety = hull10.area

points11 = y11.values   
hull11 = ConvexHull(points11)
ninety1 = hull11.area

points12 = y12.values   
hull12 = ConvexHull(points12)
ninety2 = hull12.area

points13 = y13.values   
hull13 = ConvexHull(points13)
ninety3 = hull13.area

points14 = y14.values   
hull14 = ConvexHull(points14)
ninety4 = hull14.area

points15 = y15.values   
hull15 = ConvexHull(points15)
ninety5 = hull15.area

points16 = y16.values   
hull16 = ConvexHull(points16)
ninety6 = hull16.area

points17 = y17.values   
hull17 = ConvexHull(points17)
ninety7 = hull17.area

points18 = y18.values   
hull18 = ConvexHull(points18)
ninety8 = hull18.area

points19 = y19.values   
hull19 = ConvexHull(points19)
ninety9 = hull19.area

points20 = y20.values   
hull20 = ConvexHull(points20)
twothous = hull20.area

points21 = y21.values   
hull21 = ConvexHull(points21)
twothous1 = hull21.area

points22 = y22.values   
hull22 = ConvexHull(points22)
twothous2 = hull22.area

points23 = y23.values   
hull23 = ConvexHull(points23)
twothous3 = hull23.area

points24 = y24.values   
hull24 = ConvexHull(points24)
twothous4 = hull24.area

points25 = y25.values   
hull25 = ConvexHull(points25)
twothous5 = hull25.area

points26 = y26.values   
hull26 = ConvexHull(points26)
twothous6 = hull26.area

points27 = y27.values   
hull27 = ConvexHull(points27)
twothous7 = hull27.area

points28 = y28.values   
hull28 = ConvexHull(points28)
twothous8 = hull28.area

points29 = y29.values   
hull29 = ConvexHull(points29)
twothous9 = hull29.area

points30 = y30.values   
hull30 = ConvexHull(points30)
twothous10 = hull30.area

points31 = y31.values   
hull31 = ConvexHull(points31)
twothous11 = hull31.area

points32 = y32.values   
hull32 = ConvexHull(points32)
twothous12 = hull32.area

points33 = y33.values   
hull33 = ConvexHull(points33)
twothous13 = hull33.area

points34 = y34.values   
hull34 = ConvexHull(points34)
twothous14 = hull34.area

points35 = y35.values   
hull35 = ConvexHull(points35)
twothous15 = hull35.area

points36 = y36.values   
hull36 = ConvexHull(points36)
twothous16 = hull36.area

points37 = y37.values   
hull37 = ConvexHull(points37)
twothous17 = hull37.area

points38 = y38.values   
hull38 = ConvexHull(points38)
twothous18 = hull38.area

points39 = y39.values   
hull39 = ConvexHull(points39)
twothous19 = hull39.area

data = {'Year': ['1987', '1988', '1989', '1990',
                 '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000',
                 '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
                 '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'],
        'Area': [eighty7, eighty8, eighty9,
                 ninety, ninety1, ninety2, ninety3, ninety4, ninety5, ninety6, ninety7, ninety8,
                 ninety9, twothous, twothous1, twothous2, twothous3, twothous4, twothous5, twothous6,
                 twothous7, twothous8, twothous9, twothous10, twothous11, twothous12, twothous13,
                 twothous14, twothous15, twothous16, twothous17, twothous18, twothous19]}

df3 = pd.DataFrame(data)
df3['Year']= pd.to_datetime(df3['Year'])

df3['Time'] = np.arange(len(df3.index))

df3.set_index('Year')

plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)


fig, ax = plt.subplots()
ax.plot('Time', 'Area', data=df3, color='0.75')
ax = sns.regplot(x='Time', y='Area', data=df3, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Florida Panther Range');



# Training data
X = df3.loc[:, ['Time']] 
y = df3.loc[:, 'Area']  

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)


ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Florida Panther Range');


train = df3[df3.index < 26]
test = df3[df3.index >= 26]



y = train['Area']
ARMAmodel = SARIMAX(y, order = (5, 2, 0))

ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('Florida Panther Range in HA * 100')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title("Train/Test split for Data")
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.legend()
plt.show()



arma_rmse = np.sqrt(mean_squared_error(test["Area"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

zero = ARMAmodel.predict(33)
one = ARMAmodel.predict(34)
two = ARMAmodel.predict(35)
three = ARMAmodel.predict(36)
four = ARMAmodel.predict(37)
five = ARMAmodel.predict(38)
six = ARMAmodel.predict(39)

future = {'Year' : ['2020', '2021', '2022', '2023', '2024', '2025', '2026'], 
          'Area' : [236122.43, 229718.01, 223555.71, 216315.94, 210350.89, 204574.55, 198671.35]}

pred = pd.DataFrame(future)


fig = go.Figure(data=[go.Table(
    header=dict(values=list(pred.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[pred.Year, pred.Area],
               fill_color='lavender',
               align='left'))
])

fig.write_image('table.png')

fig = go.Figure(go.Indicator(
    mode = "number+gauge+delta",
    gauge = {'shape': "bullet"},
    delta = {'reference': 204574.55},
    value = 198671.35,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "Area, 2025-2026"}))

fig.update_layout(
    width=1600
)

fig.write_image('card7.png')
