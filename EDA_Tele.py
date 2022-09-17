### MSDS 692 Project
### Tom Teasdale
### Christy Pearson (Instructor)
### EDA (Geographic Data)

import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import os
import plotly.io as pio

os.chdir("C:/Users/megal/Desktop/MSDS692Project")
tele=pd.read_csv("Tele_Clean.csv")

### Geographic distribution plot

# g= sns.JointGrid(x="Longitude", 
#                  y="Latitude",
#                  hue="Year",
#                  data=tele, 
#                  height=7)
# g.fig.suptitle("Geographic Distribution of Florida Panther Telemetry (1981-2020)", fontsize=12, fontweight="bold")
# g.fig.tight_layout()
# g.fig.subplots_adjust(top=0.95)
# g= g.plot_joint(sns.kdeplot)
# g= g.plot(sns.scatterplot, sns.histplot)

### Path Sampling ### Later Years Selected

#Subsetting data frame for 2018 and later
tele2018 = tele[tele["Year"] >= 2018]

#Selecting a random sample of individuals for plotting
samplecats = tele2018["Cat_Num"].tolist()
sampledcats = set(random.choices(samplecats, k=19))

# sampledcats
# Out[42]: 
# {'FP193',
#  'FP220',
#  'FP224',
#  'FP245',
#  'FP246',
#  'FP247',
#  'FP250',
#  'FP253',
#  'FP256',
#  'FP257'}


sample_cats = tele2018.loc[tele2018['Cat_Num'].isin(['FP193','FP220','FP224','FP245','FP246','FP247',
                                                     'FP250','FP253','FP256','FP257'])]


fig = px.scatter_mapbox(sample_cats, lat="Latitude", lon="Longitude", hover_name="Cat_Num", hover_data=["Cat_Num", "Flight_Date"],
                        color_discrete_sequence=["black"], zoom=9, height=1000)
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

pio.write_html(fig, file='map.html', auto_open=True)











