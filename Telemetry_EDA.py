### MSDS 692 Project
### Tom Teasdale
### Christy Pearson (Instructor)
### EDA (Geographic Data)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image



os.chdir("C:/Users/megal/Desktop/MSDS692Project")
tele=pd.read_csv("Tele_Clean.csv")

eightys = tele.loc[tele['Year'].isin([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])]
ninetys = tele.loc[tele['Year'].isin([1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000])]
twothous = tele.loc[tele['Year'].isin([2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010])]
tens = tele.loc[tele['Year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])]

# KDE plot

f, ax = plt.subplots(1, figsize=(9, 9))
f = sns.jointplot(x="Longitude", y="Latitude", data=tens, height=10)
f = f.plot_joint(sns.kdeplot, cmap="viridis_r", fill=True, alpha=0.7, levels=5)
plt.suptitle("Florida Panther Telemetry 2011-2020", y=1.02, size=28, weight="bold")


# Change directory to image folder
os.chdir("C:/Users/megal/Desktop/MSDS692Project/Viz")



### Code for combining plots
# get images    
img1 = Image.open('1980s.png')
img2 = Image.open('1990s.png')
img3 = Image.open('2000s.png')
img4 = Image.open('2010s.png')

# get width and height
w1, h1 = img1.size
w2, h2 = img2.size
w3, h3 = img3.size
w4, h4 = img4.size

# to calculate size of new image 
w = max(w1, w2, w3, w4)
h = max(h1, h2, h3, h4)

# create big empty image with place for images
new_image = Image.new('RGB', (w*2, h*2))

# put images on new_image
new_image.paste(img1, (0, 0))
new_image.paste(img2, (w, 0))
new_image.paste(img3, (0, h))
new_image.paste(img4, (w, h))

# save it
new_image.save('combined.png')














