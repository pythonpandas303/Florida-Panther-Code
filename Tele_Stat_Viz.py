### MSDS 692 Project
### Tom Teasdale
### Christy Pearson (Instructor)
### Spatial Statistics
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.pyplot import figure
from pointpats import (
    centrography,
    distance_statistics,
    QStatistic,
    random,
    PointPattern,
)


# Changing Directory and reading data into environment

os.chdir("C:/Users/megal/Desktop/MSDS692Project")
tele=pd.read_csv("Tele_Clean.csv")

# Grouping telemetry data by decade

eightys = tele.loc[tele['Year'].isin([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])]
ninetys = tele.loc[tele['Year'].isin([1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000])]
twothous = tele.loc[tele['Year'].isin([2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010])]
tens = tele.loc[tele['Year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])]

# Calculating mean and median centroids

mean_center80 = centrography.mean_center(eightys[["Longitude", "Latitude"]])
med_center80 = centrography.euclidean_median(eightys[["Longitude", "Latitude"]])

mean_center90 = centrography.mean_center(ninetys[["Longitude", "Latitude"]])
med_center90 = centrography.euclidean_median(ninetys[["Longitude", "Latitude"]])

mean_center00 = centrography.mean_center(twothous[["Longitude", "Latitude"]])
med_center00 = centrography.euclidean_median(twothous[["Longitude", "Latitude"]])

mean_center10 = centrography.mean_center(tens[["Longitude", "Latitude"]])
med_center10 = centrography.euclidean_median(tens[["Longitude", "Latitude"]])


# Plotting

# Generate scatter plot
joint_axes = sns.jointplot(
    x="Longitude", y="Latitude", data=tens, s=0.75, height=9
)
# Add mean point and marginal lines
joint_axes.ax_joint.scatter(
    *mean_center10, color="red", marker="x", s=50, label="Mean Center"
)
joint_axes.ax_marg_x.axvline(mean_center10[0], color="red")
joint_axes.ax_marg_y.axhline(mean_center10[1], color="red")

# Add median point and marginal lines
joint_axes.ax_joint.scatter(
    *med_center10,
    color="limegreen",
    marker="o",
    s=50,
    label="Median Center"
)
joint_axes.ax_marg_x.axvline(med_center10[0], color="limegreen")
joint_axes.ax_marg_y.axhline(med_center10[1], color="limegreen")
# Legend
joint_axes.ax_joint.legend(title="Mean Center = -81.31839245, 26.20070932")

# Clean axes
joint_axes.ax_joint.set_axis_off()
plt.suptitle("Florida Panther Central Tendency 2011-2020", y=1.04, size=20, weight="bold")
plt.figure(figsize=(9, 9), dpi=300)

# Display
plt.show()

# Code for grouping images

os.chdir("C:/Users/megal/Desktop/MSDS692Project/Viz")
# get images    
img1 = Image.open('cent1980.png')
img2 = Image.open('cent1990.png')
img3 = Image.open('cent2000.png')
img4 = Image.open('cent2010.png')

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
new_image.save('combined_cent.png')


### Dispersion calculations ###

coordinates = eightys[["Longitude", "Latitude"]].values

qstat = QStatistic(coordinates)
qstat.plot(title="Florida Panther Telemetry Quadrat Statistics, 1981-1990")
qstat.chi2_pvalue

coordinates1 = ninetys[["Longitude", "Latitude"]].values
qstat1 = QStatistic(coordinates1)
qstat1.plot(title="Florida Panther Telemetry Quadrat Statistics, 1991-2000")
qstat1.chi2_pvalue

coordinates2 = twothous[["Longitude", "Latitude"]].values
qstat1 = QStatistic(coordinates2)
qstat1.plot(title="Florida Panther Telemetry Quadrat Statistics, 2001-2010")
qstat1.chi2_pvalue

coordinates3 = tens[["Longitude", "Latitude"]].values
qstat1 = QStatistic(coordinates3)
qstat1.plot(title="Florida Panther Telemetry Quadrat Statistics, 2011-2020")
qstat1.chi2_pvalue

# Code for combining images
os.chdir("C:/Users/megal/Desktop/MSDS692Project/Viz")
# get images    
img1 = Image.open('1980quad.png')
img2 = Image.open('1990quad.png')
img3 = Image.open('2000quad.png')
img4 = Image.open('2010quad.png')

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
new_image.save('combined_quad.png')





