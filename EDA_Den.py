### MSDS 692 Project
### Tom Teasdale
### Christy Pearson (Instructor)

### EDA Phase (multiple plots, individual plots notated by '###')

# Importing required Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

font = {'family': 'serif',
        'color':  'Black',
        'weight': 'bold',
        'size': 12,
        }

fonttitle = {'family': 'serif',
        'color':  'Black',
        'weight': 'bold',
        'size': 16,
        }


## Change working directory as necessary
# os.chdir(working_directory)

# reading data into environment
df = pd.read_csv('Den_Clean.csv')


### Habitat Type data set and plotting ###

# Obtaining value counts of habitat types, normalizing to float
hab = df['Habitat'].value_counts(normalize=True)[:10]

# Converting Series to DataFrame
habdf = pd.DataFrame(hab)
# Restting Index
habdf = habdf.reset_index()
# Naming columns of new data frame
habdf.columns = ['Habitat Type', 'Percentage']
# Converting float to percentage
habdf['Percentage']=habdf['Percentage']*100

# Setting variable names for plotting
habitat = habdf["Habitat Type"]
perc = habdf["Percentage"]

# Figure Size
fig, ax = plt.subplots(figsize =(16, 9))
 
# Horizontal Bar Plot
ax.barh(habitat, perc)
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2))+('%'),
             fontsize = 10, fontweight ='bold',
             color ='grey')
 
# Add Plot Title
ax.set_title('Florida Panther Habitat Selection in Denning Sites (%)',
             loc ='left', )
  
# Show Plot
plt.show()

plt.savefig('Habitat_Type.png')

### Location data set and plotting ###

# Obtaining value counts of locations

loc = df['Location'].value_counts()[:10]

# Converting Series to DataFrame
locdf = pd.DataFrame(loc)
# Restting Index
locdf = locdf.reset_index()
# Naming columns of new data frame
locdf.columns = ['Location', '# of Events']

# Setting variable names for plotting
loc = locdf["Location"]
dens = locdf["# of Events"]

# Figure Size
fig, ax = plt.subplots(figsize =(16, 9))
 
# Horizontal Bar Plot
ax.barh(loc, dens)
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')
 
# Add Plot Title
ax.set_title('Observed Florida Panther Dens by Location',
             loc ='left')
 
# Show Plot
plt.show()
plt.savefig('Location.png')



habkit = df[["Habitat", "# of Kittens"]].copy()
lockit = df[["Location", "# of Kittens"]].copy()

threshold = 4
value_counts1 = habkit["Habitat"].value_counts() 
to_remove1 = value_counts1[value_counts1 <= threshold].index
habkit["Habitat"].replace(to_remove1, np.nan, inplace=True)

habkitmean = habkit.groupby(["Habitat"], as_index=False).mean().round(2)



threshold = 4
value_counts2 = lockit["Location"].value_counts() 
to_remove2 = value_counts2[value_counts2 <= threshold].index
lockit["Location"].replace(to_remove2, np.nan, inplace=True)


lockitmean = lockit.groupby(["Location"], as_index=False).mean().round(2)


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
  
if __name__ == '__main__':
    

    x = habkitmean["Habitat"]
    y = habkitmean["# of Kittens"]
      

    plt.figure(figsize = (10, 5))
    plt.bar(x, y)
    addlabels(x, y)
    plt.title("Mean Number of Kittens by Habitat", fontdict=fonttitle)
    plt.xlabel("Habitat Type", fontdict=font)
    plt.ylabel("Number of Kittens", fontdict=font)
    plt.show()

if __name__ == '__main__':
    

    x = lockitmean["Location"]
    y = lockitmean["# of Kittens"]
      

    plt.figure(figsize = (10, 5))
    plt.bar(x, y)
    addlabels(x, y)
    plt.title("Mean Number of Kittens by Location", fontdict=fonttitle)
    plt.xlabel("Location", fontdict=font)
    plt.ylabel("Number of Kittens", fontdict=font)
    plt.show()














