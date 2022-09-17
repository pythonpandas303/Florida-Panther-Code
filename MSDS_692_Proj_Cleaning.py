### MSDS 692 Project
### Tom Teasdale
### Christy Pearson (Instructor)

### Data Cleaning

import os
import pandas as pd

## Change working directory as necessary
# os.chdir(working_directory)

# reading data into environment
den = pd.read_csv("Den_DB.csv")

# Dropping unneccesary columns
den = den.drop(['OrigX', 'OrigY', 'Origdatum', 'Kittens handled', 'Purported field Sire', 'Litter Staus', 'Comments', 'Den Interval'], axis=1)
den["# of Kittens"] = den["Minimum number of kittens at den"]
den = den.drop(["Minimum number of kittens at den"], axis=1)

# reading data into environment
tele = pd.read_csv("Florida_Panther_Telemetry.csv", parse_dates=["FLGTDATE"])


# Dropping unneccesary columns
tele = tele.drop(['OBJECTID', 'AGENCY', 'TIME', 'UTM83EAST', 'UTM83NORTH', 'last_edited_date'], axis=1)

# Dropping NA's
den = den.dropna()
den = den.reset_index(drop=True)

tele = tele.dropna()
tele = tele.reset_index(drop=True)

# Renaming Columns in Telemetry Dataset

tele.rename(columns={'X':'Longitude', 'Y':'Latitude', 'CATNUMBER':'Cat_Num', 'FLGTDATE':'Flight_Date'}, inplace=True)

# Adjusting Date Column to parse out year
tele['Year'] = pd.DatetimeIndex(tele['Flight_Date']).year

#combining some location values

den['Location'] = den['Location'].replace({'SBCNP- Turner River': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'SBCNP-Turner River': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'SBCNP-Turner River ': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'SBCNP - Turner River': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'BCNP-Turner River ': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'BCNP - Turner River Unit': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'BCNP- Turner River Unit': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'SBCNP - Turner River Unit': 'BCNP-Turner River'})
den['Location'] = den['Location'].replace({'SBCNP -Deep Lake': 'SBCNP-Deep Lake'})
den['Location'] = den['Location'].replace({'SBCNP - Deep Lake': 'SBCNP-Deep Lake'})
den['Location'] = den['Location'].replace({'BCNP-Deep Lake Unit': 'SBCNP-Deep Lake'})
den['Location'] = den['Location'].replace({'Immokalee Ranch- E of Bishop Pens': 'Immokalee Ranch'})
den['Location'] = den['Location'].replace({'Immokalee Ranch-Sof the NE entrance': 'Immokalee Ranch'})
den['Location'] = den['Location'].replace({'Immokalee Ranch - Bishop Pen Gate': 'Immokalee Ranch'})
den['Location'] = den['Location'].replace({'Immokalee Ranch-SW corner': 'Immokalee Ranch'})


#combining some Habitat values

den['Habitat'] = den['Habitat'].replace({'Pine Palmetto': 'Pine/Palmetto'})
den['Habitat'] = den['Habitat'].replace({'Pine/palmetto': 'Pine/Palmetto'})
den['Habitat'] = den['Habitat'].replace({'Pine/Palmetto/fern': 'Pine/Palmetto'})
den['Habitat'] = den['Habitat'].replace({'Saw Palmetto': 'Palmetto'})
den['Habitat'] = den['Habitat'].replace({'Hardwood hammock/Fern': 'Hardwood hammock'})
den['Habitat'] = den['Habitat'].replace({'HH': 'Hardwood hammock'})
den['Habitat'] = den['Habitat'].replace({'Mied swamp': 'Mixed Swamp'})
den['Habitat'] = den['Habitat'].replace({'MS': 'Mixed Swamp'})


#Exporting clean datasets to csv

den.to_csv('Den_Clean.csv')
tele.to_csv('Tele_Clean.csv')



