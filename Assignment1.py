#-------------------------------------------------------
#	
#	COSC 757 -- DATA MINING
#	
#	Assignment 1
#	
#	
#	William Grant Hatcher
#	
#	Due: 9/24/2018
#	
#	Data Preprocessing and Exploratory Data Analysis (EDA)
#	
#-------------------------------------------------------

#python "W:\Documents\SCHOOL\Towson\2018-2022 -- DSc - Computer Security\6_Fall 2018\COSC 757 - Data Mining\Assignments\Classification Competition - 11-1\Assignment1.py"

file = input("Input File Name:")

### CH.2 LAB ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
import matplotlib
import seaborn as sns
matplotlib.style.use('ggplot')


#Read Auto-MPG dataset into a Pandas data frame
cover = pd.read_csv("W:\\Documents\\SCHOOL\\Towson\\2018-2022 -- DSc - Computer Security\\6_Fall 2018\\COSC 757 - Data Mining\\Assignments\\Classification Competition - 11-1\\%s" % file)

#Show the data frame
#print(police)

#Subset the data frame based on integer index (notice that the index starts at 0)
#police_tiny = police.iloc[0:5,[0,2,3,7]]

#Subset the data frame based on label-based index
#police_tiny = police.loc[0:5,['mpg','cylinders','displacement','horsepower']]

#Replace missing values with some constant
#police_tiny.iloc[1,1] = None
#police_tiny.iloc[3,3] = 'NA'
#police_tiny.iloc[1,1] = 0
#police_tiny.iloc[3,3] = 'Missing'

#Replace missing values with the field mean or mode
#cylinders_mean = police_tiny.mean().cylinders
#police_tiny.iloc[1,1] = cylinders_mean
#horsepower_mode = police_tiny.mode().horsepower
#police_tiny.iloc[3,3] = horsepower_mode[0]
#Notice in the above example that the mean method outputs a value and the mode 
#method outputs a table. This is the reason for the [0] at the end of brand_mode

#Replace missing values with a value generated at random from the observed distribution
#obs_horsepower = police_tiny.iloc[:,3].sample(n=1).index[0]
#police_tiny.iloc[3,3] = police_tiny.iloc[obs_horsepower,3]
#obs_cylinders = police_tiny.iloc[:,1].sample(n=1).index[0]
#police_tiny.iloc[1,1] = police_tiny.iloc[obs_cylinders,1]

#Five number summary with mean
#print(cover.manner_of_death_bin.describe())

'''
#Min-Max Normalization
mmnorm_manner_of_death_bin = (cover.manner_of_death_bin-cover.manner_of_death_bin.min())/(cover.manner_of_death_bin.max()-cover.manner_of_death_bin.min())

#Z-score Normalization
zscore_manner_of_death_bin = (cover.manner_of_death_bin-cover.manner_of_death_bin.mean())/cover.manner_of_death_bin.std()
'''

'''
cars2 = pd.read_csv("W:\\Documents\\SCHOOL\\Towson\\2018-2022 -- DSc - Computer Security\\6_Fall 2018\\COSC 757 - Data Mining\\Assignments\\Assignment 1 - 9-24\\%s" % file)
'''

#REGRESSION
'''
seaborn.lmplot(y='weight', x='mpg', data=police)  

from statsmodels.formula.api import ols
x = police.weight
y = police.mpg
mpg_weight_model = ols("mpg ~ weight", police).fit()

print(mpg_weight_model.summary())

offset, coef = mpg_weight_model._results.params

plt.plot(x, x*coef + offset)
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()

#Linear
seaborn.regplot(y='weight', x='mpg', data=police) 
plt.show() 
seaborn.regplot(y='displacement', x='mpg', data=police) 
plt.show()
seaborn.regplot(y='horsepower', x='mpg', data=police) 
plt.show()
seaborn.regplot(y='acceleration', x='mpg', data=police) 
plt.show()


seaborn.regplot(y='weight', x='displacement', data=police) 
plt.show() 
seaborn.regplot(y='weight', x='horsepower', data=police) 
plt.show()
seaborn.regplot(y='weight', x='acceleration', data=police) 
plt.show()

#multiple regression

from statsmodels.stats.anova import anova_lm
from mpl_toolkits.mplot3d import Axes3D
x = police.weight
y = police.horsepower
z = police.displacement

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm,
                       rstride=1, cstride=1)
ax.view_init(20, -120)
ax.set_xlabel('weight')
ax.set_ylabel('horsepower')
ax.set_zlabel('displacement')

whd_model = ols("displacement ~ weight + horsepower", police).fit()

# Print the summary
print(whd_model.summary())

print("\nRetrieving manually the parameter estimates:")
print(whd_model._results.params)
# should be array([-4.99754526,  3.00250049, -0.50514907])

# Peform analysis of variance on fitted linear model
anova_results = anova_lm(whd_model)

print('\nANOVA results')
print(anova_results)

plt.show()
'''


#Histograms (binning = range)

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Wilderness_Area_1'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Wilderness_Area_2'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Wilderness_Area_3'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Wilderness_Area_4'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

'''
plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Vertical_Distance_To_Hydrology'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Horizontal_Distance_To_Roadways'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Hillshade_9am'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Hillshade_noon'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Hillshade_3pm'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['Horizontal_Distance_To_Fire_Points'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()
'''

'''
#Massive Scatter Matrix!!
from pandas.plotting import scatter_matrix

scatter_matrix(cover[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Cover_Type"]],figsize = [16, 16],marker = ".", s = 0.2,diagonal="kde")
plt.show()
'''


'''
plt.figure(figsize=(8,6))
plt.hist(police.age, bins=30, color='xkcd:teal')
plt.xlabel("age",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()
'''

'''
plt.figure(figsize=(8,6))
ax = sns.countplot(cover['7745'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['7746'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['7755'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['7756'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['7757'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['7790'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['8703'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['8707'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['8708'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['8771'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['8772'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['8776'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['Cover_Type'], color='xkcd:teal')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
plt.show()
'''


#KDE Plots
'''
plt.figure(figsize=(8,6))
ax = police.manner_of_death_bin.plot.kde()
ax.set_xlabel("manner_of_death",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.armed_bin.plot.kde()
ax.set_xlabel("armed",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.armed_categories1.plot.kde()
ax.set_xlabel("armed_categories1",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.armed_categories2.plot.kde()
ax.set_xlabel("armed_categories2",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.age.plot.kde()
ax.set_xlabel("age",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.gender_bin.plot.kde()
ax.set_xlabel("gender",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police['race_bin'].plot.kde()
ax.set_xlabel("race",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.state_bin.plot.kde()
ax.set_xlabel("state",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.flee_bin.plot.kde()
ax.set_xlabel("flee",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

plt.figure(figsize=(8,6))
ax = police.threat_level_bin.plot.kde()
ax.set_xlabel("threat_level",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()
'''

#booleans --> Error

plt.figure(figsize=(8,6))
ax = police.body_camera.astype(int).plot.kde()
ax.set_xlabel("body_camera",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

#booleans --> Error

plt.figure(figsize=(8,6))
ax = police.signs_of_mental_illness.astype(int).plot.kde()
ax.set_xlabel("signs_of_mental_illness",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()



#quantiles
'''
print("QUANTILES:")
print("MPG:")
print(cars2.mpg.quantile([0.25,0.5,0.75]))
print("Cylinders:")
print(cars2.cylinders.quantile([0.25,0.5,0.75]))
print("Displacement:")
print(cars2.displacement.quantile([0.25,0.5,0.75]))
print("Horsepower:")
print(cars2.horsepower.quantile([0.25,0.5,0.75]))
print("Weight:")
print(cars2.weight.quantile([0.25,0.5,0.75]))
print("Acceleration:")
print(cars2.acceleration.quantile([0.25,0.5,0.75]))
print("Model Year:")
print(cars2['model year'].quantile([0.25,0.5,0.75]))

'''
#cars2.origin.quantile([0.25,0.5,0.75])


#Create a Scatterplot
'''
ax = cars2.plot.scatter('weight','mpg')
ax.set_xlabel("Weight",fontsize=20)
ax.set_ylabel("MPG",fontsize=20)
ax.set_title("Scatterplot of MPG by Weight",fontsize=20)
plt.show()

ax = cars2.plot.scatter('horsepower','mpg')
ax.set_xlabel("Horsepower",fontsize=20)
ax.set_ylabel("MPG",fontsize=20)
ax.set_title("Scatterplot of MPG by Horsepower",fontsize=20)
plt.show()

ax = cars2.plot.scatter('displacement','mpg')
ax.set_xlabel("Displacement",fontsize=20)
ax.set_ylabel("MPG",fontsize=20)
ax.set_title("Scatterplot of MPG by Displacement",fontsize=20)
plt.show()

ax = cars2.plot.scatter('acceleration','mpg')
ax.set_xlabel("Acceleration",fontsize=20)
ax.set_ylabel("MPG",fontsize=20)
ax.set_title("Scatterplot of MPG by Acceleration",fontsize=20)
plt.show()

ax = cars2.plot.scatter('cylinders','mpg')
ax.set_xlabel("Cylinders",fontsize=20)
ax.set_ylabel("MPG",fontsize=20)
ax.set_title("Scatterplot of MPG by Cylinders",fontsize=20)
plt.show()
'''

#Min-Max Normalization

mmnorm_mpg = (cars.mpg-cars.mpg.min())/(cars.mpg.max()-cars.mpg.min())

mmnorm_displacement = (cars.displacement-cars.displacement.min())/(cars.displacement.max()-cars.displacement.min())

mmnorm_acceleration = (cars.acceleration-cars.acceleration.min())/(cars.acceleration.max()-cars.acceleration.min())

mmnorm_horsepower = (cars.horsepower-cars.horsepower.min())/(cars.horsepower.max()-cars.horsepower.min())

mmnorm_weight = (cars.weight-cars.weight.min())/(cars.weight.max()-cars.weight.min())

'''
plt.hist(mmnorm_mpg, bins=int(mmnorm_mpg.max()-mmnorm_mpg.min()))
plt.xlabel("MPG",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(mmnorm_displacement)
plt.xlabel("Displacement",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(mmnorm_acceleration)
plt.xlabel("Acceleration",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(mmnorm_horsepower)
plt.xlabel("Horsepower",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(mmnorm_weight)
plt.xlabel("Weight",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()
'''

#Z-score Normalization

zscore_mpg = (cars.mpg-cars.mpg.mean())/cars.mpg.std()

zscore_displacement = (cars.displacement-cars.displacement.mean())/cars.displacement.std()

zscore_acceleration = (cars.acceleration-cars.acceleration.mean())/cars.acceleration.std()

zscore_horsepower = (cars.horsepower-cars.horsepower.mean())/cars.horsepower.std()

zscore_weight = (cars.weight-cars.weight.mean())/cars.weight.std()

'''
plt.hist(zscore_mpg, bins=int(zscore_mpg.max()-zscore_mpg.min()))
plt.xlabel("MPG",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(zscore_displacement)
plt.xlabel("Displacement",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(zscore_acceleration)
plt.xlabel("Acceleration",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(zscore_horsepower)
plt.xlabel("Horsepower",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(zscore_weight)
plt.xlabel("Weight",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()
'''

#Natural Log Transformation
natlog_mpg = np.log(cars.mpg)

natlog_displacement = np.log(cars.displacement)

natlog_acceleration = np.log(cars.acceleration)

natlog_horsepower = np.log(cars.horsepower)

natlog_weight = np.log(cars.weight)

'''
plt.hist(natlog_mpg, #bins=int(zscore_mpg.max()-zscore_mpg.min()))
)
plt.xlabel("MPG",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(natlog_displacement)
plt.xlabel("Displacement",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(natlog_acceleration)
plt.xlabel("Acceleration",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(natlog_horsepower)
plt.xlabel("Horsepower",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(natlog_weight)
plt.xlabel("Weight",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()
'''

#Inverse Square Root Transformation
invsqrt_mpg = 1/np.sqrt(cars.mpg)

invsqrt_displacement = 1/np.sqrt(cars.displacement)

invsqrt_acceleration = 1/np.sqrt(cars.acceleration)

invsqrt_horsepower = 1/np.sqrt(cars.horsepower)

invsqrt_weight = 1/np.sqrt(cars.weight)

'''
plt.hist(invsqrt_mpg, #bins=int(zscore_mpg.max()-zscore_mpg.min()))
)
plt.xlabel("MPG",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(invsqrt_displacement)
plt.xlabel("Displacement",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(invsqrt_acceleration)
plt.xlabel("Acceleration",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(invsqrt_horsepower)
plt.xlabel("Horsepower",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(invsqrt_weight)
plt.xlabel("Weight",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()
'''

#Square Root Transformation
sqrt_mpg = np.sqrt(cars.mpg)

sqrt_displacement = np.sqrt(cars.displacement)

sqrt_acceleration = np.sqrt(cars.acceleration)

sqrt_horsepower = np.sqrt(cars.horsepower)

sqrt_weight = np.sqrt(cars.weight)

'''
plt.hist(sqrt_mpg, #bins=int(zscore_mpg.max()-zscore_mpg.min()))
)
plt.xlabel("MPG",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(sqrt_displacement)
plt.xlabel("Displacement",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(sqrt_acceleration)
plt.xlabel("Acceleration",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(sqrt_horsepower)
plt.xlabel("Horsepower",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()

plt.hist(sqrt_weight)
plt.xlabel("Weight",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()
'''

#Calculate Skewness
ln_skew_weight =(3*(np.mean(natlog_weight)-np.median(natlog_weight)))/np.std(natlog_weight)
zscore_weight_skew = (3*(np.mean(zscore_weight)-np.median(zscore_weight)))/np.std(zscore_weight)
print("Swekness: ")
print(ln_skew_weight)

print("Skewness of Z-score: ")
print(zscore_weight_skew)


#Side-by-side Histograms of Weight and Z-Score of Weight
#fig, axarr = plt.subplots(1,2)
#cars.weight.hist(ax=axarr[0])
#zscore_weight.hist(ax=axarr[1])
#axarr[0].set_xlabel("Weight(lbs)")
#axarr[0].set_ylabel("Counts")
#axarr[1].set_xlabel("Z-score of Weight(lbs)")
#axarr[1].set_ylabel("Counts")
#plt.show()

#Normal Probability Plot 
'''
plt.ioff()
stats.probplot(cars.mpg, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(cars.displacement, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(cars.acceleration, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(cars.horsepower, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(cars.weight, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()
'''

#Normal Probability Plot MMNORM
'''
plt.ioff()
stats.probplot(mmnorm_mpg, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(mmnorm_displacement, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(mmnorm_acceleration, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(mmnorm_horsepower, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(mmnorm_weight, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()
'''

#Normal Probability Plot Zscore
'''
plt.ioff()
stats.probplot(zscore_mpg, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(zscore_displacement, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(zscore_acceleration, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(zscore_horsepower, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(zscore_weight, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()
'''

#Normal Probability Plot natlog
'''
plt.ioff()
stats.probplot(natlog_mpg, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(natlog_displacement, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(natlog_acceleration, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(natlog_horsepower, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(natlog_weight, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()
'''

#Normal Probability Plot invsqrt
'''
plt.ioff()
stats.probplot(invsqrt_mpg, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(invsqrt_displacement, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(invsqrt_acceleration, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(invsqrt_horsepower, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(invsqrt_weight, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()
'''

#Normal Probability Plot sqrt
'''
plt.ioff()
stats.probplot(sqrt_mpg, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(sqrt_displacement, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(sqrt_acceleration, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(sqrt_horsepower, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()

plt.ioff()
stats.probplot(sqrt_weight, dist="norm", plot=pylab)
plt.title("",fontsize=20)
plt.show()
'''


#Create Histogram with Normal Distribution
'''
invsqrt_weight_sorted = sorted(invsqrt_weight)
fit = stats.norm.pdf(invsqrt_weight_sorted, np.mean(invsqrt_weight_sorted), np.std(invsqrt_weight_sorted))
pylab.plot(invsqrt_weight_sorted, fit, '-')
pylab.hist(invsqrt_weight,normed=True)
plt.show()

#Create three flag variables for four categories
north_flag = np.zeros(10)
east_flag = np.zeros(10)
south_flag = np.zeros(10)
region = ["north","south","east","west","north","south","east","west","north","south"]
for i in range(0,len(region)):
    if region[i] == "north":
        north_flag[i] = 1
    if region[i] == "east":
        east_flag[i] = 1
    if region[i] == "south":
        south_flag[i] = 1
        
#Transforming the data
x = cars.weight[0]
#Transform x using y = 1/sqrt(x)
y = 1/np.sqrt(x)
#Detransform x using x = 1/(y)**2
detransformedx = 1/(y)**2

#Find duplicated records in a data frame
cars.duplicated()

#Concatenate 3 Series and Create an index field
x = pd.Series([1,1,3,2,1,1,2,3,4,3])
y = pd.Series([9,9,8,7,6,5,4,3,2,1])
z = pd.Series([2,1,2,3,4,5,6,7,8,9])
indexed_m = pd.DataFrame(dict(x=x,y=y,z=z)).reset_index()
'''


#Binning HORSEPOWER
#Enter the data and call it xdata
xdata = cars.horsepower
n = len(xdata)
nbins = 3
whichbin = np.zeros(n)


#Equal Frequency Binning
freq = n/nbins
xsorted = np.sort(xdata)
for i in range(1,nbins+1):
    for j in range(0,n):
        if (i-1)*freq < j+1 & j+1 <= i*freq:
            whichbin[j] = i
            
print("Whichbin:", whichbin)
print("Xsorted:", xsorted)

#plt.scatter(x=xsorted,y=whichbin)
plt.scatter(x=whichbin,y=xsorted)
plt.show()
			
			
#Binning with k-means
#from sklearn.cluster import KMeans
#model = KMeans(n_clusters=nbins)
#model.fit(xdata.reshape(-1,1))
#model.labels_


#Equal Width Binning
range_xdata = max(xdata)-min(xdata)+1
binwidth = range_xdata/nbins
for i in range(1,nbins+1):
    for j in range(0,n-1):
        if (i-1)*binwidth < xdata[j] & xdata[j] <= i*binwidth:
            whichbin[j] = i
    
	
print("Whichbin:", whichbin)
print("Xsorted:", xsorted)

#plt.scatter(x=xsorted,y=whichbin)
plt.scatter(x=whichbin,y=xsorted)
plt.show()
			

			
#Binning Displacement
#Enter the data and call it xdata
xdata = cars.displacement
n = len(xdata)
nbins = 3
whichbin = np.zeros(n)


#Equal Frequency Binning
freq = n/nbins
xsorted = np.sort(xdata)
for i in range(1,nbins+1):
    for j in range(0,n):
        if (i-1)*freq < j+1 & j+1 <= i*freq:
            whichbin[j] = i
            
print("Whichbin:", whichbin)
print("Xsorted:", xsorted)

#plt.scatter(x=xsorted,y=whichbin)
plt.scatter(x=whichbin,y=xsorted)
plt.show()
			
			
#Binning with k-means
#from sklearn.cluster import KMeans
#model = KMeans(n_clusters=nbins)
#model.fit(xdata.reshape(-1,1))
#model.labels_


#Equal Width Binning
range_xdata = max(xdata)-min(xdata)+1
binwidth = range_xdata/nbins
for i in range(1,nbins+1):
    for j in range(0,n-1):
        if (i-1)*binwidth < xdata[j] & xdata[j] <= i*binwidth:
            whichbin[j] = i
    
	
print("Whichbin:", whichbin)
print("Xsorted:", xsorted)

#plt.scatter(x=xsorted,y=whichbin)
plt.scatter(x=whichbin,y=xsorted)
plt.show()
	
### CH.3 LAB ###
	


cars = pd.read_csv("W:\\Documents\\SCHOOL\\Towson\\2018-2022 -- DSc - Computer Security\\6_Fall 2018\\COSC 757 - Data Mining\\Assignments\\Assignment 1 - 9-24\\%s" % file)

print(cars[0:10])

sum_origin = cars.origin.value_counts()
print("Sum of Origin:")
print(sum_origin)

prop_origin = float(sum_origin[1])/float(len(cars.origin))
print("Proportion of Origin:", prop_origin)

plt.figure()
ax = sum_origin.plot(kind='bar')
ax.set_title("Bar Graph of Origin")
ax.set_ylabel("Count")
plt.show()

from pandas.plotting import scatter_matrix
plt.figure()
scatter_matrix(cars[["weight","horsepower","displacement"]],diagonal="kde")
plt.show()

plt.figure()
scatter_matrix(cars[["mpg","weight","acceleration"]],diagonal="kde")
plt.show()

#Weight Displacement natlog
#Horsepower invsqrt
print(natlog_weight, natlog_displacement, invsqrt_horsepower)

wdi = pd.DataFrame(natlog_weight)
#wdi['weight'] = natlog_weight
wdi['displacement']= natlog_displacement
wdi['horsepower']= invsqrt_horsepower

print("WDI: ", wdi)

plt.figure()
scatter_matrix(wdi[['weight', 'horsepower', 'displacement']],diagonal="kde")
plt.show()

print("FREEZE!!!")

#----- Freezes
'''
weight_mpg = cars[["weight","mpg"]]
counts = pd.crosstab(cars["weight"],cars["mpg"])
print("Counts:", counts)

plt.figure()
counts.plot.bar(stacked=True)
plt.show()

counts_rows = pd.crosstab(cars["weight"],cars["mpg"], normalize="index")
print("Crosstab of Weight and MPG:", counts_rows)

plt.figure()
counts_rows.plot.bar(stacked=True)
plt.show()

counts_columns = pd.crosstab(cars["weight"],cars["mpg"], normalize="columns")
print("Crosstab of Weight and MPG - Columns:", counts_columns)

plt.figure()
counts_columns.transpose().plot.bar(stacked=True)
plt.show()

plt.figure()
ax = cars["Displacement"].hist()
ax.set_title("Histogram of Displacement")
ax.set_ylabel("Count")
ax.set_xlabel("Displacement")
plt.show()

plt.figure()
counts.plot.bar(stacked=False)
plt.show()

plt.figure()
counts.transpose().plot.bar(stacked=False)
plt.show()

plt.figure()
pd.crosstab(cars["displacement"],cars["mpg"]).transpose().plot.bar(stacked=True)
plt.show()

plt.figure()
pd.crosstab(cars["displacement"],cars["mpg"],normalize="columns").transpose().plot.bar(stacked=True)
plt.show()
'''

from scipy import stats

stats.ttest_ind(cars[churn["Churn"]==False]["Intl Calls"],cars[churn["Churn"]==True]["Intl Calls"],equal_var=False)

from pandas.plotting import scatter_matrix
plt.figure()
scatter_matrix(cars[["weight","horsepower","displacement"]],diagonal="kde")
plt.show()

from pandas.plotting import scatter_matrix
plt.figure()
scatter_matrix(cars[["mpg","weight","acceleration"]],diagonal="kde")
plt.show()

import scipy.stats as stats
slope, intercept, r_value, p_value, std_err = stats.linregress(cars["Day Charge"],cars["Day Mins"])
print ("r-squared:", r_value)
print ("p value:", p_value)


fig, ax = plt.subplots()
plt.figure()
df = pd.DataFrame(churn)
groups = df.groupby('Churn')

colors=['red', 'blue', 'yellow', 'purple']

for row, group in groups:
	group.plot(x="Day Mins",y="Eve Mins", ax=ax, kind='scatter', color=colors.pop(0), subplots=True)

#ax = churn.plot.scatter(x=["Day Mins"],y=["Eve Mins"],c="Churn")

ax.set_xlabel("Day Minutes")
ax.set_ylabel("Evening Minutes")
plt.show()


fig, ax = plt.subplots()
plt.figure()
df = pd.DataFrame(churn)
groups = df.groupby('Churn')


for row, group in groups:
	group.plot(x="Day Mins",y="CustServ Calls", ax=ax, kind='scatter', color=colors.pop(0), subplots=True)
#ax = churn.plot.scatter(x=["Day Mins"],y=["CustServ Calls"],c="Churn")
ax.set_xlabel("Day Minutes")
ax.set_ylabel("Customer Service Calls")
plt.show()

### ---- USE THIS !!!!! ---- ###
MinCallsTest = np.corrcoef(cars["weight"],cars["displacement"])
print("Correlation of Weight and Displacement: ", MinCallsTest)

MinChargeTest = np.corrcoef(cars["weight"],cars["horsepower"])
print("Correlation of Weight and Horsepower: ", MinChargeTest)

CallsChargeTest = np.corrcoef(cars["displacement"],cars["horsepower"])
print("Correlation of Displacement and Horsepower: ", CallsChargeTest)

MinCallsTest = np.corrcoef(cars["mpg"],cars["displacement"])
print("Correlation of MPG and Displacement: ", MinCallsTest)

MinChargeTest = np.corrcoef(cars["mpg"],cars["horsepower"])
print("Correlation of MPG and Horsepower: ", MinChargeTest)

CallsChargeTest = np.corrcoef(cars["mpg"],cars["weight"])
print("Correlation of MPG and Weight: ", CallsChargeTest)

CallsChargeTest = np.corrcoef(cars["mpg"],cars["acceleration"])
print("Correlation of Displacement and Acceleration: ", CallsChargeTest)