# import libraries
import codecademylib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# load data
heart = pd.read_csv('heart_disease.csv')
print(heart.head())

# looking at the association between thalach and heart disease.
sns.boxplot(data=heart, x='heart_disease', y='thalach')
plt.show()
'''from the boxplot diagram, we can see that people who were diagnosed with heart disease generally have a lower maximum heart rate achieved in exercise test. The median of people with heart disease is much lower than people without one. And there is little overlap between these two boxes. This indicates that there is a relationshop between these variables.'''

thalach_hd = heart.thalach[heart.heart_disease == 'presence']
thalach_no_hd = heart[heart.heart_disease == 'absence'].thalach
thalach_hd_mean = np.mean(thalach_hd)
thalach_hd_median = np.median(thalach_hd)
thalach_no_hd_mean = np.mean(thalach_no_hd)
thalach_no_hd_median = np.median(thalach_no_hd)
print(thalach_hd_mean-thalach_no_hd_mean)
print(thalach_hd_median-thalach_no_hd_median)

# Investigating that whether the average thalach of a heart disease patient is significantly different from the average thalach for a person without heart disease.

from scipy.stats import ttest_ind
tstat, pval = ttest_ind(thalach_hd, thalach_no_hd)
print(pval)

'''The p-value is 3.5e-14 which is way less than 0.05. Therefore, there is a significant difference in average thalach for people with heart disease compared to people with no heart disease.'''

# looking at the association between age and heart disease.
plt.clf()
sns.boxplot(data=heart, x='heart_disease', y='age')
plt.show()

'''The boxplot shows that people with heart disease are generally older than people with no heart disease. The median is much higher. There is some overlap between these two boxes, but not too much.'''

age_hd = heart.age[heart.heart_disease == 'presence']
age_no_hd = heart.age[heart.heart_disease == 'absence']
mean_diff = np.mean(age_hd) - np.mean(age_no_hd)
median_diff = np.median(age_hd) - np.median(age_no_hd)
print(mean_diff, median_diff)
tstat, pval_age = ttest_ind(age_hd, age_no_hd)
print(pval_age)

'''The p-value is 0.00009 which is less than 0.05, therefore, there is a significant difference in average age for people with heart disease compared to people with no heart disease.'''

# Create a function that can indicate whether a quantitative variable (like age, trestbps, chol, etc) have a significant association with the binary categorical variable heart_disease.

def is_associated(column):
  #create a boxplot
  plt.clf()
  sns.boxplot(data=heart, x='heart_disease', y=column)
  plt.show()
  #create seperate array for both samples
  hd = heart[heart.heart_disease == 'presence'][column]
  no_hd = heart[heart.heart_disease == 'absence'][column]
  #calculate the mean_diff and median_diff
  mean_diff = np.mean(hd) -np.mean(no_hd)
  median_diff = np.median(hd) - np.median(no_hd)
  print('The mean difference is ' + str(mean_diff))
  print('The median difference is ' + str(median_diff))
  #investigate the p-value
  tstat, pval = ttest_ind(hd, no_hd)
  print(pval)
  if pval < 0.05:
    print('The p-value is significant. Reject the null hypothesis.')
  else:
    print('The p-value is not significant. Cannot reject the null hyphthesis.')

# looking at the association between trestbps and heart disease.
is_associated('trestbps')
# looking at the association between chol and heart disease.
is_associated('chol')

# Investigating the relationship between thalach(quantitative) and cp (non-binary categorical).
plt.clf()
sns.boxplot(data=heart, x='cp', y='thalach')
plt.show()
'''from the boxplot, it is difficult to decide which average thalach is significantly higher than the others, but chest pain type of asymptomatic seems to have a much lower average thalach than other chest pain types.'''
thalach_typical = heart[heart.cp == 'typical angina'].thalach
thalach_asymptom = heart[heart.cp == 'asymptomatic'].thalach
thalach_nonangin = heart[heart.cp == 'non-anginal pain'].thalach
thalach_atypical = heart[heart.cp == 'atypical angina'].thalach

#using ANOVA and Tukey's Range Test
from scipy.stats import f_oneway
fstat, pval = f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)
print(pval)

'''The p-value is 0.00000000019 which is less than 0.05. Therefore, we can conclude that there is at least one pair of chest pain categories that have significantly different thalach.'''

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_results = pairwise_tukeyhsd(heart.thalach, heart.cp,0.05)
print(tukey_results)
'''the significantly different pairs are:
asymptomatic and atypical angina
asymptomatic and non-anginal pain
asymptomatic and typical angina
Therefore, people who are asymptomatic seem to have a lower maximum heart rate (associated with heart disease) than people who have other kinds of chest pain.
'''

# Investigate the relationship between cp(categorical) and heart_disease(categorical).
Xtab = pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab)
from scipy.stats import chi2_contingency
chi2, pval, dof, expected = chi2_contingency(Xtab)
print(pval)

'''The pval is 0.000000000000000013 which is less than 0.05. Therefore, there is a significant assciation between chest pain type and whether or not someone is diagnosed with heart disease.'''

# Create a function to test the hypotheses of the association between two categorical variables.

def is_associated_cat(column):
  #create a cross table of these two variables
  table = pd.crosstab(heart[column], heart.heart_disease)
  print(table)
  #investigate the p-value
  chi2, pval, dof, expected = chi2_contingency(table)
  print(pval)
  if pval < 0.05:
    print('The p-value is significant. Reject the null hypothesis.')
  else:
    print('The p-value is not significant. Cannot reject the null hyphthesis.')

is_associated_cat('sex')
is_associated_cat('exang')
is_associated_cat('fbs')

''' Using a 0.05 significance threshold, both 'sex' (p = 0.0000027) and 'exang' (p = 0.00000000000014) are significantly associated with heart disease. 'fbs' is not significantly associated with heart disease (p = 0.78)'''





