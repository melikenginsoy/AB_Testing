########################################################
# Comparison of Bidding Methods Conversion with AB Test
########################################################

##################
# Business Problem
##################

"""
Company-A recently introduced a new bidding type, 'average bidding', as an alternative to
the existing bidding type called 'maximum bidding'.One of our customers decided to test this new feature
and they expects us to analyze the results by doing an A/B test.
"""

##################
# Dataset Story
##################

"""
In this dataset, which includes the website information of a company, there is information
such as the number of advertisements that users see and click, as well as earnings information from here.
There are two separate data sets, the control and test groups.
"""

# 4 Variable           40 Observation              26 KB

# Impression  :Ad views
# Click       :Number of clicks on the displayed ad
# Purchase    :Number of products purchased after ads clicked
# Earning     :Earnings after purchased products


###############################
# Preparing and Analyzing Data
###############################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import MultiComparison

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# We are converting two separate dataframes by reading the dataset consisting of control and test group data.

control_df = pd.read_excel("Projects/AB_Testing/ab_testing.xlsx", sheet_name="Control Group")
test_df = pd.read_excel("Projects/AB_Testing/ab_testing.xlsx", sheet_name="Test Group")

control_df.describe().T
"""
              count         mean         std         min         25%         50%          75%          max
Impression 40.00000 101711.44907 20302.15786 45475.94296 85726.69035 99790.70108 115212.81654 147539.33633
Click      40.00000   5100.65737  1329.98550  2189.75316  4124.30413  5001.22060   5923.80360   7959.12507
Purchase   40.00000    550.89406   134.10820   267.02894   470.09553   531.20631    637.95709    801.79502
Earning    40.00000   1908.56830   302.91778  1253.98952  1685.84720  1975.16052   2119.80278   2497.29522
"""

test_df.describe().T
"""
              count         mean         std         min          25%          50%          75%          max
Impression 40.00000 120512.41176 18807.44871 79033.83492 112691.97077 119291.30077 132050.57893 158605.92048
Click      40.00000   3967.54976   923.09507  1836.62986   3376.81902   3931.35980   4660.49791   6019.69508
Purchase   40.00000    582.10610   161.15251   311.62952    444.62683    551.35573    699.86236    889.91046
Earning    40.00000   2514.89073   282.73085  1939.61124   2280.53743   2544.66611   2761.54540   3171.48971
"""

# We add group variable to both dataframes and concat them.

control_df["Group"] = "Control"
test_df["Group"] = "Test"
df = pd.concat([control_df, test_df], ignore_index=True)

"""
     Impression      Click  Purchase    Earning    Group
0   82529.45927 6090.07732 665.21125 2311.27714  Control
1   98050.45193 3382.86179 315.08489 1742.80686  Control
2   82696.02355 4167.96575 458.08374 1797.82745  Control
3  109914.40040 4910.88224 487.09077 1696.22918  Control
4  108457.76263 5987.65581 441.03405 1543.72018  Control
..          ...        ...       ...        ...      ...
75  79234.91193 6002.21358 382.04712 2277.86398     Test
76 130702.23941 3626.32007 449.82459 2530.84133     Test
77 116481.87337 4702.78247 472.45373 2597.91763     Test
78  79033.83492 4495.42818 425.35910 2595.85788     Test
79 102257.45409 4800.06832 521.31073 2967.51839     Test
[80 rows x 5 columns]
"""
df.info()

"""
#   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Impression  80 non-null     float64
 1   Click       80 non-null     float64
 2   Purchase    80 non-null     float64
 3   Earning     80 non-null     float64
 4   Group       80 non-null     object 
dtypes: float64(4), object(1)
"""

df.groupby("Group")["Purchase"].mean()
"""
Group
Control   550.89406
Test      582.10610
"""
######################
# A/B Test
######################

# 1. Define hypotheses
# 2. Assumption Control
#   - 1. Normality Assumption
#   - 2. Variance Homogeneity
# 3. Application of the Hypothesis


# 1. Defining the A/B Test Hypotheses:

    # H0 : M1 = M2
    # H1 : M1!= M2

    # H0: There is no statistically significant difference between the purchase averages
    # for the Control and Test groups.

# 2. Assumption Control:

    # Normality assumption of control and test group:
    # H0: Normal distribution assumption is provided.
    # H1: The assumption of normal distribution is not provided.

test_stat, pvalue = shapiro(df.loc[df["Group"] == "Control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    # Test Stat = 0.9773, p-value = 0.5891

test_stat, pvalue = shapiro(df.loc[df["Group"] == "Test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    # Test Stat = 0.9589, p-value = 0.1541

    # H0 cannot be rejected since p value > 0.05 in both groups.
    # The assumption of normal distribution is provided.

    # Variance Homogeneity:
    # H0: Variances are homogeneous.
    # H1: Variances are not homogeneous.

test_stat, pvalue = levene(df.loc[df["Group"] == "Control", "Purchase"],
                           df.loc[df["Group"] == "Test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    # Test Stat = 2.6393, p-value = 0.1083
    # H0 cannot be rejected because p value > 0.05.
    # Variances are homogeneous.

# 3. Application of the Hypothesis:

    # Since assumption of normality and homogeneity of variance are provided,
    # the parametric t test is used.
test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "Control", "Purchase"],
                              df.loc[df["Group"] == "Test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    # Test Stat = -0.9416, p-value = 0.3493

    # H0 : M1 = M2
    # H1 : M1!= M2

    # Since p value > 0.05, the H0 hypothesis cannot be rejected.

    # There is no statistically significant difference between the purchase averages
    # for the Control and Test groups, with a 95% confidence interval and a 5% margin of error.