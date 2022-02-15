
# run this in python interpreter 

from statsmodels.stats.proportion import proportions_ztest
stat, p_value = proportions_ztest(count=60, nobs=100, value=0.5, alternative='larger')
print(p_value)
