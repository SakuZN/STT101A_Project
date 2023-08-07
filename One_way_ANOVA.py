import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# Load the data
data = pd.read_csv('Food Delivery Time Prediction Case Study.csv')

# Perform one-way ANOVA
model = ols('Q("Time_taken(min)") ~ C(Q("Type_of_vehicle"))', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

sns.boxplot(x="Type_of_vehicle", y="Time_taken(min)", data=data, width=0.5)
plt.title('Boxplot grouped by Type_of_vehicle')
plt.tight_layout()
plt.savefig('graphs/ANOVA_boxplot_grouped_by_type_of_vehicle.png')
plt.show()
