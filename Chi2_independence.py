import pandas as pd
from scipy.stats import chi2_contingency, chi2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# Load the data
data = pd.read_csv('Food Delivery Time Prediction Case Study.csv')

# Create a contingency table
contingency_table = pd.crosstab(data['Type_of_order'], data['Type_of_vehicle'])

# Perform the Chi-square test of independence
chi2val, p, dof, expected = chi2_contingency(contingency_table)
# Also get critical value for 95% confidence*
significance_level = 0.05
critical_value = chi2.ppf(1 - significance_level, dof)
print('Chi-square statistic: {:.3f}'.format(chi2val))
print('Critical value: {:.3f}'.format(critical_value))
print('p-value: {:.3f}'.format(p))
print('Degrees of freedom: {}'.format(dof))
print('Expected values: \n{}'.format(expected))
print('Contingency table: \n{}'.format(contingency_table))
#Also output the sum per row and column
print('Sum per row: \n{}'.format(contingency_table.sum(axis=1)))
print('Sum per column: \n{}'.format(contingency_table.sum(axis=0)))
# Output total sum
print('Total sum: {:.0f}'.format(contingency_table.sum().sum()))

# Heatmap
sns.heatmap(contingency_table, annot=True, fmt="d", cmap='viridis')
plt.title('Observed Contingency Table')
plt.tight_layout()
plt.savefig('graphs/chi2_observed_contingency_table.png')
plt.show()

sns.heatmap(expected, annot=True, fmt="0.3f", cmap='viridis')
plt.title('Expected Contingency Table')
plt.tight_layout()
plt.savefig('graphs/chi2_expected_contingency_table.png')
plt.show()

# Barplot for sum per row and column
contingency_table.sum(axis=1).plot(kind='bar',
                                   color='blue',
                                   alpha=0.75,
                                   title='Sum per row',
                                   )
plt.tight_layout()
plt.savefig('graphs/chi2_sum_per_row.png')
plt.show()

contingency_table.sum(axis=0).plot(kind='bar',
                                   color='green',
                                   alpha=0.75,
                                   title='Sum per column'
                                   )
plt.tight_layout()
plt.savefig('graphs/chi2_sum_per_column.png')
plt.show()

# Grouped bar chart
plt.figure(figsize=(12, 8))

grouped = data.groupby(['Type_of_order', 'Type_of_vehicle']).size().unstack()

grouped.plot(kind='bar', stacked=False)

plt.title('Bar Chart of Type_of_order vs Type_of_vehicle', fontsize=15, pad=20)
plt.xlabel('Type_of_order', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Type_of_vehicle', title_fontsize='11', fontsize='11', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('graphs/chi2_grouped_bar_chart.png', bbox_inches='tight')
plt.show()
