import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
import numpy as np
import pprint
np.set_printoptions(precision=3, suppress=True)

# Load the data
data = pd.read_csv('Food Delivery Time Prediction Case Study.csv')

# Perform descriptive statistics

# Age: Median
age_median = data['Delivery_person_Age'].median()

# Rating: Mean
rating_mean = data['Delivery_person_Ratings'].mean()

# Restaurant Latitude and Longitude: Mean, Standard Deviation
restaurant_lat_mean = data['Restaurant_latitude'].mean()
restaurant_lat_std = data['Restaurant_latitude'].std()
restaurant_long_mean = data['Restaurant_longitude'].mean()
restaurant_long_std = data['Restaurant_longitude'].std()

# Delivery Latitude and Longitude: Mean, Standard Deviation
delivery_lat_mean = data['Delivery_location_latitude'].mean()
delivery_lat_std = data['Delivery_location_latitude'].std()
delivery_long_mean = data['Delivery_location_longitude'].mean()
delivery_long_std = data['Delivery_location_longitude'].std()

# Type of Order :Mode, Relative frequency
order_type_mode = data['Type_of_order'].mode()[0]
order_type_rel_freq = data['Type_of_order'].value_counts(normalize=True)

# Type of Vehicle :Mode, Relative frequency
vehicle_type_mode = data['Type_of_vehicle'].mode()[0]
vehicle_type_rel_freq = data['Type_of_vehicle'].value_counts(normalize=True)

# Time Taken :Median, Standard Deviation
time_taken_median = data['Time_taken(min)'].median()
time_taken_std = data['Time_taken(min)'].std()

var = {
    "age_median": age_median,
    "rating_mean": rating_mean,
    "restaurant_lat_mean": restaurant_lat_mean,
    "restaurant_lat_std": restaurant_lat_std,
    "restaurant_long_mean": restaurant_long_mean,
    "restaurant_long_std": restaurant_long_std,
    "delivery_lat_mean": delivery_lat_mean,
    "delivery_lat_std": delivery_lat_std,
    "delivery_long_mean": delivery_long_mean,
    "delivery_long_std": delivery_long_std,
    "order_type_mode": order_type_mode,
    "order_type_rel_freq": order_type_rel_freq,
    "vehicle_type_mode": vehicle_type_mode,
    "vehicle_type_rel_freq": vehicle_type_rel_freq,
    "time_taken_median": time_taken_median,
    "time_taken_std": time_taken_std
}
pprint.pprint(var)

# Plot Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Delivery_person_Age'], kde=False, color="skyblue")
plt.title('Age Distribution')
plt.tight_layout()
plt.savefig('graphs/Descriptive_age_distribution.png')
plt.show()

# Plot Ratings distribution
plt.figure(figsize=(10, 8))
sns.histplot(data['Delivery_person_Ratings'], kde=False, color="olive", bins=30)
plt.title('Ratings Distribution', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('graphs/Descriptive_ratings_distribution.png')
plt.show()

# Plot Time Taken distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Time_taken(min)'], kde=False, bins=30, color="gold")
plt.title('Time Taken Distribution')
plt.tight_layout()
plt.savefig('graphs/Descriptive_time_taken_distribution.png')
plt.show()

# Plot Type of Order distribution
plt.figure(figsize=(8, 6))
data['Type_of_order'].value_counts().plot(kind='bar', color='c')
plt.title('Type of Order Distribution')
plt.tight_layout()
plt.savefig('graphs/Descriptive_type_of_order_distribution.png')
plt.show()

# Plot Type of Vehicle distribution
plt.figure(figsize=(7, 5))
data['Type_of_vehicle'].value_counts().plot(kind='bar', color='c')
plt.title('Type of Vehicle Distribution')
plt.tight_layout()
plt.savefig("graphs/Descriptive_Type_of_Vehicle_Distribution.png")
plt.show()

# Plot Restaurant Location distribution
plt.figure(figsize=(7, 5))
sns.scatterplot(x=data['Restaurant_longitude'], y=data['Restaurant_latitude'], color='r')
plt.title('Restaurant Location Distribution')
plt.tight_layout()
plt.savefig("graphs/Descriptive_Restaurant_Location_Distribution.png")
plt.show()

# Plot Delivery Location distribution
plt.figure(figsize=(7, 5))
sns.scatterplot(x=data['Delivery_location_longitude'], y=data['Delivery_location_latitude'], color='g')
plt.title('Delivery Location Distribution')
plt.tight_layout()
plt.savefig("graphs/Descriptive_Delivery_Location_Distribution.png")
plt.show()
