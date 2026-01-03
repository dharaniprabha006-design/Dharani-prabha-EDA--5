
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import plotly.express as px

# -----------------------------
# STEP 2: Import Dataset
# -----------------------------
df = pd.read_csv("Combined12.csv")

print("First 5 Rows of Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# -----------------------------
# STEP 3: Rename Columns (map to simple names)
# -----------------------------
df.rename(columns={
    "temp_mean(c)": "Temperature",
    "temp_min(c)": "Temp_Min",
    "temp_max(c)": "Temp_Max",
    "Wind_Speed": "WindSpeed",
    "Wind_Bearing": "WindDirection",
    "global_radiation": "Radiation"
}, inplace=True)

print("\nUpdated Column Names:")
print(df.columns.tolist())

# -----------------------------
# STEP 4: Export Clean Copy
# -----------------------------
df.to_csv("weather_data_copy.csv", index=False)
print("\nDataset exported successfully!")

# -----------------------------
# STEP 5: Data Cleaning
# -----------------------------
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# STEP 6: Outlier Detection
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Temperature"])
plt.title("Outlier Detection - Temperature")
plt.show()

# -----------------------------
# STEP 7: Data Transformation
# -----------------------------
numeric_columns = ['Temperature', 'Temp_Min', 'Temp_Max',
                   'WindSpeed', 'WindDirection', 'Pressure', 'Radiation']

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nData Standardization Completed!")

# -----------------------------
# STEP 8: Descriptive Statisti
# cs
# -----------------------------
print("\nDescriptive Statistics:")
print(df.describe())

print("\nMean Values:")
print(df.mean(numeric_only=True))

print("\nMedian Values:")
print(df.median(numeric_only=True))

print("\nMode Values:")
print(df.mode().iloc[0])

# -----------------------------
# STEP 9: Basic Visualization
# -----------------------------
# Temperature Trend
plt.plot(df['Temperature'])
plt.title("Temperature Trend")
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.show()

# Histogram - Wind Speed
plt.hist(df['WindSpeed'], bins=20)
plt.title("Wind Speed Distribution")
plt.xlabel("Wind Speed")
plt.ylabel("Frequency")
plt.show()

# Bar Plot – Wind Direction Distribution
df['WindDirection'].plot(kind='hist', bins=20)
plt.title("Wind Direction Distribution")
plt.xlabel("Wind Direction")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# STEP 10: Advanced Visualization
# -----------------------------
sns.pairplot(df[['Temperature', 'Temp_Min', 'Temp_Max', 'WindSpeed']])
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.violinplot(x=df['WindSpeed'], y=df['Temperature'])
plt.title("Temperature vs WindSpeed")
plt.show()

# -----------------------------
# STEP 11: Interactive Visualization
# -----------------------------
fig = px.line(df, y='Temperature', title="Interactive Temperature Trend")
fig.show()

# -----------------------------
# STEP 12: Probability Analysis
# -----------------------------
sns.histplot(df['Radiation'], kde=True)
plt.title("Radiation Probability Distribution")
plt.show()

# -----------------------------
# STEP 13: Clustering using K-Means
# -----------------------------
X = df[['Temperature', 'WindSpeed', 'Pressure']]

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.scatter(df['Temperature'], df['WindSpeed'], c=df['Cluster'])
plt.xlabel("Temperature")
plt.ylabel("Wind Speed")
plt.title("Weather Clusters using K-Means")
plt.show()

# -----------------------------
# STEP 14: Summary & Insights
# -----------------------------
print("\nSummary & Insights:")
print("1. Temperature, min/max temperature and wind speed show measurable variation.")
print("2. Radiation distribution indicates variability in sunlight intensity.")
print("3. Correlation heatmap helps identify relationships among features.")
print("4. K-Means clustering groups similar weather conditions based on temperature, wind, and pressure.")

# ================= END OF PROJECT =================