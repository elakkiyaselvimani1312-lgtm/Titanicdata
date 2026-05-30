import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# First 5 rows
print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True)

# Dataset information
print("\nDataset Info:")
print(df.info())

# Statistical Summary
print("\nSummary:")
print(df.describe())

# Survival Count
df['Survived'].value_counts().plot(kind='bar')
plt.title("Survival Count")
plt.show()

# Gender vs Survival
pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar')
plt.title("Gender vs Survival")
plt.show()

# Age Distribution
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()