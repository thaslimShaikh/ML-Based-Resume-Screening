import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("C:/Users/786sh/OneDrive/Desktop/ML_Project/resume_screening_dataset.csv")

# Step 1: Removal of Corrupted Data
numerical_columns = ['years_of_experience', 'expected_salary', 'technical_test_score']
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert errors to NaN

# Remove rows with corrupted data (NaN values)
df.dropna(subset=numerical_columns, inplace=True)

# Step 2: Handling Missing Values
# 2.1. Impute categorical columns with the most frequent value
categorical_columns = ['skills', 'highest_education', 'job_applied_for', 'current_location']
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

# 2.2. Impute numerical columns with the mean
numerical_imputer = SimpleImputer(strategy='mean')
df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

# Step 3: Data Visualization and Exploration
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_columns])
plt.title('Box Plot of Numerical Features')
plt.show()

df[numerical_columns].hist(bins=20, figsize=(10, 6))
plt.suptitle('Distribution of Numerical Features')
plt.show()

print("Data Exploration:")
print(df[numerical_columns].describe())
print("\nMode:")
print(df[numerical_columns].mode())
print("\nMean:")
print(df[numerical_columns].mean())
print("\nMedian:")
print(df[numerical_columns].median())
print("\nVariance:")
print(df[numerical_columns].var())
print("\nStandard Deviation:")
print(df[numerical_columns].std())

# Step 4: Outlier Detection
z_scores = np.abs(zscore(df[numerical_columns]))
df['z_score'] = z_scores.max(axis=1)
outliers = df[df['z_score'] > 3]
print(f"Outliers detected: {len(outliers)}")
df_cleaned = df[df['z_score'] <= 3].copy()
df_cleaned.drop('z_score', axis=1, inplace=True)

# Step 5: Class Imbalance Problem
# Check if 'application_status' exists
if 'application_status' not in df_cleaned.columns:
    raise KeyError("The 'application_status' column is missing from the dataset.")

class_distribution = df_cleaned['application_status'].value_counts()
print("\nClass Distribution:")
print(class_distribution)

# Step 6: Feature Analysis
# Convert the categorical target column to numeric labels for visualization
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_cleaned['application_status'])

# PCA for Feature Extraction
X = df_cleaned.drop('application_status', axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding of categorical features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA result
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', alpha=0.7)
plt.title('PCA Projection of Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Feature Importance using RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X, y_encoded)
feature_importance = pd.DataFrame(model.feature_importances_,
                                  index=X.columns,
                                  columns=['importance']).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.head(10))

# Step 7: Data Normalization
# Normalizing using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizing using MinMaxScaler
min_max_scaler = MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)

# Normalizing using RobustScaler
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)

print("Data preprocessing completed successfully!")
