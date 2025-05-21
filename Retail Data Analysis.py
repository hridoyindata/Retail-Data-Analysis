# ================================
# Retail Sales Data Science Project
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 1. Data Cleaning and Feature Engineering
# ----------------------------------------

# Load raw data
df = pd.read_csv('retail_sales_dataset.csv')

# Clean missing values and duplicates
df = df.fillna(0)
df = df.drop_duplicates()

# Create extra columns for analysis
if 'Quantity' in df.columns and 'Price per Unit' in df.columns:
    df['Total Amount'] = df['Quantity'] * df['Price per Unit']

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

# Save cleaned dataset
df.to_csv('retail_sales_dataset_cleaned.csv', index=False)
print("Data cleaning complete. Cleaned data saved.")

# ----------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ----------------------------------------

# Reload cleaned data
df = pd.read_csv('retail_sales_dataset_cleaned.csv')

# --- Product Category Sales ---
cat_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values()
print("\n--- Product Category Sales ---")
print(cat_sales)
cat_sales.plot(kind='bar', title='Total Sales by Product Category')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# --- Sales by Age ---
age_sales = df.groupby('Age')['Total Amount'].sum().sort_values()
print("\n--- Sales by Age ---")
print(age_sales)
age_sales.plot(kind='bar', figsize=(12,4), title='Total Sales by Customer Age')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# --- 3. Trend Analysis: Sales by Month ---
import matplotlib.pyplot as plt

# Group by month and calculate total sales
month_sales = df.groupby('Month')['Total Amount'].sum()

# Month names for labeling on data points
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot
plt.figure(figsize=(10, 5))
plt.plot(month_sales.index, month_sales.values, marker='o')
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(ticks=range(1, 13))  # X-axis stays as numbers 1-12

# Add month name labels on top of each point
for i, value in enumerate(month_sales.values):
    plt.text(x=i+1, y=value + 500, s=month_names[i], ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# --- Sales by Gender ---
gender_sales = df.groupby('Gender')['Total Amount'].sum()
print("\n--- Sales by Gender ---")
print(gender_sales)
gender_sales.plot(kind='pie', autopct='%1.1f%%', title='Sales by Gender', ylabel='')
plt.tight_layout()
plt.show()

# --- Monthly Trends by Category ---
category_month = df.groupby(['Product Category', 'Month'])['Total Amount'].sum().unstack()
print("\n--- Monthly Sales Trend by Category ---")
print(category_month)
category_month.T.plot(figsize=(10,6), marker='o', title='Monthly Sales Trend by Category')
plt.ylabel('Total Sales')
plt.show()

# --- Average Basket Size by Age ---
if 'Gender' in df.columns and 'Total Amount' in df.columns:
    avg_basket_gender = df.groupby('Gender')['Total Amount'].mean()
    print("\n--- Average Basket Size by Gender ---")
    print(avg_basket_gender)
    avg_basket_gender.plot(kind='bar', title='Avg Basket Size by Gender')
    plt.ylabel('Average Basket Size')
    plt.show()
if 'Age' in df.columns and 'Total Amount' in df.columns:
    avg_basket_age = df.groupby('Age')['Total Amount'].mean()
    print("\n--- Average Basket Size by Age ---")
    print(avg_basket_age.sort_values(ascending=False).head(10))
    avg_basket_age.plot(kind='bar', figsize=(12,4), title='Avg Basket Size by Age (Top 10)')
    plt.ylabel('Average Basket Size')
    plt.show()

# --- Category by Gender ---
cat_gender = df.groupby(['Product Category', 'Gender'])['Total Amount'].sum().unstack()
print("\n--- Category Sales by Gender ---")
print(cat_gender)
cat_gender.plot(kind='bar', title='Product Category Sales by Gender')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# --- Heatmap: Product Category by Age ---
cat_age = df.groupby(['Product Category', 'Age'])['Total Amount'].sum().unstack().fillna(0)

# Define a custom color map (blue → green → red → black)
from matplotlib.colors import LinearSegmentedColormap

colors = [(0, 0, 0.6),    # Dark Blue (low sales)
          (0, 0.6, 0),    # Green (moderate sales)
          (1, 0, 0),      # Red (high sales)
          (0, 0, 0)]      # Black (very high sales)

custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

plt.figure(figsize=(12, 6))
ax = sns.heatmap(cat_age, cmap=custom_cmap)
plt.title('Sales Heatmap: Product Category by Age')
plt.xlabel('Age')
plt.ylabel('Product Category')

# Custom legend
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=(0, 0, 0.6), label='Low Sales (Blue)'),
    mpatches.Patch(color=(0, 0.6, 0), label='Moderate Sales (Green)'),
    mpatches.Patch(color=(1, 0, 0), label='High Sales (Red)'),
    mpatches.Patch(color=(0, 0, 0), label='Very High Sales (Black)')
]
plt.legend(handles=legend_patches, title="Sales Volume", bbox_to_anchor=(0.5, 0.5), loc='upper left')

plt.tight_layout()
plt.show()
# ----------------------------------------
# 3. Advanced Analysis
# ----------------------------------------

# --- Repeat Purchase Rate by Age ---
customer_counts = df.groupby('Customer ID').size()
repeat_customers = customer_counts[customer_counts > 1].index
repeat_ages = df[df['Customer ID'].isin(repeat_customers)].groupby('Age').size()
all_ages = df.groupby('Age').size()
repeat_rate = (repeat_ages / all_ages).fillna(0)
print("\n--- Repeat Purchase Rate by Age ---")
print(repeat_rate.sort_values(ascending=False))
repeat_rate.plot(kind='bar', title='Repeat Purchase Rate by Age', figsize=(12,4))
plt.ylabel('Repeat Rate')
plt.show()

# --- Price per Unit vs. Quantity Sold ---
plt.scatter(df['Price per Unit'], df['Quantity'])
plt.title('Price per Unit vs. Quantity Sold')
plt.xlabel('Price per Unit')
plt.ylabel('Quantity Sold')
plt.show()
corr = df['Price per Unit'].corr(df['Quantity'])
print(f"\nCorrelation between Price per Unit and Quantity Sold: {corr:.2f}")

# ----------------------------------------
# 4. Predictive Modeling
# ----------------------------------------

# -- Decision Tree Classifier for High-Value Transactions --
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Label High/Low value
median_value = df['Total Amount'].median()
df['High_Value'] = (df['Total Amount'] > median_value).astype(int)

features = ['Age', 'Gender', 'Product Category', 'Month']
X = df[features].copy()
for col in ['Gender', 'Product Category']:
    X[col] = LabelEncoder().fit_transform(X[col])
y = df['High_Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
print(f"\nDecision Tree Accuracy: {clf.score(X_test, y_test):.2f}")

plt.figure(figsize=(18, 8))
plot_tree(clf, feature_names=features, class_names=['Low', 'High'], filled=True)
plt.title("Decision Tree to Predict High Value Transactions")
plt.show()

importances = pd.Series(clf.feature_importances_, index=features)
print("\nDecision Tree Feature Importances:")
print(importances.sort_values(ascending=False))

# -- Random Forest Regression for Sales Prediction --
df_model = df.copy()
for col in ['Gender', 'Product Category']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])
df_model['Age2'] = df_model['Age'] ** 2
df_model['Age_Product'] = df_model['Age'] * df_model['Product Category']

features_rf = ['Age', 'Age2', 'Gender', 'Product Category', 'Month', 'Age_Product']
X_rf = df_model[features_rf]
y_rf = df_model['Total Amount']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)

# Linear Regression
lr = LinearRegression().fit(X_train_rf, y_train_rf)
lr_pred = lr.predict(X_test_rf)
lr_rmse = mean_squared_error(y_test_rf, lr_pred, squared=False)
lr_r2 = r2_score(y_test_rf, lr_pred)
print(f"\nLinear Regression RMSE: {lr_rmse:.2f}, R²: {lr_r2:.3f}")
print("\nLinear Regression Coefficients:")
print(pd.Series(lr.coef_, index=features_rf).sort_values(ascending=False))

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_rf, y_train_rf)
rf_pred = rf.predict(X_test_rf)
rf_rmse = mean_squared_error(y_test_rf, rf_pred, squared=False)
rf_r2 = r2_score(y_test_rf, rf_pred)
print(f"\nRandom Forest RMSE: {rf_rmse:.2f}, R²: {rf_r2:.3f}")
importances_rf = pd.Series(rf.feature_importances_, index=features_rf)
print("\nRandom Forest Feature Importances:")
print(importances_rf.sort_values(ascending=False))

plt.figure(figsize=(9,5))
importances_rf.sort_values().plot(kind='barh')
plt.title('Random Forest Feature Importances for Predicting Total Sales Amount', fontsize=14)
plt.xlabel('Importance Score (Relative Contribution to Prediction)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n--- All analyses completed successfully. ---")
