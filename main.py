# Advanced EDA For Genomic Data Analysis: Identifying Genetic Variations Through Visualization

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title('Advanced EDA For Genomic Data Analysis')
# File upload section
uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    np.random.seed(0)
    synthetic_data = np.random.rand(100, 10)
    df = pd.DataFrame(synthetic_data, columns=[f'Gene_{i+1}' for i in range(10)])
    df['Target'] = np.random.choice([0, 1], size=100)
    df['Age'] = np.random.randint(20, 80, size=100)
    df['Gender'] = np.random.choice(['Male', 'Female'], size=100)
    df['Ethnicity'] = np.random.choice(['Ethnicity_A', 'Ethnicity_B', 'Ethnicity_C'], size=100)

# Convert non-numeric columns to numeric, if possible, or drop them
df_numeric = df.select_dtypes(include=[np.number])

# Display the data
st.write('### Data Preview')
st.write(df.head())
st.write('Shape of the dataset:', df.shape)
# Summary Statistics
st.write('### Basic Exploratory Data Analysis')
st.write('Summary Statistics:')
st.write(df_numeric.describe())

# Correlation Heatmap
corr = df_numeric.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

if 'Target' in df.columns:
    st.subheader("Basic EDA")
    st.write("Missing Values:", df.isnull().sum().sum())
    st.write("Class Distribution")
    st.bar_chart(df['Target'].value_counts())

    # SNP Distribution Visualization
    st.subheader("SNP Variant Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df.iloc[:, 0], bins=30, kde=False, ax=ax)
    ax.set_title("Distribution of SNP 0 Variants")
    st.pyplot(fig)
# PCA Visualization
st.subheader("PCA Visualization")
pca = PCA(n_components=2)
pca_results = pca.fit_transform(df.drop(columns=['Target', 'Age', 'Gender', 'Ethnicity']))
fig, ax = plt.subplots()
sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=df['Target'], palette='coolwarm', ax=ax)
ax.set_title("PCA Visualization of Genomic Data")
st.pyplot(fig)

# t-SNE Visualization
st.subheader("t-SNE Visualization")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(df.drop(columns=['Target', 'Age', 'Gender', 'Ethnicity']))
fig, ax = plt.subplots()
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=df['Target'], palette='coolwarm', ax=ax)
ax.set_title("t-SNE Visualization of Genomic Data")
st.pyplot(fig)
# Machine Learning - Random Forest Classifier
st.subheader("Machine Learning Model: Random Forest")
y = df['Target']
X = df.drop(columns=['Target', 'Age', 'Gender', 'Ethnicity'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write("### Conclusion")
st.write("In this project, we performed advanced exploratory data analysis (EDA) on genomic data, identified genetic variations through visualization, and built a machine learning model for classification. The visualizations and model help in understanding the data and its patterns more effectively.")
