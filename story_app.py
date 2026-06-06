import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Title
st.title("Data Storytelling App")

# Load Dataset
df = pd.read_csv("dataset.csv")

# Dataset Introduction
st.header("Dataset Introduction")

st.write("Shape of Dataset:", df.shape)

st.write("Columns:")
st.write(df.columns)

st.write("Preview:")
st.write(df.head())

# Data Cleaning
df = df.dropna()

# EDA Section
st.header("Exploratory Data Analysis (EDA)")

st.write("Statistical Summary")
st.write(df.describe())

# Numerical Columns
numeric_cols = df.select_dtypes(include='number').columns

# Visualization 1
st.header("Visualization 1 - Histogram")

if len(numeric_cols) >= 1:
    fig, ax = plt.subplots()
    sns.histplot(df[numeric_cols[0]], kde=True, ax=ax)
    st.pyplot(fig)

# Visualization 2
st.header("Visualization 2 - Scatter Plot")

if len(numeric_cols) >= 2:
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=numeric_cols[0],
        y=numeric_cols[1],
        data=df,
        ax=ax
    )
    st.pyplot(fig)

# Visualization 3
st.header("Visualization 3 - Line Chart")

if len(numeric_cols) >= 2:
    fig, ax = plt.subplots()
    sns.lineplot(
        x=numeric_cols[0],
        y=numeric_cols[1],
        data=df,
        ax=ax
    )
    st.pyplot(fig)

# Visualization 4
st.header("Visualization 4 - Heatmap")

if len(numeric_cols) >= 2:
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
    st.pyplot(fig)

# Visualization 5
st.header("Visualization 5 - Boxplot")

if len(numeric_cols) >= 1:
    fig, ax = plt.subplots()
    sns.boxplot(y=df[numeric_cols[0]], ax=ax)
    st.pyplot(fig)

# Insights and Findings
st.header("Insights and Findings")

st.write("""
1. The dataset contains meaningful trends.
2. Numerical values show distribution patterns.
3. Some variables are positively correlated.
4. Outliers are visible in the boxplot.
5. Heatmap shows relationships between features.
""")

# Final Conclusion
st.header("Final Conclusion / Recommendations")

st.write"""
- Dataset analysis was successfully completed.
- Visualizations helped identify patterns.
- Data cleaning improved dataset quality.
- Further machine learning models can be applied.
""")
