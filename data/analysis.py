# Step => 1 Import required libraries
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MultiLabelBinarizer

# Step => 2 Construct full paths to CSV files (portable across environments)
medicine_df = pd.read_csv("./data/cleaned_medicine_details.csv")
symptoms_df = pd.read_csv("./data/cleaned_disease_and_symptoms.csv")
profile_df = pd.read_csv("./data/cleaned_d_s_a_p_d.csv") 
precaution_df = pd.read_csv("./data/cleaned_disease_precaution.csv")

# Step => 3 Set visual style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
top_diseases = profile_df['Disease'].value_counts().head(10)
sns.barplot(x=top_diseases.values, y=top_diseases.index, palette="Blues_d")
plt.title('Top 10 Diseases (Profile Dataset)')
plt.xlabel('Count')
plt.ylabel('Disease')
if 'Gender' in profile_df.columns:
    plt.subplot(1, 2, 2)
    sns.countplot(x='Gender', data=profile_df, palette='Set2')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 5))
if 'Age' in profile_df.columns:
    plt.subplot(1, 2, 1)
    sns.histplot(profile_df['Age'].dropna(), bins=20, kde=True, color='green')
    plt.title('Age Distribution (Profile)')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
if 'Side_effects' in medicine_df.columns:
    plt.subplot(1, 2, 2)
    all_side_effects = medicine_df['Side_effects'].dropna().str.split(',').explode().str.strip()
    top_side_effects = all_side_effects.value_counts().head(10)
    sns.barplot(x=top_side_effects.values, y=top_side_effects.index, palette='Oranges_d')
    plt.title('Top 10 Reported Side Effects (Medicine)')
    plt.xlabel('Count')
    plt.ylabel('Side Effect')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 5))
if {'Excellent Review %', 'Average Review %', 'Poor Review %'}.issubset(medicine_df.columns):
    plt.subplot(1, 2, 1)
    review_means = medicine_df[['Excellent Review %', 'Average Review %', 'Poor Review %']].mean()
    sns.barplot(x=review_means.index, y=review_means.values, palette='Purples')
    plt.title('Average Medicine Review Ratings (%)')
    plt.ylabel('Average %')
plt.subplot(1, 2, 2)
symptom_matrix = symptoms_df.copy()
symptom_matrix['Symptoms'] = symptom_matrix['Symptoms'].str.split(',')
expanded = symptom_matrix.explode('Symptoms')
expanded['Symptoms'] = expanded['Symptoms'].str.strip()
disease_symptom_matrix = pd.crosstab(expanded['Disease'], expanded['Symptoms'])
if disease_symptom_matrix.shape[1] >= 2:
    corr_matrix = disease_symptom_matrix.corr().round(2)
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, cbar=True)
    plt.title('Symptom Correlation Based on Diseases')
plt.tight_layout()
plt.show()
