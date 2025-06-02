# Step => 1 Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# Step => 2 Load all datasets
def load_healthcare_datasets():
    medicine_df = pd.read_csv("./dataset/medicine_details.csv")
    disease_symptom_df = pd.read_csv("./dataset/disease_and_symptoms.csv")
    patient_profile_df = pd.read_csv("./dataset/disease_symptom_and_patient_profile_dataset.csv")
    precaution_df = pd.read_csv("./dataset/disease_precaution.csv")
    return medicine_df, disease_symptom_df, patient_profile_df, precaution_df

# Step => 3 Fill missing values using median, mode, or unknown values
def fill_missing_values(medicine_df, disease_symptom_df, patient_profile_df):
    patient_profile_df['Age'] = patient_profile_df['Age'].fillna(patient_profile_df['Age'].median())
    for col in ['Excellent Review %', 'Average Review %', 'Poor Review %']:
        medicine_df[col] = medicine_df[col].fillna(medicine_df[col].median())
    disease_symptom_df['Symptom_4'] = disease_symptom_df['Symptom_4'].fillna(disease_symptom_df['Symptom_4'].mode()[0])
    symptom_cols_5_12 = [f'Symptom_{i}' for i in range(5, 13)]
    disease_symptom_df[symptom_cols_5_12] = disease_symptom_df[symptom_cols_5_12].fillna('Unknown')
    return medicine_df, disease_symptom_df, patient_profile_df

# Step => 4 Drop unwanted columns and duplicates
def drop_unwanted_and_duplicates(medicine_df, disease_symptom_df, patient_profile_df, precaution_df):
    columns_to_drop = [f'Symptom_{i}' for i in range(13, 18)]
    disease_symptom_df = disease_symptom_df.drop(columns=columns_to_drop)
    return (
        medicine_df.drop_duplicates(),
        disease_symptom_df.drop_duplicates(),
        patient_profile_df.drop_duplicates(),
        precaution_df.drop_duplicates()
    )

# Step => 5 Remove outliers using Z-score and IQR methods
def remove_outliers(medicine_df, patient_profile_df):
    for col in ['Excellent Review %', 'Average Review %', 'Poor Review %']:
        z_scores = zscore(medicine_df[col])
        medicine_df = medicine_df[(abs(z_scores) <= 3)]
    Q1 = patient_profile_df['Age'].quantile(0.25)
    Q3 = patient_profile_df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    patient_profile_df = patient_profile_df[(patient_profile_df['Age'] >= lower_bound) & (patient_profile_df['Age'] <= upper_bound)]
    return medicine_df, patient_profile_df

# Step => 6 Encode categorical(string labels) columns into numbers
def encode_categorical_columns(medicine_df, disease_symptom_df, patient_profile_df, precaution_df):
    le = LabelEncoder()
    if 'Manufacturer' in medicine_df.columns:
        medicine_df['Manufacturer'] = le.fit_transform(medicine_df['Manufacturer'])
    encode_cols = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']
    for col in encode_cols:
        if col in patient_profile_df.columns:
            patient_profile_df[col] = le.fit_transform(patient_profile_df[col])
    if 'Disease' in precaution_df.columns:
        precaution_df['Disease'] = le.fit_transform(precaution_df['Disease'])
    symptom_cols = ['Disease'] + [f'Symptom_{i}' for i in range(1, 13)]
    for col in symptom_cols:
        if col in disease_symptom_df.columns:
            disease_symptom_df[col] = le.fit_transform(disease_symptom_df[col].astype(str))
    return medicine_df, disease_symptom_df, patient_profile_df, precaution_df

# Step => 7 Save cleaned data into new CSV files
def save_cleaned_datasets(medicine_df, disease_symptom_df, patient_profile_df, precaution_df):
    medicine_df.to_csv('cleaned_medicine_details.csv', index=False)
    disease_symptom_df.to_csv('cleaned_disease_and_symptoms.csv', index=False)
    patient_profile_df.to_csv('cleaned_d_s_a_p_d.csv', index=False)
    precaution_df.to_csv('cleaned_disease_precaution.csv', index=False)

# Step => 8 Execute all steps in order and clean all datasets
def preprocess_healthcare_data():
    med_df, symp_df, pat_df, prec_df = load_healthcare_datasets()
    med_df, symp_df, pat_df = fill_missing_values(med_df, symp_df, pat_df)
    med_df, symp_df, pat_df, prec_df = drop_unwanted_and_duplicates(med_df, symp_df, pat_df, prec_df)
    med_df, pat_df = remove_outliers(med_df, pat_df)
    med_df, symp_df, pat_df, prec_df = encode_categorical_columns(med_df, symp_df, pat_df, prec_df)
    save_cleaned_datasets(med_df, symp_df, pat_df, prec_df)

# Step => 9 Call the main function
preprocess_healthcare_data()
