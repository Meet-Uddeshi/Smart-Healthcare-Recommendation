# Step 1 => Import required libraries
import pandas as pd

# Step 2 => Load datasets
medicine_df = pd.read_csv('./data/cleaned_medicine_details.csv')
disease_symptoms_df = pd.read_csv('./data/cleaned_disease_and_symptoms.csv')
disease_profile_df = pd.read_csv('./data/cleaned_d_s_a_p_d.csv')
disease_precaution_df = pd.read_csv('./data/cleaned_disease_precaution.csv')

# Step 3 => Extract medicine composition list
medicine_df['Composition_List'] = medicine_df['Composition'].str.split('+')

# Step 4 => Count number of side effects
medicine_df['Num_Side_Effects'] = medicine_df['Side_effects'].apply(
    lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
)

# Step 5 => Define chronic diseases list (example list, expand as needed)
chronic_diseases = ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis']

# Step 6 => Classify disease as chronic or acute in disease_symptoms_df
disease_symptoms_df['Disease_Type'] = disease_symptoms_df['Disease'].apply(
    lambda x: 'Chronic' if x in chronic_diseases else 'Acute'
)

# Step 7 => Also classify diseases in disease_profile_df
disease_profile_df['Disease_Type'] = disease_profile_df['Disease'].apply(
    lambda x: 'Chronic' if x in chronic_diseases else 'Acute'
)

# Step 8 => Save feature engineered medicine details
medicine_df.to_csv('./data/feature_engineered_medicine_details.csv', index=False)

# Step 9 => Save disease symptoms with classification
disease_symptoms_df.to_csv('./data/classified_disease_and_symptoms.csv', index=False)

# Step 10 => Save disease profile with classification
disease_profile_df.to_csv('./data/classified_disease_profile.csv', index=False)