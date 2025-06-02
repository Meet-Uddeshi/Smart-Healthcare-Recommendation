from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

# Step => 1 Load datasets
disease_symptoms_df = pd.read_csv('./dataset/disease_and_symptoms.csv')
medicine_df = pd.read_csv('./data/cleaned_medicine_details.csv')

# Step => 2 Preprocess data
symptom_cols = [col for col in disease_symptoms_df.columns if col.startswith('Symptom_')]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(disease_symptoms_df[col].dropna().str.strip().str.lower().unique())
all_symptoms = sorted(all_symptoms)
def combine_symptoms(row):
    symptoms = []
    for col in symptom_cols:
        val = row[col]
        if pd.notna(val) and val.strip() != '':
            symptoms.append(val.strip().lower())
    return ' '.join(symptoms)
disease_symptoms_df['symptom_text'] = disease_symptoms_df.apply(combine_symptoms, axis=1)

# Step => 3 Load trained vectorizer
with open('./models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Step => 4 Vectorize disease symptoms
disease_vectors = vectorizer.transform(disease_symptoms_df['symptom_text'])

# Step => 5 Preprocess medicine data
medicine_df['Uses'] = medicine_df['Uses'].fillna('').str.lower()

# Step => 6 Define medicine recommendation function
def recommend_medicines(disease_name, top_n=5):
    disease_keywords = disease_name.lower().split()
    filtered = medicine_df[
        medicine_df['Uses'].apply(lambda x: any(keyword in x for keyword in disease_keywords))
    ]
    if filtered.empty:
        return medicine_df[['Medicine Name', 'Uses', 'Manufacturer']].head(top_n).to_dict(orient='records')
    return filtered[['Medicine Name', 'Uses', 'Manufacturer']].head(top_n).to_dict(orient='records')

# Step => 7 Homepage route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',
                           symptoms=all_symptoms,
                           selected_symptoms=[],
                           age='',
                           gender='',
                           error=None,
                           disease=None,
                           result=None)

# Step => 8 Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    age = request.form.get('age')
    gender = request.form.get('gender')
    symptoms = request.form.getlist('symptoms')
    if not age or not gender or not symptoms:
        return render_template('index.html',
                               symptoms=all_symptoms,
                               selected_symptoms=symptoms,
                               age=age,
                               gender=gender,
                               error="Please fill all fields.",
                               disease=None,
                               result=None)
    symptom_text = ' '.join([s.strip().lower() for s in symptoms])
    symptom_vec = vectorizer.transform([symptom_text])
    similarities = cosine_similarity(symptom_vec, disease_vectors).flatten()
    best_idx = similarities.argmax()
    predicted_disease = disease_symptoms_df.iloc[best_idx]['Disease']
    recommendations = recommend_medicines(predicted_disease)
    return render_template('index.html',
                           symptoms=all_symptoms,
                           selected_symptoms=symptoms,
                           age=age,
                           gender=gender,
                           error=None,
                           disease=predicted_disease,
                           result=recommendations)
if __name__ == '__main__':
    app.run(debug=True)
