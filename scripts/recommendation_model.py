from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__, template_folder='templates', static_folder='static')

# Step => 1 Load all the needed data/models once on startup
with open('./models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('./models/disease_vectors.pkl', 'rb') as f:
    disease_vectors = pickle.load(f)
with open('./models/medicine_vectors.pkl', 'rb') as f:
    medicine_vectors = pickle.load(f)
with open('./models/disease_symptoms_df.pkl', 'rb') as f:
    disease_symptoms_df = pickle.load(f)
with open('./models/medicine_df.pkl', 'rb') as f:
    medicine_df = pickle.load(f)
    
def recommend_medicines(disease_name, top_n=5):
    disease_name = disease_name.lower()
    disease_idx = disease_symptoms_df[disease_symptoms_df['Disease'].str.lower() == disease_name].index
    if len(disease_idx) == 0:
        return []  # disease not found
    disease_idx = disease_idx[0]
    disease_vec = disease_vectors[disease_idx]
    similarities = cosine_similarity(disease_vec, medicine_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    recommended_meds = medicine_df.iloc[top_indices]['Medicine Name'].tolist()
    return recommended_meds

@app.route('/')
def home():
    return render_template('index.html', result=None)
@app.route('/recommend', methods=['POST'])
def recommend():
    disease = request.form.get('disease')
    if not disease:
        return render_template('index.html', result=[], error="Please enter a disease name.")
    
    recommendations = recommend_medicines(disease, top_n=7)
    if not recommendations:
        return render_template('index.html', result=[], error="Disease not found or no recommendations.")
    
    return render_template('index.html', result=recommendations, disease=disease)

if __name__ == '__main__':
    app.run(debug=True)
