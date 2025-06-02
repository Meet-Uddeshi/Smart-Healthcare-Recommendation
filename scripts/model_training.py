# Step => 1 Importing required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Step => 2 Loading and preparing the medicine dataset
def load_and_prepare_medicine_data(filepath):
    df = pd.read_csv(filepath)
    df.fillna('', inplace=True)
    def parse_composition_list(x):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            try:
                return ' '.join(eval(x))
            except:
                return x
        return x
    df['Composition_List'] = df['Composition_List'].apply(parse_composition_list)
    df['combined_features'] = df['Uses'] + ' ' + df['Composition_List'] + ' ' + df['Side_effects']
    return df

# Step => 3 Creating TF-IDF matrix for content-based features
def create_tfidf_matrix(df, feature_column='combined_features'):
    # Create TF-IDF matrix from combined text features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[feature_column])
    return tfidf_vectorizer, tfidf_matrix

# Step => 4 Recommend top-N medicines based on disease and review preference
def recommend_medicines_for_disease(
    disease_name,
    medicine_df,
    tfidf_vectorizer,
    tfidf_matrix,
    review_preference='Excellent',
    top_n=5
):
    relevant_meds = medicine_df[medicine_df['Uses'].str.contains(disease_name, case=False, na=False)].copy()
    review_col = {
        'Excellent': 'Excellent Review %',
        'Average': 'Average Review %',
        'Poor': 'Poor Review %'
    }[review_preference]
    if not relevant_meds.empty:
        relevant_tfidf = tfidf_vectorizer.transform(relevant_meds['combined_features'])
        disease_vec = tfidf_vectorizer.transform([disease_name])
        similarity_scores = cosine_similarity(disease_vec, relevant_tfidf).flatten()
        relevant_meds['similarity_score'] = similarity_scores
        recommendations = relevant_meds.sort_values(
            by=['similarity_score', review_col],
            ascending=[False, False]
        ).head(top_n)
        return recommendations[['Medicine Name', 'Uses', 'Composition', 'Side_effects', 'Manufacturer', review_col]]
    else:
        disease_vec = tfidf_vectorizer.transform([disease_name])
        similarity_scores = cosine_similarity(disease_vec, tfidf_matrix).flatten()
        medicine_df['similarity_score'] = similarity_scores
        recommendations = medicine_df.sort_values(
            by=['similarity_score', review_col],
            ascending=[False, False]
        ).head(top_n)
        return recommendations[['Medicine Name', 'Uses', 'Composition', 'Side_effects', 'Manufacturer', review_col]]

# Step => 5 Running the full recommendation pipeline
def main_recommendation_pipeline(medicine_filepath, predicted_disease, review_pref='Excellent', top_n=5):
    medicine_df = load_and_prepare_medicine_data(medicine_filepath)
    tfidf_vectorizer, tfidf_matrix = create_tfidf_matrix(medicine_df)
    recommendations_df = recommend_medicines_for_disease(
        disease_name=predicted_disease,
        medicine_df=medicine_df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        review_preference=review_pref,
        top_n=top_n
    )
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf_vectorizer, './models/tfidf_vectorizer_model.pkl')
    os.makedirs('data', exist_ok=True)
    recommendations_df.to_csv('./data/user_recommendations.csv', index=False)
    return recommendations_df

# Step => 6 Example usage
if __name__ == "__main__":
    # Define input parameters
    medicine_csv_path = './data/feature_engineered_medicine_details.csv'
    predicted_disease = 'Diabetes'
    review_preference = 'Excellent'  # Choose: 'Excellent', 'Average', 'Poor'
    final_recommendations = main_recommendation_pipeline(medicine_csv_path, predicted_disease, review_preference)
    print(final_recommendations)
