# Step 1 => Import required libraries
import pandas as pd
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2 => Load user and medicine data
users_df = pd.read_csv('./data/cleaned_user_data.csv')  
med_df = pd.read_csv('./data/cleaned_medicine_details.csv')  

# Step 3 => Create synthetic ratings.csv
user_ids = users_df['UserID'].unique()
medicine_names = med_df['Medicine Name'].dropna().unique()
synthetic_data = []
for user in user_ids:
    rated_meds = random.sample(list(medicine_names), k=random.randint(5, 10))  
    for med in rated_meds:
        rating = random.randint(1, 5)  # Rating from 1 to 5
        synthetic_data.append([user, med, rating])
ratings_df = pd.DataFrame(synthetic_data, columns=['UserID', 'Medicine Name', 'Rating'])
os.makedirs("data", exist_ok=True)
ratings_df.to_csv('./data/ratings.csv', index=False)

# Step 4 => Build Collaborative Filtering Model
user_item_matrix = ratings_df.pivot_table(index='UserID', columns='Medicine Name', values='Rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
def get_user_recommendations(target_user, top_n=5):
    similar_users = user_similarity_df[target_user].sort_values(ascending=False)[1:]  # exclude self
    weighted_ratings = pd.Series(dtype=float)
    for user, similarity in similar_users.items():
        user_ratings = user_item_matrix.loc[user]
        weighted_ratings = weighted_ratings.add(user_ratings * similarity, fill_value=0)
    rated_meds = user_item_matrix.loc[target_user][user_item_matrix.loc[target_user] > 0].index
    weighted_ratings = weighted_ratings.drop(labels=rated_meds, errors='ignore')
    return weighted_ratings.sort_values(ascending=False).head(top_n)

# Step 5 => Build Content-Based Filtering Model
med_df['combined'] = (
    med_df['Uses'].fillna('') + ' ' +
    med_df['Composition'].fillna('') + ' ' +
    med_df['Side_effects'].fillna('')
)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(med_df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(med_df.index, index=med_df['Medicine Name'])
def get_content_recommendations(medicine_name, top_n=5):
    if medicine_name not in indices:
        return pd.Series([])  
    idx = indices[medicine_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    medicine_indices = [i[0] for i in sim_scores]
    return med_df['Medicine Name'].iloc[medicine_indices]

# Step 6 => Merge Both Models into Hybrid
def hybrid_recommendation(user_id, top_n=5):
    collab_recs = get_user_recommendations(user_id, top_n * 2)  # Get more to allow duplicates
    combined_recs = []
    for med in collab_recs.index:
        content_recs = get_content_recommendations(med, top_n=2)
        combined_recs.extend(content_recs)
    hybrid_df = pd.DataFrame(combined_recs, columns=["Medicine Name"])
    hybrid_df = hybrid_df['Medicine Name'].value_counts().reset_index()
    hybrid_df.columns = ['Medicine Name', 'Score']
    return hybrid_df.head(top_n)

# Step 7 => Test the system
test_user_id = user_ids[0]  
final_recommendations = hybrid_recommendation(test_user_id, top_n=5)
print(f"Final Hybrid Recommendations for {test_user_id}:\n")
print(final_recommendations)
