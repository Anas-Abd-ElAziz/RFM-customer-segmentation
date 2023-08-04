import streamlit as st
import pandas as pd
import joblib

# Load the dumped model and scaler
kmeans = joblib.load('rfm_model.pkl')
df = pd.read_csv('clusters.csv')

def recommend_merchants_with_category(user_id):
    user_data = df[df['User_Id'] == user_id]
    cluster_label = user_data['Cluster'].iloc[0]
    recommended_category = user_data[user_data['Cluster'] == cluster_label]['Category In English'].mode().iloc[0]
    top_merchants = df[(df['Cluster'] == cluster_label) & (df['Category In English'] == recommended_category)]['Mer_Id'].value_counts().nlargest(5).index.tolist()
    return top_merchants

# Function to recommend a category based on user ID
def recommend_category(user_id):
    user_data = df[df['User_Id'] == user_id]
    cluster_label = user_data['Cluster'].iloc[0]
    recommended_category = user_data[user_data['Cluster'] == cluster_label]['Category In English'].mode().iloc[0]
    return recommended_category

# Function to preprocess user input
def preprocess_user_input(user_input):
    try:
        user_id = int(user_input)
        return user_id
    except ValueError:
        return None

# Streamlit app
def main():
    st.title('RFM Recommender App')
    user_input = st.text_input('Enter User ID:')
    user_id = preprocess_user_input(user_input)

    if user_id is not None:
        st.write(f"Recommendations for User ID: {user_id}")
		
        st.subheader("Recommended Category:")
        top_categories = recommend_category(user_id)
        st.write(top_categories)
		
        st.subheader("Recommended Merchants based on that category:")
        top_merchants = recommend_merchants_with_category(user_id)
        st.write(top_merchants)
    else:
        st.write("Please enter a valid User ID.")

if __name__ == '__main__':
    main()
