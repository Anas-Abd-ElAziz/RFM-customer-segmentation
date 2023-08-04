import streamlit as st
import pandas as pd
import joblib

# Load the dumped model and scaler
kmeans = joblib.load('rfm_model.pkl')
df = pd.read_csv('clusters.csv')

def recommend_merchant(user_id):
	user_data = df[df['User_Id'] == user_id]
	cluster_label = user_data['Cluster'].iloc[0]
	recommended_merchant = user_data[user_data['Cluster'] == cluster_label]['Mer_Id'].mode().iloc[0]
	return recommended_merchant

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
        st.subheader("Recommended Merchant:")
        top_merchants = recommend_merchant(user_id)
        st.write(top_merchants)

        st.subheader("Recommended Category:")
        top_categories = recommend_category(user_id)
        st.write(top_categories)
    else:
        st.write("Please enter a valid User ID.")

if __name__ == '__main__':
    main()
