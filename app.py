
import pandas as pd
import nltk
import gensim
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK's SnowballStemmer for stemming
stemmer = nltk.stem.SnowballStemmer('english')
# Load the CSV data into a DataFrame



from flask import Flask, request, jsonify, json

app = Flask(__name__)

def run_ml_model(input_param):
    df = pd.read_csv('dataset/formulation.csv')
    user_input = input_param
    df1=df
    user_symptoms = user_input.split(',')

    user_symptoms_lower = [symptom.lower() for symptom in user_symptoms]
    user_symptoms_stemmed = [stemmer.stem(symptom) for symptom in user_symptoms_lower]
    user_symptoms = ','.join(user_symptoms_stemmed)

    def stem_words(word_list):
        return [stemmer.stem(word) for word in word_list]

    # Initialize a larger BERT model and tokenizer


    df1['combined_text'] = df1['Symptoms']

    def tokenize_symptoms(symptoms):
        # Split symptoms based on commas and treat them as individual tokens
        tokens = [token.strip() for token in symptoms.split(',') if token.strip()]
        return tokens

    # Apply the symptom tokenization function to the combined_text

    # Preprocess and tokenize the "Symptoms" column using the custom tokenizer
    df['Symptoms1'] = df1['Symptoms'].apply(tokenize_symptoms)

    # Tokenize the input symptoms
    input_symptoms_tokens1 = tokenize_symptoms(user_symptoms)
    df['Symptoms1'] = df1['Symptoms1'].apply(stem_words)

    # Create TF-IDF vectors for symptoms
    tfidf_vectorizer = TfidfVectorizer()
    symptoms_tfidf = tfidf_vectorizer.fit_transform(df['Symptoms1'].apply(lambda x: ' '.join(x)))

    # Create a TF-IDF vector for the input symptoms
    input_symptoms_tfidf = tfidf_vectorizer.transform([' '.join(input_symptoms_tokens1)])

    # Calculate cosine similarity between input symptoms and each formulation's symptoms
    cosine_similarities = cosine_similarity(symptoms_tfidf, input_symptoms_tfidf)

    # Add the similarity scores to the DataFrame
    df['Similarity_Score'] = cosine_similarities

    # Find the formulation with the highest similarity score
    recommended_row = df[df['Similarity_Score'] == df['Similarity_Score'].max()]
    recommended_formulation = recommended_row['Formulation'].values[0]
    similarity_score = recommended_row['Similarity_Score'].values[0]
    ##########################################################################################################
    # Tokenize the input symptoms
    input_symptoms_tokens = [stemmer.stem(symptom.lower()) for symptom in user_input.split(',') if symptom.strip()]

    # Tokenize and stem the symptoms in the DataFrame
    df['Symptoms'] = df['Symptoms'].apply(
        lambda x: [stemmer.stem(symptom.lower()) for symptom in x.split(',') if symptom.strip()])

    # Train Word2Vec embeddings on your symptom data
    model = gensim.models.Word2Vec(df['Symptoms'], vector_size=100, window=5, min_count=1,
                                   sg=0)  # Adjust parameters as needed

    # Function to get the vector representation of a list of words
    def get_word_vectors(word_list, model):
        vectors = [model.wv[word] for word in word_list if word in model.wv]
        if not vectors:
            # If no valid word vectors found, return a zero vector
            return [0.0] * model.vector_size
        return sum(vectors) / len(vectors)

    # Create vectors for user input symptoms
    user_input_vector = get_word_vectors(input_symptoms_tokens, model)

    # Create vectors for symptoms in the DataFrame
    df['SymptomVectors'] = df['Symptoms'].apply(lambda x: get_word_vectors(x, model))

    # Calculate cosine similarity between input symptoms and each formulation's symptoms
    df['Similarity Score'] = df['SymptomVectors'].apply(lambda x: cosine_similarity([user_input_vector], [x])[0][0])

    ###########################################################################################################
    # Sort formulations by Similarity Score (descending)
    df['Exact Match'] = df['Symptoms'].apply(lambda x: any(symptom in user_symptoms for symptom in x))
    df = df.sort_values(by=['Exact Match', 'Similarity Score', 'Similarity_Score'],
                        ascending=[False, False, False]).reset_index(drop=True)

    df=df.iloc[:10]

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input parameter from the request JSON

        input_param = request.form.get('input_param')

        # Call your ML function
        prediction_result = run_ml_model(input_param)



        # ac = prediction_result.to_dict(orient='records')




        # Return the prediction result as JSON
        bc=prediction_result.to_json()

        return bc

    except Exception as e:
        # Handle any errors and return an error response
        return jsonify({'error': str(e)}), 400

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
