from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pyngrok import ngrok, conf
import nest_asyncio
import uvicorn
import asyncio

# Load and preprocess dataset
file_path = '/content/hotel_bookings.csv'
df = pd.read_csv(file_path)

# Data Cleaning (Avoid FutureWarnings)
df = df.assign(
    children=df['children'].fillna(0),
    country=df['country'].fillna('Unknown'),
    agent=df['agent'].fillna(0).astype(int),
    company=df['company'].fillna(0).astype(int)
)

# Revenue Calculation (Fixed)
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['adr_total'] = df['adr'] * df['total_nights']

# Vectorization for RAG (Include more fields for better context)
model = SentenceTransformer('all-MiniLM-L6-v2')
df['booking_info'] = df[['hotel', 'country', 'arrival_date_year', 'arrival_date_month',
                         'lead_time', 'adr', 'reservation_status', 'customer_type',
                         'market_segment', 'deposit_type']].astype(str).agg(' '.join, axis=1)
embeddings = model.encode(df['booking_info'].tolist())

# Create FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Load a question-answering pipeline for semantic understanding
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Enhanced Analytics Functions
def get_analytics():
    # Revenue by Year
    revenue_by_year = df.groupby(['arrival_date_year']).agg({'adr_total': 'sum'}).reset_index()

    # Cancellation Rate
    cancellation_rate = df['is_canceled'].mean() * 100

    # Geographical Distribution
    geo_distribution = df['country'].value_counts().to_dict()

    # Lead Time Distribution
    lead_time_distribution = df['lead_time'].describe().to_dict()

    # Booking Trends by Month
    booking_trends = df.groupby(['arrival_date_year', 'arrival_date_month']).size().reset_index(name='bookings')

    # Popular Room Types
    popular_room_types = df['assigned_room_type'].value_counts().to_dict()

    # Customer Types
    customer_types = df['customer_type'].value_counts().to_dict()

    # Hotel with Highest Cancellation Rate
    cancellation_by_hotel = df.groupby('hotel')['is_canceled'].mean().reset_index()
    highest_cancel_hotel = cancellation_by_hotel.loc[cancellation_by_hotel['is_canceled'].idxmax()]

    # Highest Booking Price
    highest_price = df['adr_total'].max()

    return {
        'revenue_by_year': revenue_by_year.to_dict(orient='records'),
        'cancellation_rate': cancellation_rate,
        'geo_distribution': geo_distribution,
        'lead_time_distribution': lead_time_distribution,
        'booking_trends': booking_trends.to_dict(orient='records'),
        'popular_room_types': popular_room_types,
        'customer_types': customer_types,
        'highest_cancel_hotel': highest_cancel_hotel.to_dict(),
        'highest_price': highest_price
    }

# Enhanced Question-Answering Function with Semantic Understanding
def ask_question(query):
    # Precomputed analytical insights
    analytics = get_analytics()

    # Check if the query is for an analytical insight
    if 'highest_cancel_hotel' in query.lower():
        return f"The hotel with the highest cancellation rate is '{analytics['highest_cancel_hotel']['hotel']}' with a rate of {analytics['highest_cancel_hotel']['is_canceled'] * 100:.2f}%."
    elif 'customer_types' in query.lower():
        return f"Customer types distribution: {analytics['customer_types']}"
    elif 'lead_time_distribution' in query.lower():
        return f"Lead time distribution: Min={analytics['lead_time_distribution']['min']}, Max={analytics['lead_time_distribution']['max']}, Mean={analytics['lead_time_distribution']['mean']:.2f}"
    elif 'highest_price' in query.lower():
        return f"The highest booking price is {analytics['highest_price']:.2f}."
    elif 'average price' in query.lower():
        avg_price = df['adr'].mean()
        return f"The average price of a hotel booking is {avg_price:.2f}."
    elif 'total revenue' in query.lower() and 'july 2017' in query.lower():
        revenue_july_2017 = df[(df['arrival_date_year'] == 2017) & (df['arrival_date_month'] == 'July')]['adr_total'].sum()
        return f"The total revenue for July 2017 is {revenue_july_2017:.2f}."
    elif 'locations with highest cancellations' in query.lower():
        cancellations_by_country = df[df['is_canceled'] == 1]['country'].value_counts().idxmax()
        return f"The location with the highest booking cancellations is {cancellations_by_country}."

    # Fallback to vector search and LLM for general queries
    query_vector = model.encode([query])
    _, indices = index.search(query_vector, k=5)
    context = " ".join([df.iloc[idx]['booking_info'] for idx in indices[0]])
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# Define FastAPI app
app = FastAPI()

# Root endpoint with HTML form for query input
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>Hotel Booking Analytics and RAG System</title>
        </head>
        <body>
            <h1>Welcome to the Hotel Booking Analytics and RAG System!</h1>
            <form action="/" method="post">
                <label for="query">Enter your question:</label><br>
                <input type="text" id="query" name="query" placeholder="e.g., What is the cancellation rate?"><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
def handle_query(query: str = Form(...)):
    answer = ask_question(query)
    return f"""
    <html>
        <head>
            <title>Hotel Booking Analytics and RAG System</title>
        </head>
        <body>
            <h1>Welcome to the Hotel Booking Analytics and RAG System!</h1>
            <form action="/" method="post">
                <label for="query">Enter your question:</label><br>
                <input type="text" id="query" name="query" placeholder="e.g., What is the cancellation rate?"><br><br>
                <input type="submit" value="Submit">
            </form>
            <h2>Answer:</h2>
            <p>{answer}</p>
        </body>
    </html>
    """

# Set ngrok authentication token (replace with your token)
NGROK_AUTH_TOKEN = ""  # Replace with your ngrok token
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# Run the API using ngrok
try:
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
except Exception as e:
    print("Failed to start ngrok tunnel:", e)
    raise

# Patch the event loop for Colab
nest_asyncio.apply()

# Run the FastAPI server with graceful shutdown
async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

try:
    asyncio.run(run_server())
except KeyboardInterrupt:
    print("Server stopped gracefully.")