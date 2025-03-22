# Hotel Booking Analytics & RAG System  

A FastAPI-based web application for analyzing hotel booking data and answering natural language queries using Retrieval-Augmented Generation (RAG).  

---

## Features  
- **Data Cleaning & Preprocessing**: Handles missing values and calculates metrics like total revenue.  
- **Vectorization & FAISS Indexing**: Uses Sentence Transformers for embeddings and FAISS for efficient similarity search.  
- **Precomputed Analytics**: Provides insights like cancellation rates, revenue trends, and customer distributions.  
- **Natural Language Querying**: Answers user questions using a combination of analytics and a question-answering pipeline.  
- **Web Interface**: Simple HTML form for query input and result display.  

---

## Setup  

1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/MULAKALASIVARAMAKRISHNA/LLM_Based_Analytics/  
   ```  

2. **Install Dependencies**:  
   ```bash  
   pip install fastapi uvicorn pandas numpy faiss-cpu sentence-transformers transformers pyngrok nest-asyncio  
   ```  

3. **Set Up Ngrok**:  
   Replace `NGROK_AUTH_TOKEN` in the script with your [ngrok authentication token](https://dashboard.ngrok.com/get-started/your-authtoken).  

4. **Run the Application**:  
   ```bash  
   python app.py  
   ```  

5. **Access the Web Interface**:  
   Open the ngrok URL provided in the terminal to interact with the system.  

---

## Usage  

Enter natural language queries like:  
- "What is the cancellation rate?"  
- "Which hotel has the highest cancellations?"  
- "What is the total revenue for July 2017?"  
- "What is the average booking price?"  

The system will display the answer below the query form.  

---

## Example Queries  
- **Cancellation Rate**: "What is the cancellation rate?"  
- **Highest Cancellation Hotel**: "Which hotel has the highest cancellation rate?"  
- **Customer Distribution**: "What are the customer types distribution?"  
- **Revenue Trends**: "What is the total revenue for July 2017?"  

---

## Technologies Used  
- **FastAPI**: Web framework for building the API.  
- **Sentence Transformers**: For generating embeddings.  
- **FAISS**: For efficient similarity search.  
- **Transformers**: For question-answering pipeline.  
- **Pyngrok**: For exposing the app via a public URL.  


---

## Acknowledgments  
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.  
- [Sentence Transformers](https://www.sbert.net/) for embeddings.  
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search.  
- [Transformers](https://huggingface.co/transformers/) for question-answering.  

---

Explore hotel booking insights effortlessly! 
