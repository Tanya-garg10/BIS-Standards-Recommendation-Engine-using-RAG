import os
from openai import OpenAI
from src.vector_db import VectorDB

class RAGPipeline:
    def __init__(self, data_path, use_llm=True):
        self.vector_db = VectorDB()
        self.vector_db.load_data(data_path)
        self.use_llm = use_llm
        if self.use_llm:
            # Ensure OPENAI_API_KEY is set in environment
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))

    def run(self, query, top_k=5):
        # 1. Retrieve top 10 documents
        retrieved_docs = self.vector_db.search(query, top_k=10)
        
        if not retrieved_docs:
            return {"results": [], "reasoning": "No relevant standards found."}

        # If LLM is not enabled or no key provided, just return top k from vector search
        if not self.use_llm or self.client.api_key == "dummy-key":
            top_docs = retrieved_docs[:top_k]
            return {
                "results": [
                    {
                        "standard_id": doc["standard_id"], 
                        "title": doc["title"], 
                        "description": doc.get("description", ""), 
                        "confidence": doc["confidence"], 
                        "reason": "Matched via vector similarity."
                    }
                    for doc in top_docs
                ]
            }

        # 2. Rerank and extract exact info using LLM (Avoid Hallucination)
        context_str = "\n".join([f"- {doc['standard_id']}: {doc['title']} | {doc['description']}" for doc in retrieved_docs])
        
        prompt = f"""
Given the user query: "{query}"

Select the top {top_k} most relevant BIS standards strictly from the following retrieved context.
DO NOT hallucinate or invent any standards. If no standards are perfectly relevant, select the closest ones based on the context.

Retrieved Context:
{context_str}

Return the output as a valid JSON array of objects. Each object should have:
- "standard_id" (e.g., IS 269)
- "reason" (a short 1-line explanation of why it was chosen based on the context)

Only return the JSON array, no other text.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that strictly follows instructions and outputs only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            result_content = response.choices[0].message.content.strip()
            
            # Clean up markdown formatting if present
            if result_content.startswith("```json"):
                result_content = result_content[7:-3]
            elif result_content.startswith("```"):
                result_content = result_content[3:-3]
                
            import json
            llm_results = json.loads(result_content)
            
            # Add confidence scores from original retrieval
            final_results = []
            for item in llm_results:
                matched_doc = next((doc for doc in retrieved_docs if doc['standard_id'] == item.get('standard_id')), None)
                if matched_doc:
                    item['title'] = matched_doc['title']
                    item['description'] = matched_doc.get('description', '')
                    item['confidence'] = matched_doc['confidence']
                    final_results.append(item)
                    
            return {"results": final_results}
            
        except Exception as e:
            print(f"LLM Error: {e}")
            # Fallback to pure retrieval
            top_docs = retrieved_docs[:top_k]
            return {
                "results": [
                    {
                        "standard_id": doc["standard_id"], 
                        "title": doc["title"], 
                        "description": doc.get("description", ""), 
                        "confidence": doc["confidence"], 
                        "reason": "LLM failed, matched via vector similarity."
                    }
                    for doc in top_docs
                ]
            }
