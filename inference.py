import argparse
import json
import time
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Run RAG Inference for BIS Standards")
    parser.add_argument("--input", required=True, help="Path to input JSON file containing queries")
    parser.add_argument("--output", required=True, help="Path to save the output JSON")
    parser.add_argument("--data", default="data/standards.json", help="Path to standards database")
    args = parser.parse_args()

    # Load queries
    try:
        with open(args.input, 'r') as f:
            queries = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Initialize Pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(data_path=args.data, use_llm=False)
    
    results = []
    
    print(f"Processing {len(queries)} queries...")
    for idx, item in enumerate(queries):
        query_id = item.get("id", idx + 1)
        query_text = item.get("query", "")
        
        start_time = time.time()
        
        # Run inference
        pipeline_output = pipeline.run(query_text, top_k=5)
        retrieved_standards = [res["standard_id"] for res in pipeline_output["results"]]
        
        latency = time.time() - start_time
        
        # Retain original keys from input (like expected_standards, query)
        output_item = item.copy()
        output_item["retrieved_standards"] = retrieved_standards
        output_item["latency_seconds"] = round(latency, 4)
        
        results.append(output_item)
        print(f"Processed query {query_id} in {latency:.4f}s")

    # Save output
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
