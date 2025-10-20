import time
from data_preparation import test_data, corpus
from vector_stores import retrievers

def evaluate_retriever(retriever_name, retriever, test_data, corpus, k=3):
    """
    Evaluate retriever performance using precision, recall, and F1-score
    """
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    print(f"\n🔄 Evaluating retriever: {retriever_name}")
    print("=" * 80)
    start_time = time.time()
    
    for idx, row in test_data.iterrows():
        query = row['query_text']
        true_corpus_id = int(row['corpus-id'])
        
        # Get the true answer text
        true_answer_row = corpus[corpus['_id'] == true_corpus_id]
        true_answer_text = true_answer_row['text'].iloc[0] if not true_answer_row.empty else "Not found"
        
        print(f"\n🔍 Query {idx + 1}:")
        print(f"Question: {query}")
        print(f"Expected Answer ID: {true_corpus_id}")
        print(f"Expected Answer: {true_answer_text[:200]}...")
        
        # Retrieve documents
        retrieval_start = time.time()
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieval_end = time.time()
        
        print(f"⏱️ Retrieval Time: {retrieval_end - retrieval_start:.4f}s")
        print(f"\n📋 Retrieved Options:")
        
        # Extract corpus IDs from metadata
        retrieved_ids = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = int(doc.metadata.get('corpus_id', -1))
            retrieved_ids.append(doc_id)
            is_correct = "✅ CORRECT" if doc_id == true_corpus_id else "❌"
            print(f"Option {i} (ID: {doc_id}) {is_correct}:")
            print(f"Text: {doc.page_content[:150]}...")
        
        # Calculate metrics
        true_positive = 1 if true_corpus_id in retrieved_ids else 0
        precision = true_positive / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positive / 1  # Only 1 relevant per query
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        
        print(f"\n📊 Metrics: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print("-" * 80)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)
    
    print(f"\n📊 Final Retriever Results for {retriever_name}:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")
    print(f"⏱️  Total Retrieval Time: {total_time:.2f}s")
    print(f"⏱️  Average Time per Query: {total_time/len(test_data):.4f}s")
    
    return {
        'retriever_name': retriever_name,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'total_time': total_time,
        'avg_time_per_query': total_time/len(test_data),
        'individual_scores': {
            'precisions': all_precisions,
            'recalls': all_recalls,
            'f1s': all_f1s
        }
    }

# Evaluate all retrievers
if __name__ == "__main__":
    retriever_results = {}
    
    for retriever_name, retriever in retrievers.items():
        results = evaluate_retriever(retriever_name, retriever, test_data, corpus, k=3)
        retriever_results[retriever_name] = results
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 RETRIEVER EVALUATION SUMMARY")
    print("=" * 80)
    for name, results in retriever_results.items():
        print(f"\n{name}:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  Time: {results['total_time']:.2f}s")