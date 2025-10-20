import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from data_preparation import test_data, corpus
from generation import rag_chains

# Download required NLTK data
nltk.download('punkt')

def evaluate_generator(rag_chain_name, rag_chain, test_data, corpus, k=3, 
                       delay_between_requests=1, break_after=5, break_duration=5):
    """
    Evaluate generator performance using BLEU and ROUGE scores with rate limiting
    """
    all_bleu_scores = []
    all_rouge1_scores = []
    all_rouge2_scores = []
    all_rougeL_scores = []
    
    print(f"\n🔄 Evaluating generator: {rag_chain_name}")
    print("=" * 80)
    start_time = time.time()
    
    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for idx, row in test_data.iterrows():
        query = row['query_text']
        true_corpus_id = int(row['corpus-id'])
        
        # Get expected answer
        true_answer_row = corpus[corpus['_id'] == true_corpus_id]
        expected_answer = true_answer_row['text'].iloc[0] if not true_answer_row.empty else ""
        
        print(f"\n📝 Sample {idx + 1}:")
        print(f"Question: {query}")
        print(f"Expected Answer: {expected_answer[:200]}...")
        
        # Generate answer using RAG chain
        gen_start = time.time()
        try:
            result = rag_chain.invoke({"input": query})
            gen_end = time.time()
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            print(f"⏸️  Waiting {break_duration * 2} seconds before retry...")
            time.sleep(break_duration * 2)
            continue
        
        generated_answer = result.get('answer', '')
        
        print(f"Generated Answer: {generated_answer[:200]}...")
        print(f"⏱️ Generation Time: {gen_end - gen_start:.4f}s")
        
        # Calculate BLEU score
        expected_tokens = expected_answer.split()
        generated_tokens = generated_answer.split()
        
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [expected_tokens],
            generated_tokens,
            smoothing_function=smoothing_function
        )
        all_bleu_scores.append(bleu_score)
        
        # Calculate ROUGE scores
        rouge_scores = rouge_scorer_obj.score(expected_answer, generated_answer)
        
        rouge1 = rouge_scores['rouge1'].fmeasure
        rouge2 = rouge_scores['rouge2'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure
        
        all_rouge1_scores.append(rouge1)
        all_rouge2_scores.append(rouge2)
        all_rougeL_scores.append(rougeL)
        
        print(f"\n📊 Scores:")
        print(f"BLEU: {bleu_score:.4f}")
        print(f"ROUGE-1: {rouge1:.4f}")
        print(f"ROUGE-2: {rouge2:.4f}")
        print(f"ROUGE-L: {rougeL:.4f}")
        print("-" * 80)
        
        # Add delay between requests
        if idx < len(test_data) - 1:
            print(f"⏸️  Waiting {delay_between_requests}s before next request...")
            time.sleep(delay_between_requests)
        
        # Take a break every N requests
        if (idx + 1) % break_after == 0 and idx < len(test_data) - 1:
            print(f"\n🛑 Break time! Processed {idx + 1} queries.")
            print(f"⏸️  Taking a {break_duration}s break...")
            time.sleep(break_duration)
            print(f"✅ Resuming...\n")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate averages
    if all_bleu_scores:
        avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
        avg_rouge1 = sum(all_rouge1_scores) / len(all_rouge1_scores)
        avg_rouge2 = sum(all_rouge2_scores) / len(all_rouge2_scores)
        avg_rougeL = sum(all_rougeL_scores) / len(all_rougeL_scores)
    else:
        avg_bleu = avg_rouge1 = avg_rouge2 = avg_rougeL = 0
    
    print(f"\n📊 Final Generator Results for {rag_chain_name}:")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")
    print(f"⏱️  Total Generation Time: {total_time:.2f}s")
    print(f"⏱️  Average Time per Query: {total_time/len(test_data):.4f}s")
    
    return {
        'rag_chain_name': rag_chain_name,
        'bleu': avg_bleu,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'total_time': total_time,
        'avg_time_per_query': total_time/len(test_data),
        'individual_scores': {
            'bleu_scores': all_bleu_scores,
            'rouge1_scores': all_rouge1_scores,
            'rouge2_scores': all_rouge2_scores,
            'rougeL_scores': all_rougeL_scores
        }
    }

# Evaluate all RAG chains
if __name__ == "__main__":
    generator_results = {}
    
    for rag_chain_name, rag_chain in rag_chains.items():
        results = evaluate_generator(
            rag_chain_name, 
            rag_chain, 
            test_data, 
            corpus, 
            k=3,
            delay_between_requests=1,
            break_after=5,
            break_duration=5
        )
        generator_results[rag_chain_name] = results
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 GENERATOR EVALUATION SUMMARY")
    print("=" * 80)
    for name, results in generator_results.items():
        print(f"\n{name}:")
        print(f"  BLEU: {results['bleu']:.4f}")
        print(f"  ROUGE-1: {results['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['rougeL']:.4f}")
        print(f"  Time: {results['total_time']:.2f}s")