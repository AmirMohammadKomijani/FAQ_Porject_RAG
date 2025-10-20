#!/usr/bin/env python3

print("🚀 RAG System Benchmark - Persian QA")
print("=" * 80)

# Step 1: Prepare data
print("\n📂 Step 1: Data Preparation")
print("-" * 80)
from data_preparation import documents, test_data, corpus

# Step 2: Load embedding models
print("\n🔧 Step 2: Loading Embedding Models")
print("-" * 80)
from embedding_models import embedding_models

# Step 3: Build vector stores
print("\n🏗️ Step 3: Building Vector Stores")
print("-" * 80)
from vector_stores import vector_stores, retrievers

# Step 4: Setup generation
print("\n🤖 Step 4: Setting up Generation Chains")
print("-" * 80)
from generation import rag_chains

# Step 5: Evaluate retrievers
print("\n📊 Step 5: Evaluating Retrievers")
print("-" * 80)
from evaluate_retriever import evaluate_retriever
retriever_results = {}
for retriever_name, retriever in retrievers.items():
    results = evaluate_retriever(retriever_name, retriever, test_data, corpus, k=3)
    retriever_results[retriever_name] = results

# Step 6: Evaluate generators
print("\n📊 Step 6: Evaluating Generators")
print("-" * 80)
from evaluate_generator import evaluate_generator
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

# Final Summary
print("\n" + "=" * 80)
print("🎯 FINAL BENCHMARK SUMMARY")
print("=" * 80)

print("\n📊 RETRIEVER RESULTS:")
for name, results in retriever_results.items():
    print(f"\n{name}:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")

print("\n📊 GENERATOR RESULTS:")
for name, results in generator_results.items():
    print(f"\n{name}:")
    print(f"  BLEU: {results['bleu']:.4f}")
    print(f"  ROUGE-1: {results['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']:.4f}")
    print(f"  ROUGE-L: {results['rougeL']:.4f}")

print("\n✅ Benchmark complete!")