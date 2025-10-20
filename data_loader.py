from datasets import load_dataset

ds = load_dataset("MCINext/synthetic-persian-chatbot-rag-faq-retrieval", "default")
corpus = pd.read_json("hf://datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval/corpus.jsonl", lines=True)
queries = load_dataset("MCINext/synthetic-persian-chatbot-rag-faq-retrieval", "queries")
ds, corpus, queries

from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

# Load your datasets (adjust the path/name as needed)
# Assuming you have them loaded as:
# train_test_data - contains train/test splits with query-id, corpus-id, score
# corpus_data - contains corpus with _id, title, text
# queries_data - contains queries with _id, text

def prepare_rag_dataset(train_test_data, corpus_data, queries_data):
    """
    Prepare and enrich the dataset for RAG system
    """

    # Convert to pandas for easier manipulation
    corpus_df = corpus_data['corpus'].to_pandas()
    queries_df = queries_data['queries'].to_pandas()
    train_df = train_test_data['train'].to_pandas()
    test_df = train_test_data['test'].to_pandas()

    print("\n" + "="*80)
    print("ORIGINAL DATASET SHAPES")
    print("="*80)
    print(f"Corpus: {corpus_df.shape}")
    print(f"Queries: {queries_df.shape}")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")

    # Convert score to integer
    print("\n" + "="*80)
    print("CONVERTING SCORE TO INTEGER")
    print("="*80)
    train_df['score'] = train_df['score'].astype(int)
    test_df['score'] = test_df['score'].astype(int)
    print(f"Train score dtype: {train_df['score'].dtype}")
    print(f"Test score dtype: {test_df['score'].dtype}")
    print(f"Train score unique values: {train_df['score'].unique()}")
    print(f"Test score unique values: {test_df['score'].unique()}")

    # Step 1: Filter for score = 1 only
    print("\n" + "="*80)
    print("STEP 1: FILTERING FOR SCORE = 1")
    print("="*80)
    train_filtered = train_df[train_df['score'] == 1].copy()
    test_filtered = test_df[test_df['score'] == 1].copy()
    train_filtered['corpus-id'] = train_filtered['corpus-id'].astype(int)
    test_filtered['corpus-id'] = test_filtered['corpus-id'].astype(int)

    print(f"Train: {len(train_filtered)} rows (was {len(train_df)}, removed {len(train_df) - len(train_filtered)})")
    print(f"Test: {len(test_filtered)} rows (was {len(test_df)}, removed {len(test_df) - len(test_filtered)})")

    # Step 2: Merge with corpus data
    print("\n" + "="*80)
    print("STEP 2: MERGING WITH CORPUS DATA")
    print("="*80)
    corpus_df_renamed = corpus_df.rename(columns={
        '_id': 'corpus-id',
        'title': 'corpus_title',
        'text': 'corpus_text'
    })

    train_enriched = train_filtered.merge(
        corpus_df_renamed,
        on='corpus-id',
        how='left'
    )

    test_enriched = test_filtered.merge(
        corpus_df_renamed,
        on='corpus-id',
        how='left'
    )

    print(f"Train enriched shape: {train_enriched.shape}")
    print(f"Test enriched shape: {test_enriched.shape}")

    # Step 3: Merge with queries data
    print("\n" + "="*80)
    print("STEP 3: MERGING WITH QUERIES DATA")
    print("="*80)
    queries_df_renamed = queries_df.rename(columns={
        '_id': 'query-id',
        'text': 'query_text'
    })

    train_enriched = train_enriched.merge(
        queries_df_renamed,
        on='query-id',
        how='left'
    )

    test_enriched = test_enriched.merge(
        queries_df_renamed,
        on='query-id',
        how='left'
    )

    print(f"Train final shape: {train_enriched.shape}")
    print(f"Test final shape: {test_enriched.shape}")
    print(f"\nColumns: {train_enriched.columns.tolist()}")

    # Step 4: Check for any missing data
    print("\n" + "="*80)
    print("STEP 4: CHECKING FOR MISSING VALUES")
    print("="*80)
    print(f"Missing values in train:")
    print(train_enriched.isnull().sum())
    print(f"\nMissing values in test:")
    print(test_enriched.isnull().sum())

    # Step 5: Display sample data
    print("\n" + "="*80)
    print("SAMPLE FROM ENRICHED TRAINING DATA")
    print("="*80)
    if len(train_enriched) > 0:
        sample = train_enriched.iloc[0]
        print(f"\nQuery ID: {sample['query-id']}")
        print(f"Query Text: {sample['query_text']}")
        print(f"\nCorpus ID: {sample['corpus-id']}")
        print(f"Corpus Title: {sample['corpus_title']}")
        print(f"Corpus Text: {sample['corpus_text'][:200]}...")
        print(f"\nScore: {sample['score']}")

    return train_enriched, test_enriched, corpus_df

# Usage example:
# train_enriched, test_enriched, corpus_df = prepare_rag_dataset(
#     train_test_data,
#     corpus_data,
#     queries_data
# )

# Save the enriched datasets
def save_enriched_data(train_enriched, test_enriched, corpus_df, output_dir='./enriched_data'):
    """
    Save the enriched datasets for later use
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    train_enriched.to_csv(f'{output_dir}/train_enriched.csv', index=False)
    test_enriched.to_csv(f'{output_dir}/test_enriched.csv', index=False)
    corpus_df.to_csv(f'{output_dir}/corpus.csv', index=False)

    print(f"\nData saved to {output_dir}/")
    print(f"- train_enriched.csv: {len(train_enriched)} rows")
    print(f"- test_enriched.csv: {len(test_enriched)} rows")
    print(f"- corpus.csv: {len(corpus_df)} rows")

# Additional utility function to get statistics
def get_dataset_statistics(train_enriched, test_enriched, corpus_df):
    """
    Get useful statistics about the dataset
    """
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    print(f"\nCorpus Statistics:")
    print(f"- Total documents: {len(corpus_df)}")
    print(f"- Average text length: {corpus_df['text'].str.len().mean():.0f} characters")
    print(f"- Min text length: {corpus_df['text'].str.len().min()}")
    print(f"- Max text length: {corpus_df['text'].str.len().max()}")

    print(f"\nTraining Set:")
    print(f"- Total query-corpus pairs: {len(train_enriched)}")
    print(f"- Unique queries: {train_enriched['query-id'].nunique()}")
    print(f"- Unique corpus documents: {train_enriched['corpus-id'].nunique()}")

    print(f"\nTest Set:")
    print(f"- Total query-corpus pairs: {len(test_enriched)}")
    print(f"- Unique queries: {test_enriched['query-id'].nunique()}")
    print(f"- Unique corpus documents: {test_enriched['corpus-id'].nunique()}")



# Prepare the dataset
train_enriched, test_enriched, corpus_df = prepare_rag_dataset(ds, corpus, queries)

# Get statistics
get_dataset_statistics(train_enriched, test_enriched, corpus_df)

# Save the enriched data
save_enriched_data(train_enriched, test_enriched, corpus_df)

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)
print("\nYou now have:")
print("1. train_enriched - Training data with full text")
print("2. test_enriched - Test data with full text")
print("3. corpus_df - Complete corpus for RAG vector store")
print("\nNext steps:")
print("- Build vector store from corpus_df")
print("- Implement RAG system")
print("- Evaluate using test_enriched")



corpus = pd.read_csv('/kaggle/working/enriched_data/corpus.csv')
test_data = pd.read_csv('/kaggle/working/enriched_data/test_enriched.csv')
test_data = test_data[:50]