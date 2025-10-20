import pandas as pd
from langchain.schema import Document

# Load datasets
print("📂 Loading datasets...")
corpus = pd.read_csv('corpus.csv')
test_data = pd.read_csv('test_enriched.csv')

print(f"✅ Corpus loaded: {len(corpus)} documents")
print(f"✅ Test data loaded: {len(test_data)} queries")

# Display sample
print("\n📋 Corpus sample:")
print(corpus.head(1))
print("\n📋 Test data sample:")
print(test_data.head(1))

# Convert corpus to Document objects
print("\n🔄 Converting corpus to Document objects...")
documents = []

for idx, row in corpus.iterrows():
    doc = Document(
        page_content=row['text'],
        metadata={
            'corpus_id': int(row['_id']),
            'title': row['title'],
            'source': 'persian_faq'
        }
    )
    documents.append(doc)

print(f"✅ Created {len(documents)} Document objects")

# Display sample document
print("\n📄 Sample Document:")
print(documents[0])

# Save for later use
print("\n💾 Data preparation complete!")
print(f"Total documents: {len(documents)}")
print(f"Total test queries: {len(test_data)}")