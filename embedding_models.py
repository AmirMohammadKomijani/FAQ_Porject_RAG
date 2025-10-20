from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

print("🔧 Setting up embedding models...\n")

# Model 1: HooshvareLab/bert-fa-base-uncased
print("1️⃣ Loading HooshvareLab/bert-fa-base-uncased...")
embeddings_hooshvare = HuggingFaceEmbeddings(
    model_name="HooshvareLab/bert-fa-base-uncased",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ HooshvareLab model loaded")

# Model 2: HooshvareLab/sentence-bert-fa-base-stsb
print("\n2️⃣ Loading HooshvareLab/sentence-bert-fa-base-stsb...")
embeddings_sentence_bert = HuggingFaceEmbeddings(
    model_name="HooshvareLab/sentence-bert-fa-base-stsb",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Sentence-BERT model loaded")

# Model 3: HooshvareLab/bert-fa-zeroshot-clf-base
print("\n3️⃣ Loading HooshvareLab/bert-fa-zeroshot-clf-base...")
embeddings_zeroshot = HuggingFaceEmbeddings(
    model_name="HooshvareLab/bert-fa-zeroshot-clf-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Zero-shot model loaded")

# Store all models in a dictionary
embedding_models = {
    'hooshvare_bert': embeddings_hooshvare,
    'sentence_bert': embeddings_sentence_bert,
    'zeroshot_bert': embeddings_zeroshot
}

print("\n📊 All embedding models loaded:")
for name in embedding_models.keys():
    print(f"  - {name}")

# Test embedding dimension
print("\n🧪 Testing embedding dimensions...")
test_text = "سلام"
test_embedding = embeddings_hooshvare.embed_query(test_text)
print(f"Embedding dimension: {len(test_embedding)}")