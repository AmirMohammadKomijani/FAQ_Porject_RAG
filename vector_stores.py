from langchain_community.vectorstores import FAISS
from data_preparation import documents
from embedding_models import embedding_models
import time

print("🏗️ Building vector stores...\n")

vector_stores = {}

# Vector Store 1: With HooshvareLab BERT
print("1️⃣ Building FAISS with HooshvareLab BERT...")
start_time = time.time()
faiss_hooshvare = FAISS.from_documents(
    documents,
    embedding=embedding_models['hooshvare_bert']
)
time_hooshvare = time.time() - start_time
vector_stores['hooshvare_bert'] = faiss_hooshvare
print(f"✅ FAISS (HooshvareLab) built in {time_hooshvare:.2f}s")

# Vector Store 2: With Sentence-BERT
print("\n2️⃣ Building FAISS with Sentence-BERT...")
start_time = time.time()
faiss_sentence_bert = FAISS.from_documents(
    documents,
    embedding=embedding_models['sentence_bert']
)
time_sentence_bert = time.time() - start_time
vector_stores['sentence_bert'] = faiss_sentence_bert
print(f"✅ FAISS (Sentence-BERT) built in {time_sentence_bert:.2f}s")

# Vector Store 3: With Zero-shot BERT
print("\n3️⃣ Building FAISS with Zero-shot BERT...")
start_time = time.time()
faiss_zeroshot = FAISS.from_documents(
    documents,
    embedding=embedding_models['zeroshot_bert']
)
time_zeroshot = time.time() - start_time
vector_stores['zeroshot_bert'] = faiss_zeroshot
print(f"✅ FAISS (Zero-shot) built in {time_zeroshot:.2f}s")

# Create retrievers
print("\n🔍 Creating retrievers...\n")
retrievers = {}

for name, vectorstore in vector_stores.items():
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    retrievers[name] = retriever
    print(f"✅ Retriever created for {name}")

print(f"\n📊 All vector stores and retrievers ready!")
print(f"Total documents indexed: {len(documents)}")