from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector_stores import retrievers

print("🤖 Setting up generation chain...\n")

# Define template
template = """شما یک دستیار هوشمند هستید که به سوالات کاربران به زبان فارسی پاسخ می‌دهید.
از اطلاعات زیر برای پاسخ به سوال استفاده کنید. اگر پاسخ را نمی‌دانید، صادقانه بگویید که نمی‌دانید و فقط یک کلمه بگو نمیدانم..

اطلاعات مرتبط:
{context}

سوال: {input}

پاسخ به زبان فارسی:"""

print("📝 Creating prompt template...")
prompt = ChatPromptTemplate.from_template(template)

# Initialize LLM
print("🔌 Initializing ChatOpenAI (gpt-4o)...")
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
)
print("✅ LLM initialized")

# Create chains
print("\n⛓️ Creating chains...\n")

# Question-Answer chain
print("1️⃣ Creating question-answer chain...")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
print("✅ QA chain created")

# RAG chains for each retriever
rag_chains = {}

for retriever_name, retriever in retrievers.items():
    print(f"2️⃣ Creating RAG chain for {retriever_name}...")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    rag_chains[retriever_name] = rag_chain
    print(f"✅ RAG chain created for {retriever_name}")

print(f"\n📊 All RAG chains ready!")
print(f"Available RAG chains: {list(rag_chains.keys())}")