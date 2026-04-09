
# evaluate_rag.py  (à la racine, PAS dans tests/)

import os
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from mistralai import Mistral
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from RAG.retrieval   import telecharger_evenements
from RAG.embedding   import generer_embeddings
from RAG.vectorstore import construire_vectorstore_langchain, creer_retriever
from RAG.rag         import construire_chaine_rag


# initialisation
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)
llm    = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",   
    temperature=0
)


# Construction du pipeline RAG 
print(" Construction du pipeline...")
evenements  = telecharger_evenements()
resultats   = generer_embeddings(evenements, client)
vectorstore = construire_vectorstore_langchain(resultats, client)
retriever   = creer_retriever(vectorstore)
rag_chain   = construire_chaine_rag(retriever, llm)
print(" Pipeline prêt\n")


# Dataset de test 
questions = [
    "Quels événements à Toulouse ?",
    "Que faire ce week-end en Occitanie ?",
]

ground_truths = [
    #  Réponses de référence
    "Il y a plusieurs événements à Toulouse comme des concerts, expositions et marchés.",
    "Ce week-end en Occitanie vous pouvez assister à des festivals, marchés et spectacles.",
]

answers  = []
contexts = []

for q in questions:
    response, docs = rag_chain(q)
    answers.append(response)
    contexts.append([doc.page_content for doc in docs])
    print(f"Q: {q}")
    print(f"R: {response[:100]}...\n")

dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})


# Evaluation Ragas
print(" Evaluation Ragas en cours")

# On configure Ragas pour utiliser Mistral
ragas_llm        = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n Résultats Ragas :")
print(result)

# Seuils minimaux acceptables
assert result["faithfulness"] [0]    > 0.5, " Fidélité trop faible"
assert result["answer_relevancy"] [0] > 0.5, " Pertinence trop faible"
print("\n Evaluation terminée avec succès !")