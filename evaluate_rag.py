
import os
import math
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from mistralai import Mistral as MistralClient
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from RAG.retrieval   import telecharger_evenements
from RAG.embedding   import generer_embeddings
from RAG.vectorstore import construire_vectorstore_langchain, creer_retriever
from RAG.rag         import construire_chaine_rag


# Initialisation
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


# Client pour le pipeline RAG
client = MistralClient(api_key=MISTRAL_API_KEY)
llm    = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=0 # réponse la plus précise
)

# Config Ragas
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(api_key=MISTRAL_API_KEY, model="mistral-embed")
)


# Pipeline RAG
print(" Construction du pipeline")
evenements  = telecharger_evenements()
resultats   = generer_embeddings(evenements, client)
vectorstore = construire_vectorstore_langchain(resultats, client)
retriever   = creer_retriever(vectorstore)

# pour voir tous les chunks mai 2025 pour améliorer les scores
docs = retriever.invoke("événements Toulouse mai 2025")
for doc in docs:
    print(doc.metadata.get("date_debut"), "|", doc.page_content[:200])

rag_chain   = construire_chaine_rag(retriever, llm)
print(" Pipeline prêt\n")


# Ragas et ground truths

def retriever_ville(vectorstore, question, ville=None, k=5):
    """Retriever avec ou sans filtre ville selon la question."""
    r = creer_retriever(vectorstore, ville=ville, k=k)
    return r.invoke(question)

questions = [
    "Quels événements à Toulouse ont lieu en mai 2025?",
    
]

villes = [
    "Toulouse",   #  filtre sur Toulouse
    
]

ground_truths = [
    "En mai 2025 à Toulouse : la Nuit des Musées le 17 mai aux Monuments, le Festival de l'Erotisme au Grand Marché les 17 et 18 mai, le Centenaire du Planétarium à la Cité de l'espace le 7 mai, les Journées Portes Ouvertes EPITA le 17 mai.",
    
]

answers  = []
contexts = []

for q, ville in zip(questions, villes):
    docs = retriever_ville(vectorstore, q, ville, k=5)
    contexts.append([doc.page_content for doc in docs])
    response = rag_chain.invoke(q) if hasattr(rag_chain, 'invoke') else rag_chain(q)[0]
    answers.append(response)

dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

# vérification de ce que Ragas reçoit
for i, (q, a, c, g) in enumerate(zip(questions, answers, contexts, ground_truths)):
    print(f"\n=== Question {i+1} ===")
    print(f"Q: {q}")
    print(f"A: {a[:150]}")
    print(f"G: {g[:150]}")
    print(f"Contexts ({len(c)} chunks):")
    for chunk in c:
        print(f"  → {chunk[:100]}")

# Evaluation 
print(" Evaluation Ragas en cours")

# Métriques

result = evaluate(
    dataset,
    metrics=[faithfulness, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n Résultats Ragas :")
print(result)


# Assertions
def get_score(result, key):
    val = result[key]
    if isinstance(val, list):
        val = val[0]
    return float(val)

for metric_name in ["faithfulness", "context_precision", "context_recall"]:
    score = get_score(result, metric_name)
    if math.isnan(score):
        print(f"  {metric_name} = nan (calcul échoué, vérifier les logs Ragas)")
    elif score > 0.5:
        print(f" {metric_name} = {score:.3f}")
    else:
        print(f" {metric_name} = {score:.3f} — score trop faible")

print("\n Evaluation terminée")