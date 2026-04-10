
import os
import math
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (    
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from mistralai import Mistral as MistralClient
from langchain_mistralai import ChatMistralAI

from RAG.retrieval   import telecharger_evenements
from RAG.embedding   import generer_embeddings
from RAG.vectorstore import construire_vectorstore_langchain, creer_retriever
from RAG.rag         import construire_chaine_rag


# Initialisation
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# LiteLLM lit MISTRAL_API_KEY depuis les variables d'environnement
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY 

# Client pour le pipeline RAG
client = MistralClient(api_key=MISTRAL_API_KEY)
llm    = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=0
)

# Config Ragas (Ragas ne supporte pas Mistral directement)
# LiteLLM supporte Mistral et est accepté par Ragas
ragas_llm        = llm_factory("mistral/mistral-small-latest")
ragas_embeddings = embedding_factory("litellm", "mistral/mistral-embed")


# Pipeline RAG
print(" Construction du pipeline")
evenements  = telecharger_evenements()
resultats   = generer_embeddings(evenements, client)
vectorstore = construire_vectorstore_langchain(resultats, client)
retriever   = creer_retriever(vectorstore)
rag_chain   = construire_chaine_rag(retriever, llm)
print(" Pipeline prêt\n")


# Ragas est optimisé pour l'anglais, les métriques sont calibrées sur des réponses en anglais.
# on traduit les questions et ground_truths en anglais
questions = [
    "What events are happening in Toulouse?",
    "What can I do this weekend in Occitanie?",
]

ground_truths = [
    "There are several events in Toulouse including concerts, exhibitions and markets.",
    "This weekend in Occitanie you can attend festivals, markets and shows.",
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


# Evaluation 
print(" Evaluation Ragas en cours...")

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
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

for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
    score = get_score(result, metric_name)
    if math.isnan(score):
        print(f"  {metric_name} = nan (calcul échoué, vérifier les logs Ragas)")
    elif score > 0.5:
        print(f" {metric_name} = {score:.3f}")
    else:
        print(f" {metric_name} = {score:.3f} — score trop faible")

print("\n Evaluation terminée !")