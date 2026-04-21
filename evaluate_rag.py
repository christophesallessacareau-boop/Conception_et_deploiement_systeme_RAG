
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
rag_chain   = construire_chaine_rag(retriever, llm)
print(" Pipeline prêt\n")


# Ragas et ground truths

questions = [
    "Quels événements à Toulouse ont lieu en avril 2025?",
    "Quels événements à Toulouse ont lieu en mai 2025?",
]

ground_truths = [
    "Un dimanche, un quartier - Balade Urbaine Soupetard, L'échappée Belle / Rencontre avec Pamela Varela, Sitabaomba, Chez les zébus francophones / Rencontre avec Nantenaina Lova, Découverte des instruments de musique électronique, Walking in the movies / Rencontre avec Kim Lyang, Courts-métrages indiens, Mario Kart sur Switch, Atelier Yoga Intergénérationnel, Atelier de sophrologie spécial intergénérationnel aux Halles.",
    "Mai à Vélo 2025, Balade à vélo à la découverte des projets co-financés par l'Union européenne à Toulouse (31), Vélos Zextraordinaires, Balade Nocturne Toulouse à Vélo, Révision vélo, Challenge Allons-Y A Vélo (AYAV), Petit-déjeuner cycliste, Initiation à la communication radio, Nuit des Musées au Musée départemental de la Résistance."
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