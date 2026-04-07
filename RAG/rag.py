# conctruction de la chaîne RAG : retriever + prompt + LLM

def construire_chaine_rag(retriever, llm):
    """Construit le pipeline RAG : retriever + prompt + LLM."""

    def rag_pipeline(question):
        # Cherche les chunks pertinents
        docs = retriever.invoke(question)

        # Construit le contexte
        contexte = "\n\n".join([doc.page_content for doc in docs])

        # Construit le prompt
        prompt = f"""
Tu es un assistant spécialisé dans les événements en Occitanie.
Contexte :
{contexte}
Question :
{question}
Réponds de manière claire et utile, en proposant des événements pertinents.
"""
        # Appele le LLM
        response = llm.invoke(prompt)
        return response.content, docs

    return rag_pipeline