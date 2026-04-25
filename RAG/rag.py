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
            Tu es un assistant spécialisé dans les événements sur Toulouse.
            Contexte :
            {contexte}
            Question :
            {question}
            Réponds de manière claire et utile, en proposant des événements pertinents.
            Si tu ne trouves pas d'événements pertinents, réponds que tu n'as pas d'informations à ce sujet.
            Si tu ne trouves pas d'événements sur la ville demandée, propose des événements sur Toulouse.
            Présente ta réponse de manière structurée, avec des titres, et en séparant les événements en allant à la ligne à chaque nouvel événement.
        """
        # Appele le LLM
        response = llm.invoke(prompt)
        return response.content, docs

    return rag_pipeline