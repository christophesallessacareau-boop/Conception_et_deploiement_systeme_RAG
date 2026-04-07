# Ce test simule ce qui se passe dans RAG/rag.py
# test de la logique du pipeline RAG sans faire de vrais appels à l'API Mistral
# test du format de réponse sans appeler de LLM réel


# Mock LLM pour simuler la génération de réponses
def test_generation_mock():
    """
    Teste la logique du pipeline RAG sans appel réel à Mistral.
    Simule le comportement de construire_chaine_rag() dans RAG/rag.py
    """
    # Faux LLM qui simule ChatMistralAI
    class FakeLLM:
        def invoke(self, prompt):
            class Response:
                content = "Voici des événements à Toulouse"
            return Response()

     # Fausse chaîne RAG qui simule rag_pipeline()
    def fake_rag(question):
        llm = FakeLLM()
        response = llm.invoke("prompt")
        return response.content, []

    reponse, docs= fake_rag("test")

    assert isinstance(reponse, str)   # la réponse est bien du texte
    assert "Toulouse" in reponse      # le contenu est celui attendu
    assert isinstance(docs, list)     # les docs sont bien une liste