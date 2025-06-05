# Chat PyGPT

Este projeto é um chatbot baseado em **RAG (Retrieval-Augmented Generation)** desenvolvido com **Streamlit**, que permite ao usuário conversar com seus próprios documentos PDF utilizando modelos **LLM** como GPT e Gemini.

---

## Funcionalidades principais

* **Upload de PDFs**: Faça upload de um ou mais arquivos PDF pela barra lateral.
* **Processamento de documentos**: Os PDFs são divididos em trechos (**chunks**) para facilitar a busca e recuperação de informações.
* **Armazenamento vetorial**: Os trechos dos documentos são armazenados em uma base vetorial persistente usando o **ChromaDB**.
* **Busca contextual**: O chatbot utiliza os documentos carregados para responder perguntas, buscando sempre fornecer respostas baseadas no conteúdo dos arquivos.
* **Modelos LLM**: Permite escolher entre diferentes modelos de linguagem (**gpt-3.5-turbo**, **gpt-4**, **gemini-2.0-flash-001**) para gerar respostas.
* **Histórico de conversas**: Mantém o histórico das mensagens trocadas durante a sessão.
* **Respostas em Markdown**: As respostas são formatadas em Markdown, podendo incluir visualizações interativas.

---

## Como usar

1.  Instale as dependências listadas em `requirements.txt`.
2.  Configure as variáveis de ambiente necessárias (`GEMINI_API_KEY` e credenciais Google).
3.  Execute o arquivo `app.py` com Streamlit:
    ```bash
    streamlit run app.py
    ```
4.  Faça upload dos seus PDFs e comece a conversar com seus documentos!

---

## Observações

* O armazenamento vetorial é persistido na pasta `db`.
* As credenciais do Google devem estar no arquivo `credentials.json`.
* O sistema foi projetado para ser simples de usar e facilmente adaptável para outros tipos de documentos ou modelos LLM.
* Desenvolvido para facilitar a consulta e análise de informações em grandes volumes de documentos PDF, integrando IA generativa e busca semântica.