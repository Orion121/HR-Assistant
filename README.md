# 🤖 HR Assistant with Email Automation, Policy Q&A, and Resume Scoring

This project is an intelligent HR assistant built using **LangChain**, **Google Gemini**, and **Streamlit**. It automates key HR functions including policy-based Q&A, email communication, and resume evaluation — all in a single interactive interface.

---

## ✨ Features

- 🔍 **Policy Question Answering (RAG)**  
  Uses Retrieval-Augmented Generation to answer employee queries from an internal office policy PDF using semantic search.

- 📩 **Automated Email Sending**  
  Parses natural language input to extract recipient, subject, and message content, then sends emails automatically via Gmail SMTP.

- 📄 **Resume Scoring**  
  Evaluates uploaded resumes using Hugging Face’s transformer-based classification pipeline and returns structured analysis.

- 🖥️ **Interactive Streamlit UI**  
  Clean and responsive interface for prompting, uploading resumes, and viewing results in real time.

---

## 🧠 Tech Stack

| Category          | Tools/Frameworks                                                                 |
|------------------|-----------------------------------------------------------------------------------|
| Language Model    | [Gemini API (via LangChain)](https://ai.google.dev)                              |
| Orchestration     | LangChain Agents, Tools, PromptTemplates                                         |
| Retrieval         | Hugging Face Embeddings + Chroma Vector DB                                       |
| Resume Analysis   | Hugging Face Transformers (`distilbert-resume-parts-classify`)                   |
| Frontend          | Streamlit                                                                         |
| PDF Handling      | PyPDF2                                                                             |
| Email Automation  | SMTP (Gmail), Regex                                                               |

---

## 📂 File Structure


