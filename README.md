# Chat With Your Notes (Intro to RAG)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A lightweight, beginner-friendly **Retrieval-Augmented Generation (RAG)** starter repository designed for introductory AI Bootcamp masterclasses. Load any PDF document (such as a course syllabus or campus club constitution), index it locally with **ChromaDB** and **Hugging Face** embeddings, and chat with a grounded **Gradio** assistant that cites exact page numbers and refuses out-of-scope questions.

---

## How It Works (Closed-Book vs. Open-Book)

Standard LLMs act like students taking a **closed-book exam**—when they don't know a fact from your private notes, they guess plausibly (hallucinate). This project turns the LLM into a student taking an **open-book exam**:

1. **Chunk & Index:** Extracts text page-by-page from your PDF, slices it into overlapping 500-character chunks with page metadata, and embeds them locally using `all-MiniLM-L6-v2`.
2. **Semantic Retrieval:** Converts a user's question into a vector and retrieves the top 3 most relevant chunks from ChromaDB by meaning rather than exact keyword match.
3. **Grounded Generation & Refusal Guardrail:** Checks semantic distance (`threshold = 1.35`). If the question is unrelated to the PDF, it immediately refuses to answer; otherwise, it generates a concise answer using `google/flan-t5-base` and cites the exact source page numbers.

---

## Repository Structure

```text
chat-with-your-notes-rag/
├── app.py                              # Standalone Python & Gradio application
├── requirements.txt                    # Python dependencies
├── rag_engine.py                       #the main rag application 
├── data/
│   └── filename.pdf    # Sample 2-page document for testing
├── .gitignore
└── README.md
```

---

## Quickstart

### Option 1: Run in Google Colab (Zero Local Setup)
```bash
# Clone the specific Chat-with-your-notes-RAG repository
!git clone https://github.com/wecncodecrew/Chat-with-your-notes-RAG.git

import os
# Change the current working directory to the cloned repository
%cd /content/chat-with-your-notes-rag

# Install dependencies
!pip install -q -r requirements.txt

# Adjust launch configuration and run
!sed -i 's/demo.launch()/demo.launch(share=True)/g' app.py
!python app.py
```

### Option 2: Run Locally
```bash
git clone [https://github.com/your-username/chat-with-your-notes-rag.git](https://github.com/your-username/chat-with-your-notes-rag.git)
cd chat-with-your-notes-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## Stress-Test Prompts

Once the Gradio interface launches, run these three test queries using any pdf file `example: sample_club_constitution.pdf`:

| Test Type | Sample Prompt | Expected Behavior |
| :--- | :--- | :--- |
| **Direct Lookup** | *"What is the attendance policy for club members?"* | Returns the 75% attendance rule and cites **Page 1**. |
| **Paraphrase Search** | *"Can I skip club sessions without getting kicked out?"* | Finds the absence policy via vector similarity despite different wording and cites **Page 1**. |
| **Hallucination Refusal** | *"Who won the last FIFA World Cup?"* | Triggers the distance threshold (`> 1.35`) and refuses to answer. |

---

*Build for developers 🩶 by Wecncode Developer Community!*
