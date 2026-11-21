import gradio as gr
from rag_engine import RAGEngine

# Initialize the RAG engine once on startup
rag = RAGEngine()

def handle_file_upload(file_obj):
    if file_obj is None:
        return "No file uploaded."
    return rag.ingest_pdf(file_obj.name)


def chat_response(message, history):
    return rag.query(message)


with gr.Blocks(title="Chat With Your Notes (RAG)") as demo:
    gr.Markdown("# Chat With Your Notes (Intro to RAG)")
    gr.Markdown(
        "Upload a PDF document, wait for it to index, and ask questions grounded strictly in the text."
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            status_box = gr.Textbox(
                label="Indexing Status",
                value="Waiting for PDF upload...",
                interactive=False,
            )
            pdf_input.upload(
                fn=handle_file_upload, inputs=[pdf_input], outputs=[status_box]
            )

        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat_response,
                examples=[
                    "What is the main topic of this document?",
                    "Summarize the key findings.",
                    "What is the capital of Mars?",
                ],
            )

if __name__ == "__main__":
    demo.launch()
