# import logging
# import fitz  # PyMuPDF for PDF text extraction
# import faiss  # For vector similarity search
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext
# from telegram.ext.filters import Document, TEXT

# # Enable logging
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

# # Initialize models
# embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
# qa_pipeline = pipeline('question-answering')

# # Global variables for managing state
# pdf_text = ""
# index = None

# # Extract text from a PDF file
# def extract_text_from_pdf(pdf_path):
#     document = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(len(document)):
#         page = document.load_page(page_num)
#         text += page.get_text()
#     return text

# # Create FAISS index for document search
# def create_faiss_index(embedding_model, text_data):
#     embeddings = embedding_model.encode(text_data)
#     embeddings = np.array(embeddings).astype('float32')
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     return index

# # Retrieve the most relevant document chunks
# def get_relevant_document(query, embedding_model, index, text_data, top_k=3):
#     query_embedding = embedding_model.encode(query).astype('float32')
#     distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
#     relevant_texts = [text_data[idx] for idx in indices[0]]
#     return relevant_texts

# # Command: /start
# async def start(update: Update, context: CallbackContext):
#     await update.message.reply_text(
#         "Welcome to the PDF QA Bot! Upload a PDF file to begin."
#     )

# # Handle PDF file uploads
# async def handle_document(update: Update, context: CallbackContext):
#     global pdf_text, index
#     file = await update.message.document.get_file()
#     file_path = "uploaded_pdf.pdf"
#     await file.download_to_drive(file_path)
    
#     await update.message.reply_text("Processing your PDF...")
    
#     # Extract text and create FAISS index
#     pdf_text = extract_text_from_pdf(file_path)
#     pdf_chunks = pdf_text.split('\n')  # Split into lines or paragraphs
#     index = create_faiss_index(embedding_model, pdf_chunks)
    
#     await update.message.reply_text(
#         "PDF processed successfully! You can now ask questions based on the document."
#     )

# # Handle text queries
# async def handle_message(update: Update, context: CallbackContext):
#     global pdf_text, index
#     if not pdf_text or not index:
#         await update.message.reply_text(
#             "Please upload a PDF file first before asking questions."
#         )
#         return
    
#     user_query = update.message.text
#     pdf_chunks = pdf_text.split('\n')  # Split text into lines or paragraphs
#     relevant_docs = get_relevant_document(user_query, embedding_model, index, pdf_chunks)
    
#     # Combine the most relevant chunks into a single context
#     context_text = " ".join(relevant_docs)
    
#     # Generate the answer using the QA pipeline
#     try:
#         answer = qa_pipeline(question=user_query, context=context_text)
#         await update.message.reply_text(f"Answer: {answer['answer']}")
#     except Exception as e:
#         await update.message.reply_text("Sorry, I couldn't generate an answer. Please try again.")
#         logger.error(e)

# # Main function to run the bot
# def main():
#     # Replace with your Telegram bot token
#     BOT_TOKEN = "7995379796:AAF_Imy8fPlh-gWUNXkckY5nexttTlSdpGA"
    
#     # Initialize application
#     application = Application.builder().token(BOT_TOKEN).build()
    
#     # Add command and message handlers
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(MessageHandler(Document.MimeType("application/pdf"), handle_document))
#     application.add_handler(MessageHandler(TEXT & ~Document.ALL, handle_message))
    
#     # Start the bot
#     application.run_polling()

# # Run the bot
# if __name__ == '__main__':
#     main()
#-----------------------------------------------
# import logging
# from telegram import Update
# from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import fitz  # PyMuPDF for PDF processing
# from pdf2image import convert_from_path
# import pytesseract
# import nltk
# from nltk.tokenize import sent_tokenize
# import faiss
# import numpy as np

# # Ensure required NLTK data is downloaded
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # Ensure required NLTK data is downloaded


# # Configure logging
# logging.basicConfig(
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     level=logging.DEBUG,
# )
# logger = logging.getLogger(__name__)

# # Global variables
# pdf_text = ""
# index = None
# embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# # PDF text extraction functions
# def extract_text_from_pdf(pdf_path):
#     document = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(len(document)):
#         page = document.load_page(page_num)
#         text += page.get_text()

#     if not text.strip():
#         logger.debug("No text found using PyMuPDF. Attempting OCR...")
#         text = extract_text_with_ocr(pdf_path)

#     if not text.strip():
#         raise ValueError("The PDF appears to contain no readable text.")
    
#     return text

# def extract_text_with_ocr(pdf_path):
#     images = convert_from_path(pdf_path)
#     text = ""
#     for img in images:
#         text += pytesseract.image_to_string(img)
#     return text

# # Split text into chunks
# def split_text_into_chunks(text, max_chunk_size=500):
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = ""

#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= max_chunk_size:
#             current_chunk += " " + sentence
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence

#     if current_chunk.strip():
#         chunks.append(current_chunk.strip())

#     return chunks

# # Create FAISS index
# def create_faiss_index(model, texts):
#     embeddings = model.encode(texts)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings, dtype=np.float32))
#     return index

# # Find the best-matching answer
# def find_best_match(query, model, index, texts):
#     query_embedding = model.encode([query])
#     distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=1)
#     best_match_idx = indices[0][0]
#     return texts[best_match_idx]

# # Command handlers
# async def start(update: Update, context: CallbackContext):
#     await update.message.reply_text("Hello! Send me a PDF, and I will answer questions based on its content.")

# async def handle_document(update: Update, context: CallbackContext):
#     global pdf_text, index
#     file = await update.message.document.get_file()
#     file_path = "uploaded_pdf.pdf"
#     await file.download_to_drive(file_path)

#     await update.message.reply_text("Processing your PDF...")

#     try:
#         # Extract text and validate
#         pdf_text = extract_text_from_pdf(file_path)
#         logger.debug(f"Extracted text: {pdf_text[:500]}")  # Log the first 500 characters
#         pdf_chunks = split_text_into_chunks(pdf_text)
#         logger.debug(f"Number of chunks created: {len(pdf_chunks)}")
#         if not pdf_chunks:
#             raise ValueError("No meaningful text found in the PDF.")
        
#         # Create FAISS index
#         index = create_faiss_index(embedding_model, pdf_chunks)
#         logger.debug("FAISS index created successfully.")
#         await update.message.reply_text("PDF processed successfully! You can now ask questions.")
#     except Exception as e:
#         logger.error(f"Error processing PDF: {e}")
#         await update.message.reply_text(
#             f"Failed to process the PDF: {str(e)}. Please check the file and try again."
#         )

# async def handle_message(update: Update, context: CallbackContext):
#     global pdf_text, index
#     if not pdf_text or index is None:
#         await update.message.reply_text("Please upload a PDF first.")
#         return

#     query = update.message.text
#     try:
#         best_match = find_best_match(query, embedding_model, index, split_text_into_chunks(pdf_text))
#         await update.message.reply_text(best_match)
#     except Exception as e:
#         logger.error(f"Error finding match: {e}")
#         await update.message.reply_text("Sorry, I couldn't find an answer to your question.")

# # Main function
# def main():
#     TOKEN = "7995379796:AAF_Imy8fPlh-gWUNXkckY5nexttTlSdpGA"  # Replace with your bot token
#     application = ApplicationBuilder().token(TOKEN).build()

#     # Add handlers
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

#     # Run the bot
#     application.run_polling()

# if __name__ == "__main__":
#     main()
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import spacy
from PyPDF2 import PdfReader
import os

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load the QA and embedding models
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# Load SpaCy for text chunking
nlp = spacy.load("en_core_web_sm")

# Process uploaded PDFs
def process_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return None

# Chunk text using SpaCy
def split_text_into_chunks(text, max_chunk_size=500):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# Extract embeddings and find the best matching chunk
def find_answer(question, text_chunks):
    question_embedding = embedding_model.encode(question)
    chunk_embeddings = embedding_model.encode(text_chunks)

    # Compute cosine similarity
    similarities = [
        (idx, sum(a * b for a, b in zip(question_embedding, chunk_embedding)))
        for idx, chunk_embedding in enumerate(chunk_embeddings)
    ]
    best_chunk_idx = max(similarities, key=lambda x: x[1])[0]

    # Extract answer from the best chunk
    best_chunk = text_chunks[best_chunk_idx]
    result = qa_model(question=question, context=best_chunk)
    return result["answer"]

# Command: Start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Send me a PDF file, and I'll answer your questions about it.")

# Handle PDF upload
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    file_path = f"./{document.file_name}"
    file = await document.get_file()
    await file.download_to_drive(file_path)

    text = process_pdf(file_path)
    if text:
        context.user_data["pdf_text"] = text
        await update.message.reply_text("PDF uploaded and processed! You can now ask me questions about it.")
    else:
        await update.message.reply_text("Failed to process the PDF. Please check the file and try again.")

    # Clean up the file
    os.remove(file_path)

# Handle user questions
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    if "pdf_text" not in context.user_data:
        await update.message.reply_text("Please upload a PDF file first.")
        return

    text = context.user_data["pdf_text"]
    text_chunks = split_text_into_chunks(text)
    try:
        answer = find_answer(question, text_chunks)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Error finding answer: {e}")
        await update.message.reply_text("Sorry, I couldn't find an answer. Please try rephrasing your question.")

# Main function
def main():
    # Replace with your Telegram bot token
    TOKEN = "7995379796:AAF_Imy8fPlh-gWUNXkckY5nexttTlSdpGA"
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
