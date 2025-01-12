from flask import Flask, render_template, request, send_file, jsonify
import os
import google.generativeai as genai
import pdfplumber
import docx
import csv

from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi

from PyPDF2 import PdfReader, PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate




#Set an API key...
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBBZwCnWLlKi_CR4DIrszO40g19F8jXfWM'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel("models/gemini-1.5-pro")










app = Flask(__name__, template_folder="templates")

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ('pdf', 'docx','txt')



def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()

    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages])
        return text

    elif ext == 'docx':
        doc = docx.Document(file_path)
        text = ''.join([para.text for para in doc.paragraphs])
        return text

    elif ext == 'txt':
        with open(file_path, 'r') as file:
            return file.read()
    return None





def question_generator(input_text, num):
    prompt1 = f"""
    You are an AI assistant helping the user generate a question and a corresponding answer based on the following text:
    '{input_text}'
    Please generate {num} questions from the text.And also write problem number.Each problem should have:
    - A clear question
    - The correct answer clearly indicated
    Format:
    ## Problem
    Question: [Question]
    Answer: [Answer]
    """


    prompt2 = f"""
    You are an AI assistant helping the user generate multiple-choice questions (MCQs) based on the following text:
    '{input_text}'
    Please generate {num} MCQs from the text. Each question should have:
    - A clear question
    - Four answer options (labeled A, B, C, D)
    - The correct answer clearly indicated
    Format:
    ## MCQ
    Question: [question]
    A) [option A]
    B) [option B]
    C) [option C]
    D) [option D]
    Correct Answer: [correct option]
    """

    selected_option = request.form.get("options")
    if selected_option == "option1":
        response = model.generate_content(prompt1).text.strip()
    elif selected_option == "option2":
        response = model.generate_content(prompt2).text.strip()
    return response




#Save as txt file...
def save_mcqs_to_file(mcqs, filename):
    results_path = os.path.join(app.config['RESULTS_FOLDER'], filename)

    with open(results_path, 'w', newline='') as f:
        f.write(mcqs)
    return results_path





def extract_transcript_details(youtube_video_url):
    video_id = youtube_video_url.split('=')[-1]
    transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

    t = ""
    for i in transcript_text:
        t += " " + i["text"]

    return t



def generate_video_summary(transcript_text):
    prompt = """You are youtube video summarizer. You will be taking the transcript text and summarize the entire video and provide the summary in points within 200 words. The transcript text will be appended here:"""
    response = model.generate_content([transcript_text,prompt])
    return response.text





# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


# Function to set up the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to get the response to user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response["output_text"]


























@app.route('/')
def index():
    return render_template('index.html')


@app.route('/quesGen')
def quesGen():
    return render_template('quesGen.html')



@app.route('/generate', methods=['POST'])
def generate():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    num = request.form['num']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)


        text = extract_text_from_file(file_path)


        if text:
            selected_option = request.form.get("options")
            num = int(request.form['num'])
            mcqs = question_generator(text,num)

            #print("\n\n\n", mcqs)

            #Save the generated questions to a file...
            text_file = f"generated_questions_{filename.rsplit('.', 1)[0]}.txt"
            save_mcqs_to_file(mcqs, text_file)

            return render_template('result.html', mcqs=mcqs , selected_option=selected_option, text_file=text_file)
        return "invalid file format"



@app.route('/download', methods=['GET'])
def download():
    file_path = os.path.join(app.config['RESULTS_FOLDER'])
    return send_file(file_path, as_attachment=True)



@app.route('/vidSum')
def vidSum():
    return render_template("vidSummary.html")



@app.route('/generate_vidSummary', methods=['POST'])
def summary():
    if 'vidSummary' not in request.form:
        return "invalid file format"

    vidSummary = request.form['vidSummary']

    if vidSummary:
        transcript_text = extract_transcript_details(vidSummary)
        summary = generate_video_summary(transcript_text)
        return render_template("vidSummaryResult.html", summary=summary)



@app.route('/mathSolver')
def mathSolver():
    return render_template("mathSolver.html")





@app.route("/solve", methods=["POST"])
def solve():
    pdf_docs = request.files.getlist('pdf_files')
    user_question = request.form.get('user_question')
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    answer = user_input(user_question) if user_question else "No question asked."
    return render_template('mathSolver.html', answer=answer)














if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if not os.path.exists(app.config['RESULTS_FOLDER']):
        os.makedirs(app.config['RESULTS_FOLDER'])

    app.run(debug=True)
