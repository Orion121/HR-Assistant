from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                             google_api_key = os.getenv('google_api_key') )


from langchain.prompts import ChatPromptTemplate
agent_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a helpful HR assistant for a company. 
You MUST use the tools provided to answer employee questions about office policies and to send emails.
Only respond based on the knowledge from the tools or by sending emails.
If you need to answer about company rules, policies, leave, or HR topics, first use the 'office_policy_tool'.
If you need to send an email, use the 'email sender' tool.
Always respond in a way to assist HR effectively and professionally.
Do NOT provide generic or unrelated answers."""),

    ('human', '{prompt}')
])

import smtplib
from email.message import EmailMessage
import os
import re
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv()

def send_mail(input_text: str) -> str:
    try:
        reciever_match = re.search(
            r"to\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", input_text
        )
        reciever = reciever_match.group(1) if reciever_match else None
        if not reciever:
            return "❌ Could not find recipient email."

        subject_match = re.search(r"subject\s+(?:is|=)?\s*(.*?)(?=\smessage\s|$)", input_text, re.IGNORECASE)
        subject = subject_match.group(1).strip() if subject_match else None

        message_match = re.search(r"message\s+(?:is|=)?\s*(.*)", input_text, re.IGNORECASE)
        content = message_match.group(1).strip() if message_match else None

        fallback_match = re.search(r"(?:saying|that)\s+(.*)", input_text, re.IGNORECASE)
        if not content and fallback_match:
            content = fallback_match.group(1).strip()

        if not content:
            return "Could not find email message content."

        if not subject:
            subject = f"Regarding: {content[:40]}..." if len(content) > 40 else f"Regarding: {content}"

    except Exception as e:
        return f"❌ Error during parsing: {e}"

    try:
        message = EmailMessage()
        message['From'] = os.getenv('email')
        message['To'] = reciever
        message['Subject'] = subject
        message.set_content(content)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(os.getenv('email'), os.getenv('email_pass'))
            smtp.send_message(message)

        return {
            "receiver_email": reciever,
            "subject": subject,
            "email_content": content,
            "status": "Email sent successfully."
        }

    except Exception as e:
        return {
            "receiver_email": reciever,
            "subject": subject,
            "email_content": content,
            "status": f"Failed to send email: {e}"
        }

email_tool = Tool(
    name="email sender",
    func=send_mail,
    description=(
        "Use this tool to send an email. Input must include at least the receiver email using 'to ...'. "
        "Optionally, include 'subject is ...' and 'message is ...'. If not, fallback to 'saying ...' or 'that ...'. "
        "Respond with JSON including receiver_email, subject, email_content. "
        "Example: Send an email to test@example.com subject is 'Hello' message is 'Meeting is postponed to Friday.'"
    )
)



from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader('Nexora_Office_Policy.pdf')
office_policy = loader.load()


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)
trans_office_policy = text_splitter.split_documents(office_policy)

#
from langchain.vectorstores import Chroma
vector_db = Chroma.from_documents(trans_office_policy,embeddings)
retriver = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})


from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriver,
    return_source_documents=False
)


def office_policy_info(input_):
    """ this tool is used to retrieve any information regardin office policy"""
    return qa_chain.run(input_)


policy_tool = Tool(name='office_policy_tool',
                   description="Use this tool to answer any employee questions about company rules, HR policies, leave policies, "
        "working hours, work-from-home rules, and more. The answers are retrieved from an internal office policy PDF "
        "using a retrieval-based approach.\n\n"
        "Example questions:\n"
        "- 'How many work from home days are allowed per month?'\n"
        "- 'What is the maternity leave duration?'\n"
        "- 'Is there a notice period for resignation?'\n"
        "- 'Can we wear casuals during weekdays?'\n"
        "- 'What is the grievance redressal process?'\n\n"
        "The tool uses vector similarity to fetch relevant policy content and generate a helpful, accurate response.",
        func= office_policy_info
    )


from transformers import pipeline
resume_parser = pipeline('text-classification', model="manishiitg/distilbert-resume-parts-classify",
                         tokenizer="manishiitg/distilbert-resume-parts-classify",truncation=True,max_length=512)
import PyPDF2
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


from langchain.agents import initialize_agent, AgentType, tool

agent = initialize_agent(
tools=[email_tool,policy_tool],llm=llm,
agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
prompt=agent_prompt
)
import streamlit as st 
st.title("HR AGENT")
st.write("prompt for agent")

with st.form("input_form"):
    prompt = st.text_input("Enter your prompt here")
    pdf = st.file_uploader("Upload your resume here", type=['pdf'])
    submitted = st.form_submit_button("Submit")

if submitted:
    if not prompt and not pdf:
        st.warning("no input given")
    else:
        if pdf:
            text = extract_text_from_pdf(pdf)
            content = resume_parser(text)
            st.write(content)
        if prompt:
            formatted_prompt = agent_prompt.format_prompt(prompt=prompt).to_messages()
            res = agent.invoke({'input':formatted_prompt})
            st.markdown("🤖 Agent Response")
            st.write(res['output'])





