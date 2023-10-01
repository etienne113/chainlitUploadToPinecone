from chainlit.input_widget import Tags
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import tempfile
import chainlit as cl
from dotenv import load_dotenv
import os
import pinecone
import openai

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_ENV"] = os.getenv('PINECONE_ENV')

embeddings = OpenAIEmbeddings()

welcome_message = """ Welcome to the Chainlit PDF QA demo! To get started:
  1. Upload one or more PDF or text files \n
  2. Ask questions about the files
  """


def sort_strings_alphabetically(input_list):
    sorted_list = sorted(input_list)
    return sorted_list


def process_files(files):
    processed_docs = []

    for file in files:
        if file.type == "application/pdf":
            # Handle PDF files
            Loader = PyPDFLoader
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        elif file.type == "text/plain":
            # Handle  text files
            Loader = TextLoader
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        elif file.type == "text/csv":
            Loader = CSVLoader
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        else:
            raise ValueError("Unsupported file type: " + file.type)

        # Handle PDF and plain text files using existing logic
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(file.content)
            loader = Loader(temp_file.name)
            documents = loader.load()

            docs = splitter.split_documents(documents)
            metadata = cl.user_session.get('metadata')

            for i, doc in enumerate(docs):
                doc.metadata["source"] = f"source_{i}"
                doc.metadata['departments'] = metadata.get('departments')
                processed_docs.append(doc)
    cl.user_session.set("docs", docs)
    return processed_docs


def get_doc_from_pinecone(file):
    doc = process_files(file)

    if not doc:
        return None

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
    )

    index_name = os.getenv('PINECONE_INDEX_NAME')
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
    docsearch = Pinecone.from_documents(doc, embeddings, index_name=index_name)
    cl.user_session.set("doc", doc)
    return docsearch


@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to this space, you can use this to chat with your PDFs and text files").send()

    file = None
    while file is None:
        file = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf", "text/csv"],
            max_files=1,
            max_size_mb=20,
            timeout=1800,
        ).send()
    file_names = [file.name for file in file]

    msg = cl.Message(
        content=f"{len(file)} file : {','.join(file_names)} is being uploaded..."
    )
    await msg.send()
    await cl.Message(content="Now select the department(s) the uploaded document should be assigned to.").send()
    await cl.Message(content="Be aware that if you don't choose any department or include the option \"none\" in the "
                             "list of tags,"
                             "the document would be accessible for "
                             "any department").send()
    input_list = ["Software engineering", "DevOps", "Project management", "Data management", "Personal management"]
    departments_list = sort_strings_alphabetically(input_list=input_list)
    settings = await cl.ChatSettings(
        [
            Tags(
                id="departments",
                label="Please select the department(s): ",
                initial=["none"] + departments_list,
            )
        ]
    ).send()
    default_value = False
    values = settings["departments"]
    metadata_args = []
    metadata = {"departments": "none"}
    for item in values:
        if item == "none":
            default_value = True
            break
        else:
            metadata_args.append(item)
    if not default_value:
        metadata['departments'] = metadata_args
    cl.user_session.set('metadata', metadata)
    cl.user_session.set('files', file)


@cl.on_settings_update
async def handle_update(settings):
    await cl.Message(content=f"Uploading files is being processed...").send()
    files = cl.user_session.get('files')
    selected_option = settings['departments']
    metadata = {"departments": selected_option}
    cl.user_session.set("metadata", metadata)
    docsearch = await cl.make_async(get_doc_from_pinecone)(files)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limits=4097),
    )
    await cl.Message(content=f"Uploaded files processed succesfully!").send()
    await cl.Message(content=f"In this interface you can't filter your results with metadata"
                             " , you should procure yourself the corresponding interface one").send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    docs = cl.user_session.get("docs")
    metadata = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadata]

    if sources:
        found_sources = []

        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)

            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {','.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
