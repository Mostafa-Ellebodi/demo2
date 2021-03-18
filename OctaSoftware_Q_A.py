#################
#### Imports ####
#################
import streamlit as st
import haystack
import os
import shutil
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from pathlib import Path
from pyunpack import Archive
import gc



import os
from subprocess import Popen, PIPE, STDOUT
es_server = Popen(['elasticsearch-7.9.2/bin/elasticsearch'],stdout=PIPE, stderr=STDOUT)
#,preexec_fn=lambda: os.setuid(1)   # as daemo
# wait until ES has started
cmd0 = 'sleep 30'
os.system(cmd0)

#helper function for creating new empty directories
def clean_mkdir(path):
    if Path(path).exists():
        shutil.rmtree(path)
    os.makedirs(path)

#######################
#### Load T5 Model ####
#######################
model_name = "allenai/unifiedqa-t5-3b" # you can specify the model size here
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

print("loaded model successfuly")
############################
#### setup basic layout ####
############################
# To disable warnings
st.set_option('deprecation.showfileUploaderEncoding', False)


st.write("""
  # Get your questions answered!
""")
# st.subheader("upload your text file and look at TOC")


st.write("Upload your .7z file containing your pdf documents")
uploaded_file  = st.file_uploader("Please upload your .7z file: ", type=["7z"], accept_multiple_files=False)

if uploaded_file==None:
    # st.write("Please Upload Your file")
    st.stop()

zipPath = "file.7z"
with open(zipPath, 'wb') as zipFile:
    zipFile.write(uploaded_file.getvalue())


pdfPath = "./my_pdf_files/"
clean_mkdir(pdfPath)
print("created directory ")

Archive('file.7z').extractall("./my_pdf_files/")
print("extracted files")


##############################
#### Preprocess documents ####
##############################
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

all_docs = convert_files_to_dicts(dir_path="./my_pdf_files/", clean_func=clean_wiki_text, split_paragraphs=True)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by='passage',  ###################### 'word', 'sentence' or 'passage'
    split_length=1,  ################# number of units either 'word', 'sentence' or 'passage'
    split_respect_sentence_boundary=False,    
    split_overlap=0 ################# number of units overlap either 'word', 'sentence' or 'passage'
)
nested_docs = [preprocessor.process(d) for d in all_docs]
docs = [d for x in nested_docs for d in x]

print(f"n_files_input: {len(all_docs)}\nn_docs_output: {len(docs)}")

document_store.write_documents(docs)

print("fininshed document stores")

####################################################################
#### Initalize Retriever, Reader,  & FinderPreprocess documents ####
####################################################################
retriever = ElasticsearchRetriever(document_store=document_store)
print("loaded retriever")


#########################
#### Ask  a question ####
#########################

inputQuestion = st.text_input("Enter your question here", "")


if st.button("Get Answer!"):
    # try:
    if inputQuestion[-1] != "?":
        inputQuestion += "?"
    with st.spinner('Searching for the answer. Please wait...'):

        contexts = retriever.retrieve(query=inputQuestion,  top_k=10)

        for i  in tqdm(range(len(contexts)), leave = True, position=0):
            contexts[i].answer = run_model(q + ' \\n ' + contexts[i].text)[0]
            contexts[i] = contexts[i].to_dict()



        st.success("The answer is: \n" + contexts[0]['answer'])
        st.success("The context of the answer is: \n" + contexts[0]['text'])
        st.write("You can enter a new question and hit Get Answer or upload new documents")
        # gc.collect()
    # except:
    #     st.write("ERROR: Reload app and make sure you are entering values correctly")

###################################################################################################################3