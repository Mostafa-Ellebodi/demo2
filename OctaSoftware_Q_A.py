#################
#### Imports ####
#################
import streamlit as st
from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
import haystack

import os
import shutil
from pathlib import Path
import pdftotext
from PIL import Image
from pyunpack import Archive
import gc

from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.faiss import FAISSDocumentStore

#helper function for creating new empty directories
def clean_mkdir(path):
    if Path(path).exists():
        shutil.rmtree(path)
    os.makedirs(path)

########################
#### Document Store ####
########################
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

dataPath = "./my_data/"
clean_mkdir(dataPath)

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

Archive('file.7z').extractall("./my_pdf_files/")

#convert pdf files to txt files
for fn in os.listdir(pdfPath):
    print("processing", fn)
    with open(pdfPath+fn, "rb") as f:
        pdf = pdftotext.PDF(f)
        print("converted to text")
    # Save all text to a txt file.
    with open(dataPath + fn.split(".")[0]+".txt", 'w') as f:
        f.write("\n\n".join(pdf))
        print("wrritten as text")
##############################
#### Write uploaded files ####
##############################
# print("writing uploaded files")
# for uploaded_file in uploaded_files:
    # input_text = uploaded_file.read().decode()
    # fn = uploaded_file.name
    # filePath = dataPath + fn
    # # Write-Overwrites 
    # file1 = open(filePath,"w")#write mode 
    # file1.write(input_text) 
    # file1.close() 



#########################
#### Ask  a question ####
#########################

inputQuestion = st.text_input("Enter you question here", "")


if st.button("Get Answer!"):
    try:
        if inputQuestion[-1] != "?":
            inputQuestion += "?"
        with st.spinner('Searching for the answer. Please wait...'):
            #########################################
            #### Cleaning and indexing doucments ####
            #########################################
            # Let's first get some files that we want to use
            doc_dir = dataPath
            # s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
            # fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

            # Convert files to dicts
            dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

            # Now, let's write the dicts containing documents to our DB.
            document_store.write_documents(dicts)


            ################################################
            #### Initalize Retriever, Reader,  & Finder ####
            ################################################
            ###############
            ## Retriever ##
            ###############
            retriever = DensePassageRetriever(document_store=document_store,
                                            # query_embedding_model="xlnet-base-cased",
                                            # passage_embedding_model="xlnet-base-cased",
                                            query_embedding_model="bert-base-uncased",
                                            passage_embedding_model="bert-base-uncased",
                                            max_seq_len_query=64,
                                            max_seq_len_passage=512,
                                            batch_size=16,
                                            use_gpu=True,
                                            embed_title=True,
                                            use_fast_tokenizers=False)
            # Important: 
            # Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
            # previously indexed documents and update their embedding representation. 
            # While this can be a time consuming operation (depending on corpus size), it only needs to be done once. 
            # At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
            document_store.update_embeddings(retriever)

            ############
            ## Reader ##
            ############
            # Load a  local model or any of the QA models on
            # Hugging Face's model hub (https://huggingface.co/models)

            reader = FARMReader(model_name_or_path="deepset/bert-large-uncased-whole-word-masking-squad2", 
                                use_gpu=True, 
                                context_window_size=1024,
                                )

            ############
            ## Finder ##
            ############
            finder = Finder(reader, retriever)
            print("making predictions")
            # You can configure how many candidates the reader and retriever shall return
            # The higher top_k_retriever, the better (but also the slower) your answers. 
            prediction = finder.get_answers(question=inputQuestion, top_k_retriever=1, top_k_reader=1)

            #prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)
            #prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_retriever=10, top_k_reader=5)
            print("just printing success statements")
            st.success("The answer is: \n" + prediction['answers'][0]['answer'])
            st.success("The context of the answer is: \n" + prediction['answers'][0]['context'])
            st.write("You can enter a new question and hit Get Answer or upload new documents")
            gc.collect()
    except:
        st.write("ERROR: Reload app and make sure you are entering values correctly")