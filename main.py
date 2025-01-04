import os
import nest_asyncio
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse, ResultType
from pathlib import Path
from llama_index.core import Document
from llama_cloud.types import CloudDocumentCreate
from pydantic import BaseModel, Field
from typing import List
from llama_cloud.client import LlamaCloud
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs

def parse_files(pdf_files):
    parser = LlamaParse(
        result_type = ResultType("markdown"),
        num_workers=4,
        verbose=True,
    )

    documents = []

    for index, pdf_file in enumerate(pdf_files):
        print(f"Processing file {index + 1}/{len(pdf_files)}: {pdf_file}")
        docs = parser.load_data(pdf_file)


        print(docs)

def list_pdf_files(directory):
    pdf_files = [str(file) for file in Path(directory).rglob('*.pdf')]
    return pdf_files

if __name__=="__main__":
    load_dotenv()
    cv_directory = "./sampled_data"
    pdf_files = list_pdf_files(cv_directory)
    parse_files(pdf_files)