from dotenv import load_dotenv
import nest_asyncio
from llama_index.llms.openai import OpenAI

from llama_parse import LlamaParse
from pathlib import Path
from llama_index.core import Document
from llama_cloud.types import CloudDocumentCreate
from pydantic import BaseModel, Field
from typing import List
from llama_cloud.client import LlamaCloud
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs
import os 
import asyncio

from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)



load_dotenv()

global_skills = []
global_countries = []
global_domains = []

LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

nest_asyncio.apply()

llm = OpenAI(model='gpt-4o-mini')

def parse_files(pdf_files):
    """Function to parse the pdf files using LlamaParse in markdown format"""

    parser = LlamaParse(
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
    )

    documents = []

    for index, pdf_file in enumerate(pdf_files):
        print(f"Processing file {index + 1}/{len(pdf_files)}: {pdf_file}")
        docs = parser.load_data(pdf_file)
        # Updating metadata with filepath
        for doc in docs:
          doc.metadata.update({'filepath': pdf_file})
        documents.append(docs)

    return documents

def list_pdf_files(directory):
    # List all .pdf files recursively using pathlib
    # rglob ('recursive glob') searches through all subdirectories
    pdf_files = [str(file) for file in Path(directory).rglob('*.pdf')]
    return pdf_files

class Metadata(BaseModel):
    """
    A data model representing key professional and educational metadata extracted from a resume.
    This class captures essential candidate information including technical/professional skills
    and the geographical distribution of their educational background.

    Attributes:
        skills (List[str]): Technical and professional competencies of the candidate
        country (List[str]): Countries where the candidate pursued formal education

    Example:
        {
            "skills": ["Python", "Machine Learning", "SQL", "Project Management"],
            "country": ["United States", "India"],
            "domain": "Information Technology"
        }
    """

    domain: str = Field(...,
                        description="The domain of the candidate can be one of SALES/ IT/ FINANCE"
                                    "Returns an empty string if no domain is identified.")

    skills: List[str] = Field(
        ...,
        description="List of technical, professional, and soft skills extracted from the resume. "
                   "and domain expertise. Returns an empty list if no skills are identified."
    )

    country: List[str] = Field(
        ...,
        description="List of countries where the candidate completed their formal education, Only extract the country."
                   "Returns an empty list if countries are not specified."
    )


def create_llamacloud_pipeline(pipeline_name, embedding_config, transform_config, data_sink_id=None):
    """Function to create a pipeline in llamacloud"""

    client = LlamaCloud(token=LLAMA_CLOUD_API_KEY)

    pipeline = {
        'name': pipeline_name,
        'transform_config': transform_config,
        'embedding_config': embedding_config,
        'data_sink_id': data_sink_id
    }

    pipeline = client.pipelines.upsert_pipeline(request=pipeline)

    return client, pipeline


async def get_metadata(text):
    """Function to get the metadata from the given resume of the candidate"""
    prompt_template = PromptTemplate("""Generate skills, and country of the education for the given candidate resume.

    Resume of the candidate:

    {text}""")

    metadata = await llm.astructured_predict(
        Metadata,
        prompt_template,
        text=text,
    )

    return metadata

async def get_document_upload(documents, llm):
    full_text = "\n\n".join([doc.text for doc in documents])

    # Get the file path of the resume
    file_path = documents[0].metadata['filepath']

    # Extract metadata from the resume
    extracted_metadata = await get_metadata(full_text)

    skills = list(set(getattr(extracted_metadata, 'skills', [])))
    country = list(set(getattr(extracted_metadata, 'country', [])))
    domain = getattr(extracted_metadata, 'domain', '')

    global_skills.extend(skills)
    global_countries.extend(country)
    global_domains.append(domain)

    return CloudDocumentCreate(
                text=full_text,
                metadata={
                    'skills': skills,
                    'country': country,
                    'domain': domain,
                    'file_path': file_path
                }
            )



async def upload_documents(client, pipeline, documents):
    """Function to upload the documents to the cloud"""

    # Upload the documents to the cloud
    extract_jobs = []
    for doc in documents:
        extract_jobs.append(get_document_upload(doc, llm))

    documents_upload_objs = await run_jobs(extract_jobs, workers=4)

    _ = client.pipelines.create_batch_pipeline_documents(pipeline.id, request=documents_upload_objs)
    await upload_documents(client, pipeline, documents)



index = LlamaCloudIndex(
  name="resume_matching",
  project_name="Default",
  organization_id="9eea158a-7ba5-49e8-bff7-3c060289e5f6",
)

async def get_query_metadata(text):
    """Function to get the metadata from the given user query"""
    prompt_template = PromptTemplate("""Generate skills, and country of the education for the given user query.

    Extracted metadata should be from the following items:

    skills: {global_skills}
    countries: {global_countries}
    domains: {global_domains}
    user query:

    {text}""")

    extracted_metadata = await llm.astructured_predict(
        Metadata,
        prompt_template,
        text=text,
        global_skills=global_skills,
        global_countries=global_countries,
        global_domains=global_domains
    )

    return extracted_metadata

async def candidates_retriever_from_query(query: str):
    """Synthesizes an answer to your question by feeding in an entire relevant document as context."""
    print(f"> User query string: {query}")
    # Use structured predict to infer the metadata filters and query string.
    metadata_info = await get_query_metadata(query)
    filters = MetadataFilters(
    filters=[
        MetadataFilter(key="domain", operator=FilterOperator.EQ, value=metadata_info.domain),
        MetadataFilter(key="country", operator=FilterOperator.IN, value=metadata_info.country),
        MetadataFilter(key="skills", operator=FilterOperator.IN, value=metadata_info.skills)
    ],
    condition=FilterCondition.OR
)
    print(f"> Inferred filters: {filters.json()}")
    retriever = index.as_retriever(
    retrieval_mode="chunks",
    metadata_filters=filters,
    )
    # run query
    return retriever.retrieve(query)

def get_candidates_file_paths(candidates):

  file_paths = []
  for candidate in candidates:
    file_paths.append(candidate.metadata['file_path'])

  return list(set(file_paths))

async def candidates_retriever_from_jd(job_description: str):
    # Use structured predict to infer the metadata filters and query string.
    metadata_info = await get_metadata(job_description)
    filters = MetadataFilters(
    filters=[
        MetadataFilter(key="domain", operator=FilterOperator.EQ, value=metadata_info.domain),
        MetadataFilter(key="country", operator=FilterOperator.IN, value=metadata_info.country),
        MetadataFilter(key="skills", operator=FilterOperator.IN, value=metadata_info.skills)
    ],
    condition=FilterCondition.OR
)
    print(f"> Inferred filters: {filters.json()}")
    retriever = index.as_retriever(
    retrieval_mode="chunks",
    metadata_filters=filters,
    )
    # run query
    return retriever.retrieve(job_description)


def main():
    directory = './sampled_data/'
    pdf_files = list_pdf_files(directory)

    documents = parse_files(pdf_files)

    global_skills = []
    global_countries = []
    global_domains = []

    # Embedding config
    embedding_config = {
        'type': 'OPENAI_EMBEDDING',
        'component': {
            'api_key': os.environ["OPENAI_API_KEY"], # editable
            'model_name': 'text-embedding-ada-002' # editable
        }
    }

    # Transformation auto config
    transform_config = {
        'mode': 'auto',
        'config': {
            'chunk_size': 1024,
            'chunk_overlap': 20
        }
    }

    client, pipeline = create_llamacloud_pipeline('resume_matching', embedding_config, transform_config)

    index = LlamaCloudIndex(
        name="resume_matching",
        project_name="Default",
        organization_id="9eea158a-7ba5-49e8-bff7-3c060289e5f6",
    )

    async def run():
        query = "I want someone who studied in US, Java developer, and worked in IT"
        nodes = await candidates_retriever_from_query(query)
        print(get_candidates_file_paths(nodes))

        job_description_file_path = 'test_data/job_description.pdf'

        job_description_document = parse_files([job_description_file_path])

        job_description = "\n\n".join([doc.text for doc in job_description_document[0]])

        print(job_description)

        candidates_based_on_jd = await candidates_retriever_from_jd(job_description)

        candidates_file_paths = get_candidates_file_paths(candidates_based_on_jd)

        print(candidates_file_paths)

        candidates_resumes = parse_files(candidates_file_paths)

        candidates_resumes_text = "\n\n".join([doc.text for docs in candidates_resumes for doc in docs])

        query = f"""Based on the following job description, please share the analysis of why specific candidates are suitable for the job.

        Job Description:
        {job_description}

        Candidates:
        {candidates_resumes_text}
        """

        analyses = llm.complete(query)

        print(analyses)

    asyncio.run(run())

if __name__ == "__main__":
    main()