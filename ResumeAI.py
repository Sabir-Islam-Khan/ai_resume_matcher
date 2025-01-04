import os
import nest_asyncio
from llama_parse import LlamaParse
from llama_cloud.client import LlamaCloud
from dotenv import load_dotenv
from llama_parse import LlamaParse
from pathlib import Path
from llama_index.core import Document
from llama_cloud.types import CloudDocumentCreate
from pydantic import BaseModel, Field
from typing import List
from llama_cloud.client import LlamaCloud
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs
from llama_index.llms.openai import OpenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)
from metadata import Metadata

load_dotenv()


class ResumeAI:
    llamaIndexParser: LlamaParse = None
    resumeDocuments: list = []
    tempDir: str = "/temp"

    llamaCloudClient: LlamaCloud = None
    llamaCloudPipeline = None

    global_skills = []
    global_countries = []
    global_domains = []

    llm = OpenAI(model='gpt-4o-mini')

    orgId: str = "9eea158a-7ba5-49e8-bff7-3c060289e5f6"
    pipelineName: str = "resume_matching"

    openAIKey: str = os.getenv('OPENAI_API_KEY')
    llamaIndexKey: str = os.getenv('LLAMA_CLOUD_API_KEY')

    llamaIndexCloud = llamaIndexCloud = LlamaCloudIndex(
        name=pipelineName,
        project_name="Default",
        organization_id=orgId,
        api_key=llamaIndexKey,
    )

    def __init__(self):
        nest_asyncio.apply()

        self.llamaIndexParser = LlamaParse(
            result_type="markdown",
            num_workers=4,
            verbose=True,
        )

        self.create_llamacloud_pipeline()

    async def indexPdfFile(self, pdf_file: str):
        docs = self.llamaIndexParser.load_data(pdf_file)
        for doc in docs:
            doc.metadata.update({'filepath': pdf_file})
        self.resumeDocuments.append(docs)

    def create_llamacloud_pipeline(self, data_sink_id=None):
        print(self.llamaIndexKey)
        print(self.openAIKey)

        embedding_config = {
            'type': 'OPENAI_EMBEDDING',
            'component': {
                'api_key': self.openAIKey,
                'model_name': 'text-embedding-ada-002'
            }
        }

        transform_config = {
            'mode': 'auto',
            'config': {
                'chunk_size': 1024,
                'chunk_overlap': 20
            }
        }

        self.llamaCloudClient = LlamaCloud(
            token=self.llamaIndexKey)

        pipeline = {
            'name': self.pipelineName,
            'transform_config': transform_config,
            'embedding_config': embedding_config,
            'data_sink_id': data_sink_id
        }

        self.llamaCloudPipeline = self.llamaCloudClient.pipelines.upsert_pipeline(
            request=pipeline)

    async def get_metadata(self, text):
        prompt_template = PromptTemplate("""Generate skills, and country of the education for the given candidate resume.

        Resume of the candidate:

        {text}""")

        metadata = await self.llm.astructured_predict(
            Metadata,
            prompt_template,
            text=text,
        )

        return metadata

    async def get_document_upload(self, documents):
        full_text = "\n\n".join([doc.text for doc in documents])
        file_path = documents[0].metadata['filepath']
        extracted_metadata = await self.get_metadata(full_text)

        skills = list(set(getattr(extracted_metadata, 'skills', [])))
        country = list(set(getattr(extracted_metadata, 'country', [])))
        domain = getattr(extracted_metadata, 'domain', '')

        self.global_skills.extend(skills)
        self.global_countries.extend(country)
        self.global_domains.append(domain)

        return CloudDocumentCreate(
            text=full_text,
            metadata={
                'skills': skills,
                'country': country,
                'domain': domain,
                'file_path': file_path
            }
        )

    async def upload_documents(self):
        extract_jobs = []
        for doc in self.resumeDocuments:
            extract_jobs.append(self.get_document_upload(doc))

        documents_upload_objs = await run_jobs(extract_jobs, workers=4)

        _ = self.llamaCloudClient.pipelines.create_batch_pipeline_documents(
            self.llamaCloudPipeline.id, request=documents_upload_objs)

    async def get_query_metadata(self, text):
        prompt_template = PromptTemplate("""Generate skills, and country of the education for the given user query.

        Extracted metadata should be from the following items:

        skills: {global_skills}
        countries: {global_countries}
        domains: {global_domains}
        user query:

        {text}""")

        extracted_metadata = await self.llm.astructured_predict(
            Metadata,
            prompt_template,
            text=text,
            global_skills=self.global_skills,
            global_countries=self.global_countries,
            global_domains=self.global_domains
        )

        return extracted_metadata

    async def candidates_retriever_from_query(self, query: str):
        print(f"> User query string: {query}")

        metadata_info = await self.get_query_metadata(query)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="domain", operator=FilterOperator.EQ,
                               value=metadata_info.domain),
                MetadataFilter(key="country", operator=FilterOperator.IN,
                               value=metadata_info.country),
                MetadataFilter(key="skills", operator=FilterOperator.IN,
                               value=metadata_info.skills)
            ],
            condition=FilterCondition.OR
        )

        retriever = self.llamaIndexCloud.as_retriever(
            retrieval_mode="chunks",
            metadata_filters=filters,
        )

        return retriever.retrieve(query)

    def get_candidates_file_paths(self, candidates):

        file_paths = []
        for candidate in candidates:
            file_paths.append(candidate.metadata['file_path'])

        return list(set(file_paths))

    async def candidates_retriever_from_jd(self, job_description: str):
        metadata_info = await self.get_metadata(job_description)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="domain", operator=FilterOperator.EQ,
                               value=metadata_info.domain),
                MetadataFilter(key="country", operator=FilterOperator.IN,
                               value=metadata_info.country),
                MetadataFilter(key="skills", operator=FilterOperator.IN,
                               value=metadata_info.skills)
            ],
            condition=FilterCondition.OR
        )
        retriever = self.llamaIndexCloud.as_retriever(
            retrieval_mode="chunks",
            metadata_filters=filters,
        )

        return retriever.retrieve(job_description)
