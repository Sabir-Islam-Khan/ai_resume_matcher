import os
import asyncio
import tempfile
from dotenv import load_dotenv
import streamlit as st
from ResumeAI import ResumeAI


async def main():
    resumeAI = ResumeAI()

    st.header("AI Resume Matcher")

    st.subheader("Step 1: Upload Resumes")
    uploaded_files = st.file_uploader("Upload multiple resumes (PDF only)",
                                      type=['pdf'],
                                      accept_multiple_files=True)

    if uploaded_files:
        st.write(f"✅ {len(uploaded_files)} files uploaded")

    st.subheader("Step 2: Index Resumes")
    if st.button("Index Uploaded Resumes"):
        with st.status("Indexing Resumes...", expanded=True) as status:
            progress_bar = st.progress(0)

            for i, pdffile in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdffile.getvalue())
                    temp_path = tmp_file.name

                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)

                st.write(f"📄 Processing: {pdffile.name}")
                await resumeAI.indexPdfFile(temp_path, pdffile.name)
                os.unlink(temp_path)

            progress_bar.empty()
            status.update(label="Indexing Complete!", state="complete")

    st.subheader("Step 3: Upload to LlamaIndex")
    if st.button("Upload Indexed Data"):
        with st.status("Uploading to LlamaIndex...", expanded=True) as status:
            await resumeAI.upload_documents()
            status.update(label="Upload Complete!", state="complete")

    st.subheader("Step 4: Search Candidates")
    search_query = st.text_area("Describe the candidate you're looking for:",
                                height=100,
                                placeholder="Example: A software engineer with 5 years of experience in Python and Machine Learning...")

    if st.button("Search Candidates"):
        results = await resumeAI.candidates_retriever_from_query(search_query)
        resumeFiles = resumeAI.get_candidates_file_paths(results)

        print(resumeAI.get_candidates_file_paths(results))

        st.session_state.search_results = resumeFiles

    st.subheader("Results")
    with st.expander("Matching Candidates", expanded=True):
        if 'search_results' in st.session_state:
            for i, candidate in enumerate(st.session_state.search_results):
                st.write(candidate)

                st.divider()

if __name__ == "__main__":
    asyncio.run(main())
