import os
import asyncio
import tempfile
from dotenv import load_dotenv
import streamlit as st
from ResumeAI import ResumeAI

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
)


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

    st.subheader("Step 3: Upload to Vector Store")
    if st.button("Upload Indexed Data"):
        with st.status("Uploading to LlamaIndex...", expanded=True) as status:
            await resumeAI.upload_documents()
            status.update(label="Upload Complete!", state="complete")

    st.subheader("Step 4: Search Candidates")

    # Create tabs for different search methods
    search_tab, jd_tab = st.tabs(
        ["Search by Query", "Search by Job Description"])

    with search_tab:
        search_query = st.text_area("Describe the candidate you're looking for:",
                                    height=100,
                                    placeholder="Example: A software engineer with 5 years of experience in Python and Machine Learning...")

        if st.button("Search Candidates"):
            with st.spinner('🔍 Searching for matching candidates...'):
                results = await resumeAI.candidates_retriever_from_query(search_query)
                resumeFiles = resumeAI.get_candidates_file_paths(results)
                st.session_state.search_results = resumeFiles

    with jd_tab:
        uploaded_jd = st.file_uploader("Upload Job Description (PDF only)",
                                       type=['pdf'],
                                       key="jd_uploader")

        if uploaded_jd is not None:
            if st.button("Search Using Job Description"):
                with st.spinner('🔍 Processing Job Description and searching candidates...'):
                    with st.status("Processing Job Description...", expanded=True) as status:
                        # Save JD to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_jd.getvalue())
                            temp_path = tmp_file.name

                        # Get results using JD
                        results = await resumeAI.candidates_retriever_from_jd(temp_path)
                        resumeFiles = resumeAI.get_candidates_file_paths(
                            results)
                        os.unlink(temp_path)

                        st.session_state.search_results = resumeFiles
                        status.update(label="Search Complete!",
                                      state="complete")

    st.subheader("Results")
    with st.expander("Matching Candidates", expanded=True):
        if 'search_results' in st.session_state:
            for i, candidate in enumerate(st.session_state.search_results):
                st.write(candidate)

                st.divider()

if __name__ == "__main__":
    asyncio.run(main())
