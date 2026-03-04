import streamlit as st
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 🔹 PAGE CONFIG
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("🧠 AI Resume Matcher (Hybrid Semantic System)")

# 🔹 LOAD MODEL (CACHED)-
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# 🔹 FILE UPLOAD & INPUT
resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste your job description here")

# 🔹 TEXT EXTRACTION FUNCTION
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# 🔹 KEYWORD ANALYSIS FUNCTION
def keyword_match_analysis(resume_text, job_text):

    skills_list = [
        "python", "sql", "machine learning", "pandas",
        "numpy", "scikit-learn", "data analysis",
        "feature engineering", "deep learning",
        "statistics", "aws", "git"
    ]

    matched_skills = []
    missing_skills = []
    required = 0

    for skill in skills_list:
        if skill in job_text:
            required += 1
            if skill in resume_text:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)

    if required == 0:
        score = 0
    else:
        score = (len(matched_skills) / required) * 100

    return score, matched_skills, missing_skills


# 🔹 ANALYZE BUTTON
if st.button("Analyze Match"):

    if resume_file and job_description:

        resume_text = extract_text_from_pdf(resume_file).lower()
        job_text = job_description.lower()

        # 🔹 KEYWORD SCORE
        keyword_score, matched_skills, missing_skills = keyword_match_analysis(
            resume_text, job_text
        )

        # 🔹 FULL DOCUMENT SEMANTIC SCORE
        resume_embedding = model.encode([resume_text])
        job_embedding = model.encode([job_text])

        full_similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        full_semantic_score = full_similarity * 100

        # 🔹 SENTENCE-LEVEL SCORE
        resume_sentences = re.split(r'[.\n]', resume_text)
        resume_sentences = [s.strip() for s in resume_sentences if len(s.strip()) > 30]

        sentence_embeddings = model.encode(resume_sentences)
        similarities = cosine_similarity(sentence_embeddings, job_embedding)

        sentence_scores = similarities.flatten()
        top_scores = sorted(sentence_scores, reverse=True)[:5]
        sentence_score = (sum(top_scores) / len(top_scores)) * 100

        # 🔥 FINAL HYBRID SCORE
        final_score = (
            0.4 * keyword_score +
            0.3 * full_semantic_score +
            0.3 * sentence_score
        )

        # 📊 DISPLAY RESULTS
        st.markdown("## 📊 Resume Match Analysis")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Keyword Match", f"{round(keyword_score, 2)} %")
            st.metric("Full Semantic Match", f"{round(full_semantic_score, 2)} %")

        with col2:
            st.metric("Sentence-Level Match", f"{round(sentence_score, 2)} %")
            st.metric("Final Hybrid Score", f"{round(final_score, 2)} %")

        st.progress(int(final_score))

        # 🧠 SKILL GAP ANALYSIS
        st.markdown("---")
        st.markdown("## 🧠 Skill Gap Analysis")

        col3, col4 = st.columns(2)

        with col3:
            st.success("Matched Skills")
            if matched_skills:
                for skill in matched_skills:
                    st.write("✔", skill)
            else:
                st.write("No direct matches found.")

        with col4:
            st.error("Missing Skills")
            if missing_skills:
                for skill in missing_skills:
                    st.write("✘", skill)
            else:
                st.write("No missing skills 🎉")

    else:
        st.warning("Please upload resume and paste job description.")




