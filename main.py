import nltk
import streamlit as st
import pickle
import docx
import PyPDF2
import re


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


clf = pickle.load(open('clf.pkl', 'rb'))
tf = pickle.load(open('tf.pkl', 'rb'))


category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    0: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    10: "Advocate"
}



def clean_resume(resume_text):
    cleanText = re.sub(r'http\S+\s', ' ', resume_text)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.lower()



def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# DOCX से टेक्स्ट निकालने का फंक्शन
def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def main():
    st.set_page_config(page_title="Resume Screening App", page_icon="📄")
    st.title("Resume Screening App")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:

        if uploaded_file.type == "application/pdf":
            resume_text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = read_docx(uploaded_file)
        else:  # txt file
            try:
                resume_text = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                resume_text = uploaded_file.read().decode("latin-1")


        cleaned_resume = clean_resume(resume_text)
        input_features = tf.transform([cleaned_resume])


        prediction_id = clf.predict(input_features)[0]


        category_name = category_mapping.get(prediction_id, "Unknown Category")


        st.subheader("Prediction Result")
        st.success(f"Resume classified as: **{category_name}**")


if __name__ == "__main__":
    main()




