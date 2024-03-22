import nltk
import re
import streamlit as st
import pickle

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def CleanResume(txt):
    cleanTXT = re.sub('http\S+\s', ' ', txt)  # removing http and all attached letters (\s for \)
    cleanTXT = re.sub('RT|cc', ' ', cleanTXT) # removes common Twitter-related terms often found in retweets or mentions.
    cleanTXT = re.sub('@\S+', '  ', cleanTXT) # removing @
    cleanTXT = re.sub('#\S+\s', ' ', cleanTXT) # removing #
    cleanTXT = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTXT) # removes special characters
    cleanTXT = re.sub(r'[^\x00-\x7f]', ' ', cleanTXT) #  removes non-ASCII characters (that are not in range [^\x00-\x7f])
    cleanTXT = re.sub('\s+', ' ', cleanTXT) # removes extra whitespace from the text (\r \n \t)
    return cleanTXT

# Web application

def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume', type=['doc', 'txt', 'docs', 'pdf'])
    if upload_file:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = CleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
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
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category: ", category_name)

if __name__ == "__main__":
    main()