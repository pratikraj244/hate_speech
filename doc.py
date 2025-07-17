import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import streamlit.components.v1 as components
from PIL import Image
import streamlit as st
import base64
from io import BytesIO
import spacy
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import subprocess

st.set_page_config(page_title="CleanChat", layout="wide")
def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

# Load image and convert
img = Image.open(r"desc.png")
img_base64 = pil_to_base64(img)

# Set Streamlit layout
# Apply fine-tuned CSS styling
st.markdown("""
    <style>
        .top-left-logo {
            top: 20px;
            left: 75px;
            font-size: 38px;
            font-weight: 900;
            z-index: 9999;
            padding: 4px 10px;
            border-radius: 6px;
            font-family: 'Segoe UI', sans-serif;
            letter-spacing: 1px;
        }

        .top-left-logo span {
            display: inline-block;
            line-height: 1.2;
        }
    </style>

    <div class="top-left-logo">
        <span>CLEAN</span>
        <span style='background: linear-gradient(to right, #4facfe, #00f2fe);
                     -webkit-background-clip: text;
                     -webkit-text-fill-color: transparent;
                     font-weight: 900;'>
            CHAT
        </span>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
        .hero-box {
            background-color: #8ec1f7;
            padding: 40px 30px;
            border-radius: 0px;
            margin-top: 13px;
            margin-left: 5px;
            margin-right: 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 340px;          
            width: 100%; 
        }

        .hero-text {
            color: white;
            max-width: 55%;
        }

        .hero-text h1 {
            font-size: 50px;
            margin-bottom: 10px;
            margin-left: 50px;
            margin-top: 20px;
            font-weight: 800;
            font-family: 'Poppins', sans-serif
        }

        .hero-text p {
            font-size: 26px;
            margin-bottom: 40px;
            margin-left: 50px;
            max-width: 100%
        }

        .hero-image {
            max-width: 35%;
            position: relative;
            margin-right: 60px;
        }

        .hero-image img {
            width: 100%;
        }

        .emoji {
            font-size: 70px;
            position: absolute;
        }

        .emoji.smile {
            bottom: 30px;
            left: -60px;
        }

        .emoji.angry {
            top: 20px;
            right: -25px;
        }

        .input-box {
            background-color: white;
            border-radius: 20px;
            padding: 10px 20px;
            margin-left: 30px;
            width: 80%;
            border: none;
            font-size: 16px;
        }

        input::placeholder {
            color: #aaa;
        }
    </style>
""", unsafe_allow_html=True)

# Embed everything inside hero box
st.markdown(f"""
    <div class="hero-box">
        <div class="hero-text">
            <h1><span style="color:white;">ANALYZE</span> any chats<br><span style="color:#01449e;">instantly!</span></h1>
            <p>Applicable for all social media platforms such as<br>facebook, instagram, reddit and many more...</p>
        </div>
        <div class="hero-image">
            <img src="data:image/png;base64,{img_base64}" />
            <div class="emoji smile">😄</div>
            <div class="emoji angry">😡</div>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .description-text {
            font-family: 'Poppins', sans-serif;
            font-size: 18px;
            color: #55a8ff;
            line-height: 1.6;
            margin-left: 10px;
            margin-top: -30px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        div[data-testid="stTextInput"] {
            margin-left: 10px;
            width: 95%;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 12px;
            padding: 12px 16px;
            border: none;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .description-text5 {
            font-family: 'Poppins', sans-serif;
            font-size: 18px;
            line-height: 1.6;
            margin-left: 10px;
            margin-top: -5px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        div[data-testid="stTextInput"] {
            margin-left: 10px;
            width: 95%;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 12px;
            padding: 12px 16px;
            border: none;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .description-text1 {
            font-family: 'Poppins', sans-serif;
            font-size: 36px;
            color: #ff4040;
            line-height: 1.6;
            margin-left: 10px;
            margin-top: -30px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        div[data-testid="stTextInput"] {
            margin-left: 10px;
            width: 95%;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 12px;
            padding: 12px 16px;
            border: none;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .description-text2 {
            font-family: 'Poppins', sans-serif;
            font-size: 36px;
            color: #ffbf51;
            line-height: 1.6;
            margin-left: 10px;
            margin-top: -30px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        div[data-testid="stTextInput"] {
            margin-left: 10px;
            width: 95%;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 12px;
            padding: 12px 16px;
            border: none;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .description-text3 {
            font-family: 'Poppins', sans-serif;
            font-size: 36px;
            color: #6dc846;
            line-height: 1.6;
            margin-left: 10px;
            margin-top: -39px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        div[data-testid="stTextInput"] {
            margin-left: 10px;
            width: 95%;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 12px;
            padding: 12px 16px;
            border: none;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

data = pd.read_csv(r"labeled_data.csv")
#try:
    #nlp = spacy.load("en_core_web_sm")
#except OSError:
    #subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    #nlp = spacy.load("en_core_web_sm")


def preprocess(text):
  #doc = nlp(text)
  text = re.sub(r"[^\w\s\d\']","",text)
  text = re.sub(r"[\d]+","",text)
  text = re.sub(r" +"," ",text)
  return text.lower().strip()
data["tweet"] = data["tweet"].fillna("").map(preprocess)
#def pre2(text):
  #doc1 = nlp(text)
  #return " ".join([i.lemma_ for i in doc1 if not i.is_stop or not i.is_punct])
#data["tweet"] = data["tweet"].apply(pre2)
x = data["tweet"]
y = data["class"]
v = TfidfVectorizer()
x_vector = v.fit_transform(x)

max_size = 19190
sm = SMOTE(sampling_strategy={0:max_size,1:max_size,2:max_size}, random_state=42)
x_smote,y_smote=sm.fit_resample(x_vector,y)

x_train,x_test,y_train,y_test = train_test_split(x_smote,y_smote,test_size=0.2,random_state=40)
#x_train_arr = x_train.toarray()
#x_test_arr = x_test.toarray()

m = MultinomialNB()
m.fit(x_train,y_train)
#y_pred = m.predict(x_test_arr)

def fast_prediction(text):
    text = re.sub(r"[^\w\s\d\']", "", text)
    text = re.sub(r"[\d]+", "", text)
    text = re.sub(r" +", " ", text)
    text = text.lower().strip()
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return m.predict(v.transform([text]))[0]


col1,col2 = st.columns([0.8,1])
with col1:
    st.markdown(
        "<div class='description-text'>Enter any chats from social media platforms. "
        "They will be analysed whether the chats are offensive, appropriate or hate speech.</div>",
        unsafe_allow_html=True
    )
    t1 = st.text_input("")
    #out = prediction(t1).item(0)
with col2:
    if t1 == "": 
      st.markdown(
        "<div class='description-text5'>PLEASE ENTER TEXT.</div>",
        unsafe_allow_html=True
        )
    else:
      out = fast_prediction(t1).item(0)
      if out == 0:
         st.markdown(
        "<div class='description-text1'>HATE SPEECH</div> ",
        unsafe_allow_html=True
        )
         st.markdown(
        "<div class='description-text5'>These speeches are unacceptable. "
        "They should not be used in social media platform, that brings hate and discomfort.</div>",
        unsafe_allow_html=True
        )
      elif out == 1:
         st.markdown(
        "<div class='description-text2'>OFFENSIVE SPEECH</div> ",
        unsafe_allow_html=True
        )
         st.markdown(
        "<div class='description-text5'>These speeches should be used minimum. "
        "Try to have healthy conversation in social media community.</div>",
        unsafe_allow_html=True
        )
      else:
         st.markdown(
        "<div class='description-text3'>APPROPRIATE SPEECH</div> ",
        unsafe_allow_html=True
        )
         st.markdown(
        "<div class='description-text5'>These speeches are appropriate and should be used. "
        "This brings healthy interaction among the people in the platforms.</div>",
        unsafe_allow_html=True
        )
    
    
