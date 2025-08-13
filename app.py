import pandas as pd
import streamlit as st
import joblib
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import os
st.set_page_config(
    initial_sidebar_state='expanded',
    page_title='Prédicteur Spam Email - Youssouf',
    page_icon='📧',
    layout='wide'
)

# source
MODEL_URL = 'https://github.com/Youssouf-Mohamed-Nouh/SpamEmail/releases/download/v1.0/naivebayes.pkl'
MODEL_PATH = 'naivebayes.pkl'

@st.cache_resource
def charger_model():
    try:
        # Télécharger le modèle s'il n'existe pas encore localement
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Téléchargement du modèle..."):
                response = requests.get(MODEL_URL)
                response.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)

        # Charger le modèle
        model = joblib.load(MODEL_PATH)
        # Extraire les features du pipeline
        features = model.named_steps['tfidf'].get_feature_names_out()
        return model, features

    except FileNotFoundError as e:
        st.error(f'Erreur: Fichier manquant - {e}')
        st.stop()
    except Exception as e:
        st.error(f'Erreur lors du chargement : {e}')
        st.stop()

model, features = charger_model()

# En-tête
st.markdown('''
            <style>
            .main-header{
               background: linear-gradient(135deg, #91BDF2 0%, #91BDF2 100%);
               padding:2.2rem;
               border-radius:50px;
               margin-bottom:2rem;
               text-align:center;
               box-shadow: 0 20px 50px rgba(0,0,0,0.1);
            }
            </style>
            ''', unsafe_allow_html=True)
st.markdown('''
            <div class='main-header'>
            <h1>📧 Prédicteur Spam Email</h1>
            <p style='font-size:20px;'>Développé par - <strong>Youssouf</strong> Assistant Intelligent</p>
            </div>
            ''', unsafe_allow_html=True)

# Sidebar
st.markdown('''
            <style>
            .friendly-info {
                background: #e3f2fd;
                padding: 2rem;
                border-radius: 15px;
                border-left: 5px solid #2196F3;
                margin: 1.5rem 0;
            }
            .encouragement {
                background: linear-gradient(135deg, #fff3e0, #ffecb3);
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                border-left: 5px solid #ff9800;
            }
            </style>
            ''', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 🤖 À propos de votre assistant")
    st.markdown("""
    <div class="friendly-info">
        <h4>Comment je fonctionne ?</h4>
        <p>• J'utilise un modèle Naive Bayes entraîné sur des milliers d'emails</p>
        <p>• Ma précision est d'environ 99%</p>
        <p>• Je respecte votre vie privée</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💡 Rappel important")
    st.markdown("""
    <div class="encouragement">
        <p><strong>Gardez en tête :</strong></p>
        <p>✨ Je suis un outil d'aide, pas un filtre anti-spam parfait</p>
    </div>
    """, unsafe_allow_html=True)

# Formulaire
st.markdown('''
            <h2 style='color:#343a40;text-align:center;margin-bottom:25px'> 📋 Texte de l'email à analyser</h2>
            ''' , unsafe_allow_html=True)

with st.form(key='formulaire_email'):
    email_text = st.text_area("Copiez/collez votre email ici :", height=200)
    submitted = st.form_submit_button("Analyser")

# --- Prédiction ---
if submitted:
    if not email_text.strip():
        st.warning("Merci de saisir un texte d'email valide.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                # Transformer le texte avec le pipeline et prédire
                prediction = model.predict([email_text])[0]
                proba = model.predict_proba([email_text])[0]
                label = 'Spam' if prediction == 1 else 'Ham (Non spam)'
                conf_score = proba[prediction]

                labels = ['Ham (Non spam)', 'Spam']
                st.success(f"🛡️ Classification : **{label}** (Confiance: {conf_score:.2%})")
                st.write(f"Probabilités détaillées : Ham={proba[0]:.2%} | Spam={proba[1]:.2%}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

# Message de conclusion plus chaleureux
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">📧 Votre Assistant Spam Email</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous accompagner dans la détection de spam
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2025 - Mis à jour régulièrement pour améliorer la précision
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil d'aide complète mais ne remplace jamais une analyse humaine approfondie
        </p>
    </div>
</div>
""", unsafe_allow_html=True)



