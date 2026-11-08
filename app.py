# -*- coding: utf-8 -*-
"""
DDoS Shield — Détection d'attaques DDoS avec Machine Learning
Interface Streamlit basée sur le pipeline de l'atelier (dataset de logs réseau).

Lancement :  streamlit run app.py
Identifiants : voir la fonction verifier_identifiants()

NB : les données sont synthétiques (pédagogiques). Tous les chiffres affichés
sont CALCULÉS par l'application à partir du dataset : rien n'est écrit en dur.
"""

import hashlib
import hmac
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# 1. CONFIGURATION GÉNÉRALE
# --------------------------------------------------------------------------
st.set_page_config(page_title="DDoS Shield", layout="wide")

DATASET_PAR_DEFAUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

# Colonnes numériques utilisées par les modèles (comme dans l'atelier)
COLONNES_NUM = ["duration_ms", "packet_count", "byte_count", "packets_per_second",
                "bytes_per_second", "avg_packet_size_bytes", "syn_count",
                "ack_count", "connections_10s", "unique_sources_10s"]
COLONNES_CAT = ["protocol", "dst_port"]
# Colonnes exclues (identifiants) : elles favoriseraient des raccourcis
COLONNES_EXCLUES = ["timestamp", "src_ip", "dst_ip", "src_port"]
COLONNES_REQUISES = COLONNES_NUM + COLONNES_CAT + ["label"]

# Modèles qui n'ont PAS besoin de standardisation
MODELES_ARBRES = ["Random Forest", "XGBoost"]

# Libellés français pour le formulaire de prédiction
LIBELLES = {
    "duration_ms": "Durée du flux (ms)",
    "packet_count": "Nombre de paquets",
    "byte_count": "Nombre d'octets",
    "packets_per_second": "Paquets / seconde",
    "bytes_per_second": "Octets / seconde",
    "avg_packet_size_bytes": "Taille moyenne d'un paquet (octets)",
    "syn_count": "Paquets SYN",
    "ack_count": "Paquets ACK",
    "connections_10s": "Connexions (fenêtre 10 s)",
    "unique_sources_10s": "Sources uniques (fenêtre 10 s)",
}

# --------------------------------------------------------------------------
# 2. STYLE (CSS) ET LOGO
# --------------------------------------------------------------------------
CSS = """
<style>
/* ---- Thème sombre : fond noir ---- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background-color: #000000 !important;}
[data-testid="stSidebar"] {background-color: #0a0d12 !important; border-right: 1px solid #1c2530;}
.stApp, .stApp p, .stApp label, .stApp li, .stApp span, .stApp h1, .stApp h2, .stApp h3 {color: #e6edf3;}
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div {background-color: #0d1117 !important; color: #e6edf3 !important;}
.bandeau {background: linear-gradient(120deg,#05080d 0%,#0b2a4a 70%,#12437a 100%); border: 1px solid #1c2530;
          padding: 1.2rem 1.6rem; border-radius: 12px; color: #fff;
          display: flex; align-items: center; gap: 1.2rem; margin-bottom: 1.2rem;}
.bandeau h1 {margin: 0; font-size: 1.7rem; color: #fff;}
.bandeau p {margin: .2rem 0 0 0; color: #cfe3f5; font-size: .95rem;}
.carte {background: #0d1117; border: 1px solid #1c2530;
        border-radius: 10px; padding: .9rem 1.1rem; text-align: center;}
.carte .val {font-size: 1.8rem; font-weight: 700; color: #1b7fbf;}
.carte .nom {font-size: .85rem; color: #8b98a5;}
.verdict-ok  {background:#0f5132; color:#fff; padding:1rem 1.2rem; border-radius:12px; font-size:1.2rem;}
.verdict-att {background:#842029; color:#fff; padding:1rem 1.2rem; border-radius:12px; font-size:1.2rem;}
div.stButton > button {border-radius: 10px; font-weight: 600;}
</style>
"""

LOGO_SVG = """
<svg width="SIZE" height="SIZE" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#35c2ff"/><stop offset="1" stop-color="#0a5ea8"/>
    </linearGradient>
  </defs>
  <path d="M50 5 L88 20 V50 C88 74 70 90 50 96 C30 90 12 74 12 50 V20 Z"
        fill="url(#g)" stroke="#ffffff" stroke-width="3"/>
  <g stroke="#ffffff" stroke-width="2.6" stroke-linecap="round" fill="none">
    <line x1="50" y1="30" x2="32" y2="52"/><line x1="50" y1="30" x2="68" y2="52"/>
    <line x1="32" y1="52" x2="50" y2="72"/><line x1="68" y1="52" x2="50" y2="72"/>
    <line x1="32" y1="52" x2="68" y2="52"/>
  </g>
  <g fill="#ffffff">
    <circle cx="50" cy="30" r="5.5"/><circle cx="32" cy="52" r="5"/>
    <circle cx="68" cy="52" r="5"/><circle cx="50" cy="72" r="5.5"/>
  </g>
</svg>
"""


def logo(taille=64):
    """Retourne le logo (SVG) à la taille voulue."""
    return LOGO_SVG.replace("SIZE", str(taille))


def bandeau(titre, sous_titre):
    """Affiche l'en-tête de page avec le logo."""
    st.markdown(
        f'<div class="bandeau">{logo(58)}<div><h1>{titre}</h1><p>{sous_titre}</p></div></div>',
        unsafe_allow_html=True)


def carte(nom, valeur):
    """Petite carte-métrique stylée."""
    return f'<div class="carte"><div class="val">{valeur}</div><div class="nom">{nom}</div></div>'


st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 3. AUTHENTIFICATION
# --------------------------------------------------------------------------
# On compare des empreintes SHA-256 : le mot de passe n'apparaît pas en clair.
LOGIN = "ihssane"
HASH_MDP = hashlib.sha256("cyberihssane@123".encode()).hexdigest()


def verifier_identifiants(login, mdp):
    """Vérifie login + mot de passe (comparaison à temps constant)."""
    h = hashlib.sha256(mdp.encode()).hexdigest()
    return hmac.compare_digest(login, LOGIN) and hmac.compare_digest(h, HASH_MDP)


def page_login():
    """Écran de connexion."""
    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.markdown(f'<div style="text-align:center">{logo(110)}</div>', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center'>DDoS Shield</h2>", unsafe_allow_html=True)
        st.caption("Détection d'attaques DDoS par Machine Learning — connexion requise")
        with st.form("login"):
            login = st.text_input("Identifiant")
            mdp = st.text_input("Mot de passe", type="password")
            ok = st.form_submit_button("Se connecter", width="stretch")
        if ok:
            if verifier_identifiants(login, mdp):
                st.session_state["connecte"] = True
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")


# --------------------------------------------------------------------------
# 4. FONCTIONS DU PIPELINE ML (reprend l'atelier)
# --------------------------------------------------------------------------
def charger_csv(source):
    """Lit un CSV (chemin ou fichier envoyé) et vérifie les colonnes."""
    df = pd.read_csv(source)
    manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")
    return df


def pretraiter(df):
    """Doublons -> NaN remplacés par 0 -> X / y -> encodage one-hot."""
    nb_doublons = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    nb_nan = int(df.isna().sum().sum())
    df = df.fillna(0)                      # choix de l'atelier (à justifier en réel)
    y = df["label"]
    X = df.drop(columns=["label"] + [c for c in COLONNES_EXCLUES if c in df.columns])
    X = pd.get_dummies(X, columns=COLONNES_CAT, dtype=int)
    return {"X": X, "y": y, "nb_doublons": nb_doublons, "nb_nan": nb_nan, "nb_lignes": len(df)}


def metriques(y_vrai, y_pred):
    """Calcule les 4 métriques demandées."""
    return {"Accuracy": accuracy_score(y_vrai, y_pred),
            "Précision": precision_score(y_vrai, y_pred, zero_division=0),
            "Recall": recall_score(y_vrai, y_pred, zero_division=0),
            "F1-score": f1_score(y_vrai, y_pred, zero_division=0)}


def predire_classes(nom, modele, scaler, X):
    """Prédit 0/1 avec le bon format de données selon le modèle."""
    if nom in MODELES_ARBRES:
        sortie = modele.predict(X)
    else:
        sortie = modele.predict(scaler.transform(X))
    if nom == "Régression linéaire":       # sortie continue -> seuil 0,5
        sortie = (sortie >= 0.5).astype(int)
    return sortie


def entrainer(prep, graine=42):
    """Split 70/15/15, standardisation, entraînement des 5 modèles."""
    X, y = prep["X"], prep["y"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=graine, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                random_state=graine, stratify=y_tmp)
    scaler = StandardScaler().fit(X_tr)    # ajusté sur TRAIN uniquement (pas de fuite)
    X_tr_s = scaler.transform(X_tr)

    modeles = {
        "Régression linéaire": LinearRegression(),
        "Régression logistique": LogisticRegression(max_iter=1000, random_state=graine),
        "SVM": SVC(kernel="rbf", C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=graine, n_jobs=2),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", random_state=graine, n_jobs=2),
    }
    lignes = []
    for nom, m in modeles.items():
        # Les arbres utilisent les données brutes, les autres les données standardisées
        m.fit(X_tr if nom in MODELES_ARBRES else X_tr_s, y_tr)
        pred_val = predire_classes(nom, m, scaler, X_val)
        lignes.append({"Modèle": nom, **metriques(y_val, pred_val)})
    tableau = pd.DataFrame(lignes).sort_values("F1-score", ascending=False).reset_index(drop=True)

    # Évaluation finale sur TEST : une seule fois, pour le meilleur modèle (F1 validation)
    meilleur = tableau.loc[0, "Modèle"]
    pred_test = predire_classes(meilleur, modeles[meilleur], scaler, X_te)
    return {"modeles": modeles, "scaler": scaler, "tableau": tableau, "meilleur": meilleur,
            "y_test": y_te, "pred_test": pred_test, "X_test": X_te,
            "tailles": (len(X_tr), len(X_val), len(X_te)), "colonnes": list(X.columns)}


def aligner_colonnes(df_brut, colonnes):
    """Transforme des flux bruts en tableau compatible avec les modèles entraînés."""
    X = df_brut.drop(columns=[c for c in COLONNES_EXCLUES + ["label"] if c in df_brut.columns])
    X = pd.get_dummies(X.fillna(0), columns=COLONNES_CAT, dtype=int)
    return X.reindex(columns=colonnes, fill_value=0)


# --------------------------------------------------------------------------
# 5. PAGES DE L'APPLICATION
# --------------------------------------------------------------------------
def page_accueil():
    bandeau("DDoS Shield", "Détection et prédiction d'attaques DDoS par Machine Learning")
    st.markdown("""
Cette application suit le pipeline de l'atelier, étape par étape :

1. **Données** : chargement du CSV (20 000 flux réseau étiquetés normal / DDoS) ;
2. **Prétraitement** : doublons, valeurs manquantes, encodage ;
3. **Entraînement** : régression linéaire, régression logistique, SVM, Random Forest, XGBoost ;
4. **Résultats** : Accuracy, Précision, Recall, F1-score et matrice de confusion ;
5. **Prédiction** : classer un flux saisi à la main ou un fichier CSV complet.

Utilisez le menu à gauche et suivez les pages dans l'ordre.
""")
    st.info("Le jeu de données est **synthétique** (usage pédagogique). Les scores obtenus "
            "ne prouvent pas l'efficacité des modèles sur un vrai réseau.")


def page_donnees():
    bandeau("Chargement des données", "Importer et explorer le dataset de flux réseau")
    fichier = st.file_uploader("Charger un autre CSV (optionnel)", type="csv")
    if fichier is not None:
        source, nom = fichier, fichier.name
    else:
        source, nom = DATASET_PAR_DEFAUT, "dataset.csv (fourni avec l'application)"
    st.caption(f"Source utilisée : {nom}")

    if st.button("Charger le dataset", type="primary"):
        try:
            st.session_state["df"] = charger_csv(source)
            # Un nouveau dataset invalide les étapes suivantes
            for cle in ("prep", "res"):
                st.session_state.pop(cle, None)
        except Exception as e:
            st.error(f"Impossible de charger le fichier : {e}")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Aucun dataset chargé pour le moment.")
        return

    c = st.columns(4)
    c[0].markdown(carte("Lignes", f"{len(df):,}".replace(",", " ")), unsafe_allow_html=True)
    c[1].markdown(carte("Colonnes", df.shape[1]), unsafe_allow_html=True)
    c[2].markdown(carte("Flux normaux (0)", int((df["label"] == 0).sum())), unsafe_allow_html=True)
    c[3].markdown(carte("Flux DDoS (1)", int((df["label"] == 1).sum())), unsafe_allow_html=True)

    st.subheader("Aperçu")
    st.dataframe(df.head(10), width="stretch")
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Répartition des classes")
        st.bar_chart(df["label"].map({0: "Normal", 1: "DDoS"}).value_counts())
    with g2:
        st.subheader("Valeurs manquantes par colonne")
        st.dataframe(df.isna().sum().rename("NaN"), width="stretch")
    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe().T, width="stretch")


def page_pretraitement():
    bandeau("Prétraitement", "Nettoyage et préparation des variables")
    df = st.session_state.get("df")
    if df is None:
        st.warning("Chargez d'abord le dataset (page « Données »).")
        return
    st.markdown("""
- suppression des **doublons** (avant le découpage train/test) ;
- **valeurs manquantes** remplacées par 0 (choix de l'atelier ; en situation réelle, cela peut créer un signal artificiel) ;
- exclusion de `timestamp`, `src_ip`, `dst_ip`, `src_port` ;
- encodage *one-hot* de `protocol` et `dst_port`.
""")
    if st.button("Lancer le prétraitement", type="primary"):
        st.session_state["prep"] = pretraiter(df)
        st.session_state.pop("res", None)   # les anciens modèles ne sont plus valables

    prep = st.session_state.get("prep")
    if prep is None:
        return
    c = st.columns(4)
    c[0].markdown(carte("Doublons supprimés", prep["nb_doublons"]), unsafe_allow_html=True)
    c[1].markdown(carte("NaN remplacés par 0", prep["nb_nan"]), unsafe_allow_html=True)
    c[2].markdown(carte("Lignes conservées", prep["nb_lignes"]), unsafe_allow_html=True)
    c[3].markdown(carte("Variables (X)", prep["X"].shape[1]), unsafe_allow_html=True)
    st.subheader("Aperçu de X après encodage")
    st.dataframe(prep["X"].head(5), width="stretch")


def page_entrainement():
    bandeau("Entraînement", "Comparer cinq algorithmes de Machine Learning")
    prep = st.session_state.get("prep")
    if prep is None:
        st.warning("Lancez d'abord le prétraitement.")
        return
    st.markdown("Découpage **70 % entraînement / 15 % validation / 15 % test** (stratifié, "
                "`random_state=42`). Le scaler est ajusté sur l'entraînement uniquement.")
    if st.button("Entraîner les modèles", type="primary"):
        with st.spinner("Entraînement en cours (le SVM peut prendre un moment)..."):
            st.session_state["res"] = entrainer(prep)
        st.success("Entraînement terminé : consultez la page « Résultats ».")
    res = st.session_state.get("res")
    if res:
        n_tr, n_val, n_te = res["tailles"]
        c = st.columns(3)
        c[0].markdown(carte("Train", n_tr), unsafe_allow_html=True)
        c[1].markdown(carte("Validation", n_val), unsafe_allow_html=True)
        c[2].markdown(carte("Test", n_te), unsafe_allow_html=True)


def page_resultats():
    bandeau("Résultats", "Métriques de validation et évaluation finale sur le test")
    res = st.session_state.get("res")
    if res is None:
        st.warning("Entraînez d'abord les modèles.")
        return

    st.subheader("Comparaison sur le jeu de validation")
    st.dataframe(res["tableau"].round(3), width="stretch", hide_index=True)
    st.bar_chart(res["tableau"].set_index("Modèle")["F1-score"])
    st.caption("Le meilleur modèle est choisi selon le F1-score de validation.")

    st.subheader(f"Évaluation finale sur le test : {res['meilleur']}")
    m = metriques(res["y_test"], res["pred_test"])
    c = st.columns(4)
    for col, (nom, val) in zip(c, m.items()):
        col.markdown(carte(nom, f"{val:.3f}"), unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(res["y_test"], res["pred_test"],
                                                display_labels=["Normal", "DDoS"],
                                                cmap="Blues", ax=ax)
        ax.set_title("Matrice de confusion (test)")
        # Style sombre pour s'intégrer au fond noir
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#000000")
        for el in (ax.title, ax.xaxis.label, ax.yaxis.label):
            el.set_color("#e6edf3")
        ax.tick_params(colors="#e6edf3")
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        st.text("Rapport de classification (test)")
        st.code(classification_report(res["y_test"], res["pred_test"],
                                      target_names=["Normal", "DDoS"]))
        st.caption("Faux négatifs = attaques non détectées ; faux positifs = trafic normal signalé à tort.")


def page_prediction():
    bandeau("Prédiction", "Classer un flux réseau : normal ou attaque DDoS ?")
    df = st.session_state.get("df")
    res = st.session_state.get("res")
    if res is None or df is None:
        st.warning("Entraînez d'abord les modèles.")
        return

    noms = list(res["modeles"].keys())
    choix = st.selectbox("Modèle utilisé", noms, index=noms.index(res["meilleur"]),
                         help="Par défaut : le meilleur modèle selon le F1-score de validation.")
    modele = res["modeles"][choix]

    onglet1, onglet2 = st.tabs(["Saisie manuelle", "Fichier CSV"])

    # ---- Saisie manuelle : valeurs initiales = médianes du dataset chargé ----
    with onglet1:
        med = df[COLONNES_NUM].median()
        with st.form("form_flux"):
            c1, c2 = st.columns(2)
            protocole = c1.selectbox("Protocole", sorted(df["protocol"].unique()))
            port = c2.selectbox("Port de destination", sorted(df["dst_port"].unique()))
            valeurs = {}
            colonnes = st.columns(2)
            for i, nom in enumerate(COLONNES_NUM):
                valeurs[nom] = colonnes[i % 2].number_input(
                    LIBELLES[nom], min_value=0.0, value=float(med[nom]))
            lancer = st.form_submit_button("Analyser ce flux", width="stretch")
        if lancer:
            flux = pd.DataFrame([{**valeurs, "protocol": protocole, "dst_port": port}])
            X_new = aligner_colonnes(flux, res["colonnes"])
            classe = int(predire_classes(choix, modele, res["scaler"], X_new)[0])
            if classe == 1:
                st.markdown('<div class="verdict-att">Flux classé : <b>ATTAQUE DDoS</b></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-ok">Flux classé : <b>TRAFIC NORMAL</b></div>',
                            unsafe_allow_html=True)
            # Probabilité disponible seulement pour certains modèles
            if hasattr(modele, "predict_proba"):
                X_p = X_new if choix in MODELES_ARBRES else res["scaler"].transform(X_new)
                proba = modele.predict_proba(X_p)[0][1]
                st.write(f"Probabilité estimée d'attaque par le modèle : **{proba:.1%}**")
            else:
                st.caption("Ce modèle ne fournit pas de probabilité.")

    # ---- Prédiction par lot sur un CSV ----
    with onglet2:
        st.markdown("Envoyez un CSV avec les colonnes : `protocol`, `dst_port`, "
                    + ", ".join(f"`{c}`" for c in COLONNES_NUM) + ".")
        fichier = st.file_uploader("CSV de flux à analyser", type="csv", key="batch")
        if fichier is not None:
            nouveau = pd.read_csv(fichier)
            absentes = [c for c in COLONNES_NUM + COLONNES_CAT if c not in nouveau.columns]
            if absentes:
                st.error(f"Colonnes manquantes : {absentes}")
            else:
                X_new = aligner_colonnes(nouveau, res["colonnes"])
                nouveau["prediction"] = np.where(
                    predire_classes(choix, modele, res["scaler"], X_new) == 1, "DDoS", "Normal")
                st.write(nouveau["prediction"].value_counts().rename("Nombre de flux"))
                st.dataframe(nouveau.head(50), width="stretch")
                st.download_button("Télécharger les prédictions",
                                   nouveau.to_csv(index=False).encode("utf-8"),
                                   "predictions_ddos.csv", "text/csv")


# --------------------------------------------------------------------------
# 6. NAVIGATION PRINCIPALE
# --------------------------------------------------------------------------
PAGES = {
    "Accueil": page_accueil,
    "Données": page_donnees,
    "Prétraitement": page_pretraitement,
    "Entraînement": page_entrainement,
    "Résultats": page_resultats,
    "Prédiction": page_prediction,
}

if not st.session_state.get("connecte"):
    page_login()
else:
    with st.sidebar:
        st.markdown(f'<div style="text-align:center">{logo(90)}</div>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center'>DDoS Shield</h3>", unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        if st.button("Se déconnecter", width="stretch"):
            st.session_state.clear()
            st.rerun()
    PAGES[page]()
