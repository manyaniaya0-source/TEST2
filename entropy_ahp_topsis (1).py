import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Entropy-AHP-TOPSIS", layout="wide")
st.title("🔍 Modèle de Décision Multi-Critères : Entropy-AHP-TOPSIS")
st.markdown("**Sélection de fournisseurs de matériaux de construction**")

# ==================== SECTION 1: MATRICE DE DÉCISION ====================
st.header("📊 Étape 1 : Matrice de Décision")

col1, col2 = st.columns(2)
with col1:
    n_alternatives = st.number_input("Nombre d'alternatives (lignes)", min_value=2, max_value=20, value=4)
with col2:
    n_criteres = st.number_input("Nombre de critères (colonnes)", min_value=2, max_value=15, value=7)

# Noms des alternatives et critères
alternatives = [st.text_input(f"Alternative {i+1}", value=f"Fournisseur {i+1}", key=f"alt_{i}") 
                for i in range(n_alternatives)]

# Critères par défaut selon la structure hiérarchique
criteres_defaut = [
    "C1: Qualified products (%)",
    "C2: Product price ($1000)",
    "C3: Market share (%)",
    "C4: Supply capacity (kg/time)",
    "C5: New product development (%)",
    "C6: Delivery time (days)",
    "C7: Delivery on time ratio (%)"
]

criteres = []
for j in range(n_criteres):
    if j < len(criteres_defaut):
        criteres.append(st.text_input(f"Critère {j+1}", value=criteres_defaut[j], key=f"crit_{j}"))
    else:
        criteres.append(st.text_input(f"Critère {j+1}", value=f"Critère {j+1}", key=f"crit_{j}"))

# Type de critères (maximiser ou minimiser)
st.subheader("🎯 Type de critères")
type_defaut = [1, -1, 1, 1, 1, -1, 1]  # Max, Min, Max, Max, Max, Min, Max
type_criteres = []
cols = st.columns(min(n_criteres, 4))
for j in range(n_criteres):
    with cols[j % 4]:
        default_type = "Maximiser" if (j < len(type_defaut) and type_defaut[j] == 1) else "Minimiser"
        type_crit = st.selectbox(f"{criteres[j][:20]}...", ["Maximiser", "Minimiser"], 
                                 index=0 if default_type == "Maximiser" else 1, key=f"type_{j}")
        type_criteres.append(1 if type_crit == "Maximiser" else -1)

# Saisie de la matrice de décision
st.subheader("📝 Saisie des données")
matrice_decision = np.zeros((n_alternatives, n_criteres))
df_input = pd.DataFrame(matrice_decision, index=alternatives, columns=criteres)
matrice_decision_df = st.data_editor(df_input, use_container_width=True)
matrice_decision = matrice_decision_df.values

# ==================== SECTION 2: AHP HIÉRARCHIQUE ====================
st.set_page_config(page_title="Calcul AHP", layout="wide")

st.title("🌟 Application AHP - Pondération des critères")

st.markdown("""
Cette application permet de **calculer les poids AHP** à partir d'une **matrice de comparaison personnalisée**.
Entrez vos valeurs (entre 1 et 9, ou leur inverse 1/x) dans la matrice ci-dessous.
""")

# ----------------------------
# 1️⃣ Saisie du nombre de critères
# ----------------------------
n = st.number_input("Nombre de critères :", min_value=2, max_value=10, value=3, step=1)

# Noms des critères
criteres = [f"C{i+1}" for i in range(n)]

st.subheader("🧩 Matrice de comparaison par paires")

# Création d’une matrice identité par défaut
matrice = np.ones((n, n))

# Entrée des valeurs par l'utilisateur
for i in range(n):
    for j in range(i + 1, n):
        val = st.number_input(
            f"Importance de {criteres[i]} par rapport à {criteres[j]}",
            min_value=0.111, max_value=9.0, value=1.0, step=0.1,
            key=f"m_{i}_{j}"
        )
        matrice[i, j] = val
        matrice[j, i] = round(1 / val, 4)

# Afficher la matrice
st.write("### Matrice de comparaison :")
df_matrice = pd.DataFrame(matrice, columns=criteres, index=criteres)
st.dataframe(df_matrice, use_container_width=True)

# ----------------------------
# 2️⃣ Calcul des poids AHP
# ----------------------------
if st.button("Calculer les poids AHP"):
    # Normalisation des colonnes
    col_sums = matrice.sum(axis=0)
    matrice_norm = matrice / col_sums

    # Calcul du vecteur des poids (moyenne des lignes)
    poids = matrice_norm.mean(axis=1)
    poids_norm = poids / poids.sum()

    # Résultats
    df_result = pd.DataFrame({
        "Critère": criteres,
        "Poids AHP": np.round(poids_norm, 4)
    })

    st.success("✅ Calcul terminé avec succès !")
    st.write("### Résultats des poids AHP :")
    st.dataframe(df_result, use_container_width=True)
# ==================== SECTION 3: CALCULS ====================
if st.button("🚀 Calculer les résultats", type="primary"):
    
    # ÉTAPE 2: Normalisation de la matrice de décision
    st.header("📐 Étape 3 : Normalisation de la matrice")
    somme_carres = np.sqrt((matrice_decision ** 2).sum(axis=0))
    somme_carres = np.where(somme_carres == 0, 1, somme_carres)
    matrice_norm = matrice_decision / somme_carres
    
    df_norm = pd.DataFrame(matrice_norm, index=alternatives, columns=criteres)
    st.dataframe(df_norm.style.format("{:.4f}"), use_container_width=True)
    
    # ÉTAPE 3: Calcul des poids Entropy (objectifs)
    st.header("🔬 Étape 4 : Calcul des poids Entropy (objectifs)")
    
    m = n_alternatives
    k = 1 / np.log(m)
    
    # Calcul de z_ij
    somme_p = matrice_norm.sum(axis=0)
    somme_p = np.where(somme_p == 0, 1, somme_p)
    z_ij = matrice_norm / somme_p
    
    # Éviter log(0)
    z_ij_safe = np.where(z_ij > 0, z_ij, 1e-10)
    
    # Calcul de l'entropie
    entropie = -k * (z_ij_safe * np.log(z_ij_safe)).sum(axis=0)
    
    # Calcul des poids objectifs
    somme_entropie = (1 - entropie).sum()
    if somme_entropie == 0:
        poids_entropy = np.ones(n_criteres) / n_criteres
    else:
        poids_entropy = (1 - entropie) / somme_entropie
    
    df_entropy = pd.DataFrame({
        'Critère': criteres,
        'Entropie (e_j)': entropie,
        'Poids Entropy (w_e)': poids_entropy
    })
    st.dataframe(df_entropy.style.format({
        'Entropie (e_j)': '{:.4f}',
        'Poids Entropy (w_e)': '{:.4f}'
    }), use_container_width=True)
    
    # ÉTAPE 5: Combinaison des poids
    st.header("🔗 Étape 5 : Combinaison des poids Entropy-AHP")
    
    produit_poids = poids_entropy * poids_ahp
    somme_produit = produit_poids.sum()
    if somme_produit == 0:
        poids_combines = np.ones(n_criteres) / n_criteres
    else:
        poids_combines = produit_poids / somme_produit
    
    df_poids_final = pd.DataFrame({
        'Critère': criteres,
        'Poids Entropy (w_e)': poids_entropy,
        'Poids AHP (w_h)': poids_ahp,
        'Poids Combiné (w_c)': poids_combines
    })
    st.dataframe(df_poids_final.style.format({
        'Poids Entropy (w_e)': '{:.4f}',
        'Poids AHP (w_h)': '{:.4f}',
        'Poids Combiné (w_c)': '{:.4f}'
    }), use_container_width=True)
    
    # ÉTAPE 6: Matrice pondérée normalisée
    st.header("⚡ Étape 6 : Matrice pondérée normalisée")
    
    matrice_ponderee = matrice_norm * poids_combines
    
    df_ponderee = pd.DataFrame(matrice_ponderee, index=alternatives, columns=criteres)
    st.dataframe(df_ponderee.style.format("{:.4f}"), use_container_width=True)
    
    # ÉTAPE 7: Solutions idéales
    st.header("🎯 Étape 7 : Solutions idéales")
    
    solution_ideale_pos = np.zeros(n_criteres)
    solution_ideale_neg = np.zeros(n_criteres)
    
    for j in range(n_criteres):
        if type_criteres[j] == 1:  # Maximiser
            solution_ideale_pos[j] = matrice_ponderee[:, j].max()
            solution_ideale_neg[j] = matrice_ponderee[:, j].min()
        else:  # Minimiser
            solution_ideale_pos[j] = matrice_ponderee[:, j].min()
            solution_ideale_neg[j] = matrice_ponderee[:, j].max()
    
    df_ideales = pd.DataFrame({
        'Critère': criteres,
        'Type': ['Max' if t == 1 else 'Min' for t in type_criteres],
        'A⁺ (Idéale positive)': solution_ideale_pos,
        'A⁻ (Idéale négative)': solution_ideale_neg
    })
    st.dataframe(df_ideales.style.format({
        'A⁺ (Idéale positive)': '{:.4f}',
        'A⁻ (Idéale négative)': '{:.4f}'
    }), use_container_width=True)
    
    # ÉTAPE 8: Calcul des distances
    st.header("📏 Étape 8 : Calcul des distances")
    
    distances_pos = np.sqrt(((matrice_ponderee - solution_ideale_pos) ** 2).sum(axis=1))
    distances_neg = np.sqrt(((matrice_ponderee - solution_ideale_neg) ** 2).sum(axis=1))
    
    df_distances = pd.DataFrame({
        'Alternative': alternatives,
        'S⁺ (Distance à A⁺)': distances_pos,
        'S⁻ (Distance à A⁻)': distances_neg
    })
    st.dataframe(df_distances.style.format({
        'S⁺ (Distance à A⁺)': '{:.4f}',
        'S⁻ (Distance à A⁻)': '{:.4f}'
    }), use_container_width=True)
    
    # ÉTAPE 9: Proximité relative
    st.header("🏆 Étape 9 : Proximité relative (Score TOPSIS)")
    
    somme_distances = distances_pos + distances_neg
    somme_distances = np.where(somme_distances == 0, 1, somme_distances)
    proximite_relative = distances_neg / somme_distances
    
    # ÉTAPE 10: Classement final
    st.header("🥇 Étape 10 : Classement final")
    
    classement = np.argsort(proximite_relative)[::-1]
    
    resultats = pd.DataFrame({
        'Rang': range(1, n_alternatives + 1),
        'Alternative': [alternatives[i] for i in classement],
        'Score C_i': [proximite_relative[i] for i in classement]
    })
    
    st.dataframe(resultats.style.format({'Score C_i': '{:.4f}'}).background_gradient(
        subset=['Score C_i'], cmap='RdYlGn', vmin=0, vmax=1
    ), use_container_width=True)
    
    # Affichage du meilleur choix
    st.success(f"✨ **Meilleur choix : {alternatives[classement[0]]}** avec un score de {proximite_relative[classement[0]]:.4f}")
    
    # Graphique
    st.subheader("📊 Visualisation des scores")
    chart_data = pd.DataFrame({
        'Alternative': alternatives,
        'Score TOPSIS': proximite_relative
    }).sort_values('Score TOPSIS', ascending=True)
    
    st.bar_chart(chart_data.set_index('Alternative'))

st.markdown("---")
st.markdown("**📚 Référence:** A Novel Multi-Criteria Decision-Making Model for Building Material Supplier Selection Based on Entropy-AHP Weighted TOPSIS")
