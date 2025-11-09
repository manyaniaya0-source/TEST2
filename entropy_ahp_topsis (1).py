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
st.header("⚖️ Étape 2 : Méthode AHP Hiérarchique")
st.markdown("""
**Échelle de Saaty :** 1=Égal, 3=Modérément important, 5=Fortement important, 
7=Très fortement important, 9=Extrêmement important
""")

# ==================== MATRICE 1: Comparaison des 3 Facettes (A, B, C) ====================
st.subheader("🔢 Matrice 1 : Comparaison des Facettes (A, B, C)")
st.markdown("**A = Product satisfaction, B = Supply innovation capability, C = Service level**")

facettes = ["A: Product satisfaction", "B: Supply innovation", "C: Service level"]
matrice_facettes = np.ones((3, 3))

col1, col2, col3 = st.columns(3)
with col1:
    val_AB = st.number_input(
        "A vs B",
        min_value=0.111, max_value=9.0, value=1.0, step=0.5,
        key="facet_AB",
        help="Quelle est l'importance de A par rapport à B?"
    )
    matrice_facettes[0, 1] = val_AB
    matrice_facettes[1, 0] = 1 / val_AB

with col2:
    val_AC = st.number_input(
        "A vs C",
        min_value=0.111, max_value=9.0, value=1.0, step=0.5,
        key="facet_AC"
    )
    matrice_facettes[0, 2] = val_AC
    matrice_facettes[2, 0] = 1 / val_AC

with col3:
    val_BC = st.number_input(
        "B vs C",
        min_value=0.111, max_value=9.0, value=1.0, step=0.5,
        key="facet_BC"
    )
    matrice_facettes[1, 2] = val_BC
    matrice_facettes[2, 1] = 1 / val_BC

# Afficher la matrice complète
st.markdown("**Matrice de comparaison par paires (Facettes) :**")
df_mat_facettes = pd.DataFrame(matrice_facettes, 
                               index=["A", "B", "C"], 
                               columns=["A", "B", "C"])
st.dataframe(df_mat_facettes.style.format("{:.3f}"), use_container_width=True)

# Calcul des poids des facettes
somme_col_facettes = matrice_facettes.sum(axis=0)
matrice_facettes_norm = matrice_facettes / somme_col_facettes
poids_facettes = matrice_facettes_norm.mean(axis=1)

st.markdown("**✅ Poids des facettes :**")
df_facettes = pd.DataFrame({
    'Facette': ["A", "B", "C"],
    'Poids': poids_facettes
})
st.dataframe(df_facettes.style.format({'Poids': '{:.4f}'}))

st.markdown("---")

# ==================== MATRICE 2: Comparaison C1, C2, C3 (Composantes de A) ====================
st.subheader("🔢 Matrice 2 : Comparaison des Critères de A (C₁, C₂, C₃)")
st.markdown("**C₁: Qualified products (%), C₂: Product price ($1000), C₃: Market share (%)**")

matrice_groupe1 = np.ones((3, 3))
col1, col2, col3 = st.columns(3)
with col1:
    g1_12 = st.number_input("C₁ vs C₂", 0.111, 9.0, 1.0, 0.5, key="g1_12")
    matrice_groupe1[0, 1] = g1_12
    matrice_groupe1[1, 0] = 1 / g1_12
with col2:
    g1_13 = st.number_input("C₁ vs C₃", 0.111, 9.0, 1.0, 0.5, key="g1_13")
    matrice_groupe1[0, 2] = g1_13
    matrice_groupe1[2, 0] = 1 / g1_13
with col3:
    g1_23 = st.number_input("C₂ vs C₃", 0.111, 9.0, 1.0, 0.5, key="g1_23")
    matrice_groupe1[1, 2] = g1_23
    matrice_groupe1[2, 1] = 1 / g1_23

# Afficher la matrice
st.markdown("**Matrice de comparaison par paires (Groupe A) :**")
df_mat_g1 = pd.DataFrame(matrice_groupe1, 
                         index=["C₁", "C₂", "C₃"], 
                         columns=["C₁", "C₂", "C₃"])
st.dataframe(df_mat_g1.style.format("{:.3f}"), use_container_width=True)

poids_local_g1 = (matrice_groupe1 / matrice_groupe1.sum(axis=0)).mean(axis=1)

st.markdown("**✅ Poids locaux :**")
df_poids_g1 = pd.DataFrame({
    'Critère': ["C₁", "C₂", "C₃"],
    'Poids Local': poids_local_g1
})
st.dataframe(df_poids_g1.style.format({'Poids Local': '{:.4f}'}))

st.markdown("---")

# ==================== MATRICE 3: Comparaison C4, C5 (Composantes de B) ====================
st.subheader("🔢 Matrice 3 : Comparaison des Critères de B (C₄, C₅)")
st.markdown("**C₄: Supply capacity (kg/time), C₅: New product development rate (%)**")

matrice_groupe2 = np.ones((2, 2))
g2_12 = st.number_input("C₄ vs C₅", 0.111, 9.0, 1.0, 0.5, key="g2_12")
matrice_groupe2[0, 1] = g2_12
matrice_groupe2[1, 0] = 1 / g2_12

# Afficher la matrice
st.markdown("**Matrice de comparaison par paires (Groupe B) :**")
df_mat_g2 = pd.DataFrame(matrice_groupe2, 
                         index=["C₄", "C₅"], 
                         columns=["C₄", "C₅"])
st.dataframe(df_mat_g2.style.format("{:.3f}"), use_container_width=True)

poids_local_g2 = (matrice_groupe2 / matrice_groupe2.sum(axis=0)).mean(axis=1)

st.markdown("**✅ Poids locaux :**")
df_poids_g2 = pd.DataFrame({
    'Critère': ["C₄", "C₅"],
    'Poids Local': poids_local_g2
})
st.dataframe(df_poids_g2.style.format({'Poids Local': '{:.4f}'}))

st.markdown("---")

# ==================== MATRICE 4: Comparaison C6, C7 (Composantes de C) ====================
st.subheader("🔢 Matrice 4 : Comparaison des Critères de C (C₆, C₇)")
st.markdown("**C₆: Delivery time (days), C₇: Delivery on time ratio (%)**")

matrice_groupe3 = np.ones((2, 2))
g3_12 = st.number_input("C₆ vs C₇", 0.111, 9.0, 1.0, 0.5, key="g3_12")
matrice_groupe3[0, 1] = g3_12
matrice_groupe3[1, 0] = 1 / g3_12

# Afficher la matrice
st.markdown("**Matrice de comparaison par paires (Groupe C) :**")
df_mat_g3 = pd.DataFrame(matrice_groupe3, 
                         index=["C₆", "C₇"], 
                         columns=["C₆", "C₇"])
st.dataframe(df_mat_g3.style.format("{:.3f}"), use_container_width=True)

poids_local_g3 = (matrice_groupe3 / matrice_groupe3.sum(axis=0)).mean(axis=1)

st.markdown("**✅ Poids locaux :**")
df_poids_g3 = pd.DataFrame({
    'Critère': ["C₆", "C₇"],
    'Poids Local': poids_local_g3
})
st.dataframe(df_poids_g3.style.format({'Poids Local': '{:.4f}'}))

st.markdown("---")

# Calcul des poids globaux (hiérarchiques)
poids_ahp = np.zeros(n_criteres)
if n_criteres >= 7:
    poids_ahp[0:3] = poids_local_g1 * poids_facettes[0]  # C1, C2, C3
    poids_ahp[3:5] = poids_local_g2 * poids_facettes[1]  # C4, C5
    poids_ahp[5:7] = poids_local_g3 * poids_facettes[2]  # C6, C7
    if n_criteres > 7:
        poids_ahp[7:] = (1 - poids_ahp[:7].sum()) / (n_criteres - 7)
else:
    poids_ahp = np.ones(n_criteres) / n_criteres

st.markdown("---")

# Affichage du résumé des poids AHP hiérarchiques
st.subheader("📊 Résumé des Poids AHP Hiérarchiques")

if n_criteres == 7:
    df_poids_ahp = pd.DataFrame({
        'Critère': criteres,
        'Facette': ['Product satisfaction']*3 + ['Supply innovation']*2 + ['Service level']*2,
        'Poids Local': list(poids_local_g1) + list(poids_local_g2) + list(poids_local_g3),
        'Poids Global (w_h)': poids_ahp
    })
    st.dataframe(df_poids_ahp.style.format({
        'Poids Local': '{:.4f}',
        'Poids Global (w_h)': '{:.4f}'
    }), use_container_width=True)
    st.info("💡 **Poids Global** = Poids Local × Poids de la Facette")
else:
    df_poids_ahp = pd.DataFrame({
        'Critère': criteres,
        'Poids AHP (w_h)': poids_ahp
    })
    st.dataframe(df_poids_ahp.style.format({'Poids AHP (w_h)': '{:.4f}'}), use_container_width=True)

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