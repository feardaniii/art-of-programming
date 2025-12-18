import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Încercăm să importăm ydata_profiling, dar gestionăm cazul în care nu e instalat
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# Configurare pagină (trebuie să fie prima comandă Streamlit)
st.set_page_config(page_title="Tema Dashboard - Sesiunea 25", layout="wide", page_icon="📊")

# --- FUNCȚII AUXILIARE ---

def generate_client_data():
    """Generează date fictive pentru clienți dacă nu se încarcă fișier."""
    data = {
        'Nume': [f'Client {i}' for i in range(1, 101)],
        'Oras': np.random.choice(['București', 'Cluj', 'Iași', 'Timișoara', 'Brașov'], 100),
        'Varsta': np.random.randint(18, 70, 100),
        'Venit_Anual': np.random.randint(20000, 120000, 100),
        'Achizitii': np.random.randint(1, 50, 100),
        'Scor_Fidelitate': np.random.uniform(1, 10, 100).round(1) # Echivalent "Performanță"
    }
    return pd.DataFrame(data)

def generate_product_data():
    """Generează date fictive pentru produse."""
    data = {
        'Nume_Produs': [f'Produs {i}' for i in range(1, 51)],
        'Categorie': np.random.choice(['Electronice', 'Electrocasnice', 'Mobilier', 'Accesorii'], 50),
        'Pret': np.random.randint(50, 5000, 50),
        'Cantitate': np.random.randint(1, 200, 50)
    }
    return pd.DataFrame(data)

# --- TITLU PRINCIPAL ---
st.title("📊 Tema Sesiunea 25: Dashboarduri Complete")
st.markdown("Această aplicație rezolvă cerințele temei: **Analiza Clienților** și **Analiza Produselor**.")

# Folosim Tabs pentru a separa cele două părți ale temei
tab1, tab2 = st.tabs(["👥 Dashboard Clienți", "🛒 Dashboard Produse"])

# ==============================================================================
# TAB 1: DASHBOARD CLIENȚI
# ==============================================================================
with tab1:
    st.header("Analiză Clienți: Demografie & Comportament")
    
    # 1. Încărcare Date sau Generare
    uploaded_file_clients = st.file_uploader("Încarcă CSV Clienți (sau folosește date demo)", type=['csv'], key="clients")
    
    if uploaded_file_clients:
        df_clients = pd.read_csv(uploaded_file_clients)
        st.success("Fișier încărcat cu succes!")
    else:
        st.info("Se folosesc date generate automat (Demo). Încarcă un CSV pentru a schimba.")
        df_clients = generate_client_data()

    # Layout: Sidebar pentru filtre (doar când suntem în Tab 1, dar Streamlit randează sidebar global)
    # Vom pune filtrele în pagina principală folosind expander sau coloane pentru a nu le amesteca
    
    st.subheader("🔍 Filtrare Date Clienți")
    
    col_filtre1, col_filtre2, col_filtre3 = st.columns(3)
    
    with col_filtre1:
        # Filtru Oraș (Categoric)
        orase_disponibile = df_clients['Oras'].unique().tolist()
        orase_selectate = st.multiselect("Selectează Orașul", orase_disponibile, default=orase_disponibile)
    
    with col_filtre2:
        # Filtru Vârstă (Numeric)
        min_age, max_age = int(df_clients['Varsta'].min()), int(df_clients['Varsta'].max())
        age_range = st.slider("Interval Vârstă", min_age, max_age, (min_age, max_age))

    with col_filtre3:
        # Filtru Venit (Numeric)
        min_inc, max_inc = int(df_clients['Venit_Anual'].min()), int(df_clients['Venit_Anual'].max())
        income_range = st.slider("Interval Venit Anual", min_inc, max_inc, (min_inc, max_inc))

    # Aplicare Filtre
    df_filtered_clients = df_clients[
        (df_clients['Oras'].isin(orase_selectate)) &
        (df_clients['Varsta'] >= age_range[0]) & (df_clients['Varsta'] <= age_range[1]) &
        (df_clients['Venit_Anual'] >= income_range[0]) & (df_clients['Venit_Anual'] <= income_range[1])
    ]

    st.write(f"Arătăm **{len(df_filtered_clients)}** clienți din totalul de **{len(df_clients)}**.")
    st.dataframe(df_filtered_clients.head())

    # 2. Vizualizări Cerute
    st.divider()
    st.subheader("📊 Vizualizări Grafice")
    
    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        st.markdown("**1. Distribuția clienților pe orașe**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Oras', data=df_filtered_clients, palette='viridis', ax=ax1)
        ax1.set_title("Număr Clienți per Oraș")
        st.pyplot(fig1)

    with col_graph2:
        st.markdown("**2. Corelație Vârstă vs. Performanță (Scor/Venit)**")
        # Cerința temei: Corelație între vârstă și performanță
        # Vom folosi Scatter Plot și o linie de regresie
        y_axis_choice = st.selectbox("Alege metrica de performanță:", ['Scor_Fidelitate', 'Venit_Anual', 'Achizitii'])
        
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='Varsta', y=y_axis_choice, data=df_filtered_clients, hue='Oras', ax=ax2)
        sns.regplot(x='Varsta', y=y_axis_choice, data=df_filtered_clients, scatter=False, ax=ax2, color='red')
        ax2.set_title(f"Corelație: Vârstă vs {y_axis_choice}")
        st.pyplot(fig2)

    # 3. Pandas Profiling (Opțional)
    with st.expander("📈 Vezi Raport Avansat (Pandas Profiling)"):
        if PROFILING_AVAILABLE:
            if st.button("Generează Raport Clienți"):
                pr = ProfileReport(df_filtered_clients, explorative=True)
                st_profile_report(pr)
        else:
            st.warning("Te rog instalează `pandas-profiling` și `streamlit-pandas-profiling` pentru a vedea raportul.")

    # 4. Export
    csv_clients = df_filtered_clients.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descarcă Date Clienți Filtrate",
        data=csv_clients,
        file_name='clienti_filtrati.csv',
        mime='text/csv',
    )


# ==============================================================================
# TAB 2: DASHBOARD PRODUSE
# ==============================================================================
with tab2:
    st.header("Analiză Vânzări Produse")
    
    uploaded_file_prod = st.file_uploader("Încarcă CSV Produse", type=['csv'], key="prod")
    
    if uploaded_file_prod:
        df_prod = pd.read_csv(uploaded_file_prod)
    else:
        df_prod = generate_product_data()
        
    # Calcul coloană nouă (Cerința temei: Vânzări Totale)
    if 'Vanzari_Totale' not in df_prod.columns:
        df_prod['Vanzari_Totale'] = df_prod['Pret'] * df_prod['Cantitate']
        
    st.subheader("📋 Date Produse (cu coloana calculată 'Vanzari_Totale')")
    
    # Filtre simple în linie
    categs = df_prod['Categorie'].unique().tolist()
    sel_categ = st.multiselect("Filtrează după Categorie", categs, default=categs)
    
    df_filtered_prod = df_prod[df_prod['Categorie'].isin(sel_categ)]
    
    # Afișare DataFrame cu evidențierea valorilor mari
    st.dataframe(df_filtered_prod.style.highlight_max(axis=0, subset=['Vanzari_Totale'], color='lightgreen'))
    
    # Statistici rapide
    total_revenue = df_filtered_prod['Vanzari_Totale'].sum()
    best_product = df_filtered_prod.loc[df_filtered_prod['Vanzari_Totale'].idxmax()]['Nume_Produs']
    
    col_kpi1, col_kpi2 = st.columns(2)
    col_kpi1.metric("💰 Venit Total (Selecție)", f"{total_revenue:,.0f} RON")
    col_kpi2.metric("🏆 Cel mai vândut produs", best_product)

    # Vizualizări Produse
    st.subheader("📊 Top Produse după Vânzări")
    
    # Luăm top 10 produse pentru a nu aglomera graficul
    top_products = df_filtered_prod.nlargest(10, 'Vanzari_Totale')
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Vanzari_Totale', y='Nume_Produs', data=top_products, palette='magma', ax=ax3)
    ax3.set_xlabel("Total Vânzări (RON)")
    ax3.set_ylabel("Produs")
    st.pyplot(fig3)

    # Export Produse
    csv_prod = df_filtered_prod.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descarcă Raport Produse",
        data=csv_prod,
        file_name='produse_procesate.csv',
        mime='text/csv',
    )