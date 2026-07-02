import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import time

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Pro Analiz & Tahmin Odası", layout="wide", page_icon="⚽")

# --- 1. VERİ ÇEKME / SİMÜLASYON MODÜLÜ ---
# Kanka, burası FBref'ten veriyi çekeceğin yer. 
# Gerçek bir senaryoda pd.read_html(fbref_url) ile tabloları alıp temizlersin.
# Şimdilik uygulamanın çökmemesi ve matematiği test edebilmen için örnek bir veri fonksiyonu yazıyorum.
@st.cache_data(ttl=3600) # Veriyi 1 saat önbellekte tutar, siteyi yormaz
def fetch_league_data(league_name):
    # FBref scraping mantığı burada olacak. Örnek veri yapısı:
    data = {
        'Takım': ['Arsenal', 'Man City', 'Liverpool', 'Aston Villa', 'Tottenham'],
        'Atilan_Ort': [2.3, 2.5, 2.2, 1.8, 1.9],
        'Yenilen_Ort': [0.8, 0.9, 1.1, 1.4, 1.6]
    }
    return pd.DataFrame(data)

# --- 2. POISSON MATEMATİK MODÜLÜ ---
def calculate_probabilities(home_scored, home_conceded, away_scored, away_conceded, league_avg_goals=2.8):
    # Beklenen gol sayıları (xG)
    home_expectancy = (home_scored * away_conceded) / (league_avg_goals / 2)
    away_expectancy = (away_scored * home_conceded) / (league_avg_goals / 2)
    
    # 0'dan 5 gole kadar ihtimal matrisi
    max_goals = 6
    prob_matrix = np.zeros((max_goals, max_goals))
    
    for i in range(max_goals):
        for j in range(max_goals):
            prob_matrix[i, j] = poisson.pmf(i, home_expectancy) * poisson.pmf(j, away_expectancy)
    
    # İhtimalleri Toplama
    home_win = np.sum(np.tril(prob_matrix, -1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.triu(prob_matrix, 1))
    
    # Alt / Üst
    under_1_5 = np.sum(prob_matrix[np.add.outer(np.arange(max_goals), np.arange(max_goals)) < 1.5])
    over_1_5 = 1 - under_1_5
    
    under_2_5 = np.sum(prob_matrix[np.add.outer(np.arange(max_goals), np.arange(max_goals)) < 2.5])
    over_2_5 = 1 - under_2_5
    
    under_3_5 = np.sum(prob_matrix[np.add.outer(np.arange(max_goals), np.arange(max_goals)) < 3.5])
    over_3_5 = 1 - under_3_5
    
    # Karşılıklı Gol (KG)
    btts_no = np.sum(prob_matrix[0, :]) + np.sum(prob_matrix[:, 0]) - prob_matrix[0, 0]
    btts_yes = 1 - btts_no
    
    return {
        "1": home_win, "0": draw, "2": away_win,
        "A1.5": under_1_5, "U1.5": over_1_5,
        "A2.5": under_2_5, "U2.5": over_2_5,
        "A3.5": under_3_5, "U3.5": over_3_5,
        "KG_Var": btts_yes, "KG_Yok": btts_no
    }

# --- 3. GRAFİK MODÜLÜ ---
def create_pie_chart(labels, values, title, colors):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors)])
    fig.update_layout(title_text=title, margin=dict(t=40, b=0, l=0, r=0), height=300)
    return fig

# --- 4. ARAYÜZ (UI) ---
st.title("⚽ Profesyonel Maç Analiz ve Tahmin Odası")
st.markdown("---")

# Kenar Çubuğu (Sidebar)
st.sidebar.header("⚙️ Maç Seçimi")
league = st.sidebar.selectbox("Lig Seçin", ["Premier League", "La Liga", "Serie A", "Super Lig"])

df = fetch_league_data(league)

home_team = st.sidebar.selectbox("Ev Sahibi Takım", df['Takım'].tolist())
away_teams = df[df['Takım'] != home_team]['Takım'].tolist()
away_team = st.sidebar.selectbox("Deplasman Takımı", away_teams)

analyze_btn = st.sidebar.button("Maçı Analiz Et 🚀", use_container_width=True)

if analyze_btn:
    with st.spinner('FBref istatistikleri çekiliyor ve yapay zeka analiz ediyor...'):
        time.sleep(1) # Gerçekçilik katmak için kısa bir bekleme
        
        home_stats = df[df['Takım'] == home_team].iloc[0]
        away_stats = df[df['Takım'] == away_team].iloc[0]
        
        probs = calculate_probabilities(
            home_stats['Atilan_Ort'], home_stats['Yenilen_Ort'],
            away_stats['Atilan_Ort'], away_stats['Yenilen_Ort']
        )
        
        # Üst Metrikler
        col1, col2, col3 = st.columns(3)
        col1.metric(label=f"{home_team} Atılan/Yenilen (Ort)", value=f"{home_stats['Atilan_Ort']} / {home_stats['Yenilen_Ort']}")
        col2.metric(label="Maç", value=f"{home_team} vs {away_team}")
        col3.metric(label=f"{away_team} Atılan/Yenilen (Ort)", value=f"{away_stats['Atilan_Ort']} / {away_stats['Yenilen_Ort']}")
        
        st.markdown("### 📊 Analiz Sonuçları ve İhtimaller")
        
        # 3'lü Grafik Sütunu
        g_col1, g_col2, g_col3 = st.columns(3)
        
        with g_col1:
            fig_ms = create_pie_chart(
                ['Ev Sahibi (1)', 'Beraberlik (0)', 'Deplasman (2)'], 
                [probs['1'], probs['0'], probs['2']], 
                "Maç Sonucu İhtimali", 
                ['#1f77b4', '#7f7f7f', '#d62728']
            )
            st.plotly_chart(fig_ms, use_container_width=True)
            
        with g_col2:
            fig_ou = create_pie_chart(
                ['2.5 Üst', '2.5 Alt'], 
                [probs['U2.5'], probs['A2.5']], 
                "2.5 Alt/Üst Oranları", 
                ['#2ca02c', '#ff7f0e']
            )
            st.plotly_chart(fig_ou, use_container_width=True)
            
        with g_col3:
            fig_btts = create_pie_chart(
                ['KG Var', 'KG Yok'], 
                [probs['KG_Var'], probs['KG_Yok']], 
                "Karşılıklı Gol", 
                ['#9467bd', '#8c564b']
            )
            st.plotly_chart(fig_btts, use_container_width=True)

        st.markdown("### 📈 Detaylı Gol Barajları")
        bar_df = pd.DataFrame({
            "Baraj": ["1.5 Gol", "2.5 Gol", "3.5 Gol"],
            "Alt İhtimali (%)": [probs['A1.5']*100, probs['A2.5']*100, probs['A3.5']*100],
            "Üst İhtimali (%)": [probs['U1.5']*100, probs['U2.5']*100, probs['U3.5']*100]
        })
        st.dataframe(bar_df.style.format({"Alt İhtimali (%)": "{:.1f}%", "Üst İhtimali (%)": "{:.1f}%"}), use_container_width=True)
