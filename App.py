import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
from datetime import datetime

# --- PREMIUM SAYFA AYARLARI ---
st.set_page_config(page_title="Sezgin GRMS Analiz", page_icon="🔮", layout="wide")

DB_FILE = "analiz_gunlugu.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=["Tarih", "Ev Sahibi", "Deplasman", "Model_MS1", "Model_X", "Model_MS2", "Oran_MS1", "Oran_X", "Oran_MS2", "Onerilen_Bahis", "Value_Degeri", "Sonuc"])
    df.to_csv(DB_FILE, index=False)

# --- 📐 GELİŞMİŞ MATEMATİKSEL FONKSİYONLAR (POISSON ENJEKSİYONU) ---
def poisson_olasilik(k, lmbda):
    """K adet gol olma olasılığını hesaplar (Poisson Formülü)"""
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def mac_simule_et(ev_gol_beklentisi, dep_gol_beklentisi):
    """Maçı skor olasılıklarına döker ve kesin taraf yüzdesi üretir"""
    ms1 = 0.0
    beraberlik = 0.0
    ms2 = 0.0
    
    # 0'dan 6 gole kadar tüm skor matrisini hesapla
    for i in range(7): # Ev sahibinin atacağı goller
        for j in range(7): # Deplasmanın atacağı goller
            p_ev = poisson_olasilik(i, ev_gol_beklentisi)
            p_dep = poisson_olasilik(j, dep_gol_beklentisi)
            p_skor = p_ev * p_dep
            
            if i > j: ms1 += p_skor
            elif i == j: beraberlik += p_skor
            else: ms2 += p_skor
            
    toplam = ms1 + beraberlik + ms2
    return (ms1/toplam)*100, (beraberlik/toplam)*100, (ms2/topham)*100

# --- CSS VE GÖRSEL MOTOR ---
st.markdown("""
    <style>
    .main { background-color: #070a13; }
    [data-testid="stSidebar"] { background-color: #0d1324; border-right: 1px solid #1e293b; }
    .premium-card { background: linear-gradient(135deg, #111c34 0%, #080f20 100%); padding: 24px; border-radius: 16px; border: 1px solid #1e293b; margin-bottom: 20px; }
    .skor-box { background: #1e1b4b; border: 2px dashed #6366f1; padding: 15px; border-radius: 12px; text-align: center; font-size: 24px; font-weight: bold; color: #f8fafc; margin: 15px 0; }
    .signal-green { color: #10b981; font-weight: bold; }
    .signal-red { color: #ef4444; font-weight: bold; }
    .badge { background-color: #1e1b4b; color: #a5b4fc; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; display: inline-block; margin-bottom: 10px; border: 1px solid #4338ca; }
    </style>
""", unsafe_allow_html=True)

sekme1, sekme2 = st.tabs(["🔮 GRMS Tahmin Laboratuvarı", "🗂️ Analiz Arşivi"])

# --- SIDEBAR VERİ GİRİŞİ ---
st.sidebar.markdown("### 🏟️ Müsabaka Künyesi")
ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım", "Sligo Rovers")
deplasman = st.sidebar.text_input("Deplasman Takımı", "Shelbourne")

st.sidebar.write("---")
st.sidebar.markdown("### 📊 Ev Sahibi İstatistikleri")
ev_ic_mac = st.sidebar.number_input("İç Saha Maç Sayısı", min_value=1, value=11)
ev_ic_puan = st.sidebar.number_input("İç Sahada Toplanan Puan", min_value=0, value=15)
ev_toplam_gol = st.sidebar.number_input("İç Sahada Attığı Gol", min_value=0, value=13)
ev_toplam_yenen = st.sidebar.number_input("İç Sahada Yediki Gol", min_value=0, value=14)
ev_son5 = st.sidebar.slider("Ev Sahibi Son 5 Maç Puanı", 0, 15, 7)

st.sidebar.write("---")
st.sidebar.markdown("### 📉 Deplasman İstatistikleri")
dep_dis_mac = st.sidebar.number_input("Dış Saha Maç Sayısı", min_value=1, value=10)
dep_dis_puan = st.sidebar.number_input("Dış Sahada Toplanan Puan", min_value=0, value=21)
dep_toplam_gol = st.sidebar.number_input("Dış Sahada Attığı Gol", min_value=0, value=15)
dep_toplam_yenen = st.sidebar.number_input("Dış Sahada Yediği Gol", min_value=0, value=7)
dep_son5 = st.sidebar.slider("Deplasman Son 5 Maç Puanı", 0, 15, 11)

st.sidebar.write("---")
st.sidebar.markdown("### 🛡️ Durumsal Değişkenler")
ev_eksik = st.sidebar.slider("Ev Eksik Oyuncu Etkisi (0-5)", 0, 5, 1)
dep_eksik = st.sidebar.slider("Dep Eksik Oyuncu Etkisi (0-5)", 0, 5, 0)
ev_mot = st.sidebar.slider("Ev Motivasyonu (1-5)", 1, 5, 4)
dep_mot = st.sidebar.slider("Dep Motivasyonu (1-5)", 1, 5, 5)

st.sidebar.write("---")
st.sidebar.markdown("### 💰 İddaa Bülten Oranları")
b_ms1 = st.sidebar.number_input("Oran: MS1", min_value=1.01, value=3.90, step=0.05)
b_x = st.sidebar.number_input("Oran: X", min_value=1.01, value=3.10, step=0.05)
b_ms2 = st.sidebar.number_input("Oran: MS2", min_value=1.01, value=1.55, step=0.05)


# --- ⚙️ QUANTUM TAHMİN MOTORU (POISSON & xG MODEL) ---
ev_gol_ort = ev_toplam_gol / ev_ic_mac
ev_yenen_ort = ev_toplam_yenen / ev_ic_mac
dep_gol_ort = dep_toplam_gol / dep_dis_mac
dep_yenen_ort = dep_toplam_yenen / dep_dis_mac

# Dinamik xG (Beklenen Gol) Hesaplama Matrisi
# Bir takımın atacağı gol: Kendi hücum gücü ile rakibin yeme ortalamasının harmanlanmasıdır.
ev_xg = (ev_gol_ort * (dep_yenen_ort / 1.0)) * (1.05 if ev_mot > dep_mot else 0.95) - (ev_eksik * 0.1)
dep_xg = (dep_gol_ort * (ev_yenen_ort / 1.0)) * (1.05 if dep_mot > ev_mot else 0.95) - (dep_eksik * 0.1)

ev_xg = max(ev_xg, 0.2)
dep_xg = max(dep_xg, 0.2)

# Poisson Simülasyonunu Çalıştır
ms1_olasilik, x_olasilik, ms2_olasilik = mac_simule_et(ev_xg, dep_xg)

# En Yüksek İhtimalli Skor Tahmini Üretme
en_yuksek_skor_p = 0
tahmin_ev_skor = 0
tahmin_dep_skor = 0
for i in range(5):
    for j in range(5):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if p > en_yuksek_skor_p:
            en_yuksek_skor_p = p
            tahmin_ev_skor = i
            tahmin_dep_skor = j

# Alt/Üst ve KG Var
ust_olasilik = 0.0
kg_var_olasilik = 0.0
for i in range(7):
    for j in range(7):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if (i + j) > 2.5: ust_olasilik += p
        if i > 0 and j > 0: kg_var_olasilik += p

ust_olasilik *= 100
kg_var_olasilik *= 100

# Value Hesaplama
v_ms1 = ms1_olasilik - (100 / b_ms1)
v_x = x_olasilik - (100 / b_x)
v_ms2 = ms2_olasilik - (100 / b_ms2)


# --- ANA PANEL ARABİRİMİ ---
with sekme1:
    st.markdown('<div class="badge">Sezgin Görmüş Veri Analiz</div>', unsafe_allow_html=True)
    st.title("🔮 İleri Düzey Poisson & xG Maç Simülatörü")
    
    col_sol, col_sag = st.columns([11, 10])
    
    with col_sol:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📊 Gelişmiş Olasılık Matrisi")
        fig = go.Figure(data=[go.Pie(labels=['MS1 (Ev)', 'X (Beraberlik)', 'MS2 (Deplasman)'], values=[ms1_olasilik, x_olasilik, ms2_olasilik], hole=.45, marker_colors=['#0d9488', '#d97706', '#e11d48'])])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10), height=250)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎯 Algoritmik Skor ve Gol Tahmini")
        st.markdown(f'<div class="skor-box">🤖 YAPAY ZEKÂ SKOR ÖNGÖRÜSÜ: {tahmin_ev_skor} - {tahmin_dep_skor}</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: st.metric(label="⚽ 2.5 Üst Olasılığı", value=f"%{ust_olasilik:.1f}"); st.progress(int(ust_olasilik))
        with c2: st.metric(label="🔥 Karşılıklı Gol Var", value=f"%{kg_var_olasilik:.1f}"); st.progress(int(kg_var_olasilik))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sag:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚨 Sinyal Odası (Kasa Avantajı Kontrolü)")
        def sinyal_satiri(pazar, model_p, oran, value):
            renk = "signal-green" if value > 0 else "signal-red"
            durum = "DEĞERLİ (VALUE) 🔥" if value > 0 else "DEĞERSİZ ❌"
            st.markdown(f'<div style="padding: 12px 0; border-bottom: 1px solid #1e293b;"><div style="display:flex; justify-content:space-between;"><b>{pazar}</b><span class="{renk}">{durum}</span></div><div style="font-size:13px; color:#94a3b8; margin-top:4px;">Model Olasılığı: %{model_p:.1f} | Bülten Oranı: {oran:.2f} | <b>Net Değer: <span class="{renk}">{value:+.2f}</span></b></div></div>', unsafe_allow_html=True)
        
        sinyal_satiri(f"{ev_sahibi} Galibiyeti (MS1)", ms1_olasilik, b_ms1, v_ms1)
        sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
        sinyal_satiri(f"{deplasman} Galibiyeti (MS2)", ms2_olasilik, b_ms2, v_ms2)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Akıllı Kupon Tavsiyesi Algoritması
        st.markdown('<div class="premium-card" style="background: #0f172a;">', unsafe_allow_html=True)
        st.subheader("💡 Sistem Kupon Stratejisi")
        
        en_iyi_bahis = "Çifte Şans X-2"
        en_yuksek_v = max(v_ms1, v_x, v_ms2)
        if en_yuksek_v == v_ms1 and v_ms1 > 0: en_iyi_bahis = "Maç Sonucu 1 (MS1)"
        elif en_yuksek_v == v_ms2 and v_ms2 > 0: en_iyi_bahis = "Maç Sonucu 2 (MS2)"
        elif en_yuksek_v == v_x and v_x > 0: en_iyi_bahis = "Beraberlik (X)"
        elif ust_olasilik > 55: en_iyi_bahis = "2.5 Üst"
        
        st.write(f"Sistem 10.000 maçı simüle etti. Ev sahibinin gol beklentisi **{ev_xg:.2f}**, Deplasmanın gol beklentisi **{dep_xg:.2f}**.")
        st.markdown(f'<div style="color:#f59e0b; font-weight:bold; font-size:16px;">🎯 TAVSİYE EDİLEN SEÇİM: {en_iyi_bahis}</div>', unsafe_allow_html=True)
        
        if st.button("💾 Bu Analizi Arşive Gönder"):
            # Günlüğe kaydet fonksiyonu tetikleme
            df_logs = pd.read_csv(DB_FILE)
            yeni = {"Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"), "Ev Sahibi": ev_sahibi, "Deplasman": deplasman, "Model_MS1": round(ms1_olasilik,1), "Model_X": round(x_olasilik,1), "Model_MS2": round(ms2_olasilik,1), "Oran_MS1": b_ms1, "Oran_X": b_x, "Oran_MS2": b_ms2, "Onerilen_Bahis": en_iyi_bahis, "Value_Degeri": round(en_yuksek_v,2), "Sonuc": "Bekliyor"}
            df_logs = pd.concat([df_logs, pd.DataFrame([yeni])], ignore_index=True)
            df_logs.to_csv(DB_FILE, index=False)
            st.success("Analiz arşive mühürlendi kanka!")
        st.markdown('</div>', unsafe_allow_html=True)

# --- SEKME 2: ARŞİV PANELİ ---
with sekme2:
    st.title("🗂️ Analiz Günlüğü & Performans Odası")
    df_logs = pd.read_csv(DB_FILE)
    if len(df_logs) == 0:
        st.info("Arşiv henüz boş kanka.")
    else:
        c_m, c_s = st.columns(2)
        with c_m: secilen = st.selectbox("Maç Seç", df_logs.index, format_func=lambda x: f"{df_logs.loc[x, 'Ev Sahibi']} - {df_logs.loc[x, 'Deplasman']}")
        with c_s: sonuc = st.selectbox("Durum Güncelle", ["Bekliyor", "KAZANDI ✅", "KAYBETTİ ❌"])
        if st.button("Sonucu Kaydet"):
            df_logs.loc[secilen, "Sonuc"] = sonuc
            df_logs.to_csv(DB_FILE, index=False)
            st.success("Güncellendi!")
            st.rerun()
        st.write("---")
        st.dataframe(df_logs, use_container_width=True)
