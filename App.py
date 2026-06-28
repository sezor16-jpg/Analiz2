import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
from datetime import datetime
import numpy as np

# --- SEZGİN GÖRMÜŞ AI PRO v7.0 SAYFA AYARLARI ---
st.set_page_config(page_title="Sezgin Görmüş Veri Analizi v7.0", page_icon="🔮", layout="wide")

DB_FILE = "analiz_gunlugu.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=[
        "Tarih", "Ev Sahibi", "Deplasman", 
        "Model_MS1", "Model_X", "Model_MS2", "Model_Ust", "Model_KGVar",
        "Onerilen_Bahis", "Pazar_Tipi", "Sonuc"
    ])
    df.to_csv(DB_FILE, index=False)

KUPON_DB_FILE = "kuponlar_gunlugu.csv"
if not os.path.exists(KUPON_DB_FILE):
    pd.DataFrame(columns=["Kupon_ID", "Mac_Detaylari", "Yatirilan_Tutar", "Durum", "Tarih"]).to_csv(KUPON_DB_FILE, index=False)

# --- 📐 KUSURSUZ POISSON MOTORU (PROFESYONEL LİMİT: 11) ---
def poisson_olasilik(k, lmbda):
    if lmbda <= 0: return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)
    
def mac_simule_et(ev_gol_beklentisi, dep_gol_beklentisi):
    simulasyon_sayisi = 100000
    ev_goller = np.random.poisson(ev_gol_beklentisi, simulasyon_sayisi)
    dep_goller = np.random.poisson(dep_gol_beklentisi, simulasyon_sayisi)
    
    ms1_sayisi = np.sum(ev_goller > dep_goller)
    beraberlik_sayisi = np.sum(ev_goller == dep_goller)
    ms2_sayisi = np.sum(ev_goller < dep_goller)
    
    ms1_yuzde = (ms1_sayisi / simulasyon_sayisi) * 100
    x_yuzde = (beraberlik_sayisi / simulasyon_sayisi) * 100
    ms2_yuzde = (ms2_sayisi / simulasyon_sayisi) * 100
    
    return ms1_yuzde, x_yuzde, ms2_yuzde

# 1. BÜYÜK LİG VE TAKIM VERİTABANI
LIG_VERITABANI = {
    "Türkiye Trendyol Süper Lig": [
        "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir", 
        "Kasımpaşa", "Göztepe", "Samsunspor", "Alanyaspor", "Rizespor","Çorum FK",
        "Amed SK","Erzurumspor FK","Eyüpspor","Gençlerbirliği","Gaziantep FK","Kocaelispor","Konyaspor"
    ],
    "Türkiye Trendyol 1. Lig": [
        "Ankara Keçiörengücü", "Antalyaspor", "Bandırmaspor", "Batman Petrolspor", 
        "Bodrum", "Boluspor", "Bursaspor", "Esenler Erokspor", "Fatih Karagümrük",
        "Iğdır","İstanbulspor","Kayserispor","Manisa","Mardin 1969","Muğlaspor","Pendikspor","Sarıyer","Sivasspor",
        "Ümraniyespor","Vanspor"
    ],
    "İngiltere Premier Lig": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion", 
        "Chelsea", "Coventry City", "Crystal Palace", "Everton", "Fulham",
        "Hull City","Ipswich Town","Leeds United","Liverpool","Manchester City",
        "Manchester United","Newcastle United","Nottingham Forest","Sunderland","Tottenham Hotspur"
    ],
    "İngiltere Championship": [
        "Blackburn Rovers", "Bristol City", "Burnley", "Cardiff City", "Birmingham City",
        "Derby County", "Lincoln City", "Bolton Wanderers", "Charlton Athletic",
        "Middlesbrough", "Millwall", "Norwich City",
        "Portsmouth", "Preston North End", "Queens Park Rangers", "Sheffield United",
        "Stoke City", "Swansea City", "Watford", "West Bromwich Albion",
        "Wolverhampton Wanderers","West Ham United","Wrexham","Southampton"
    ],
    "İtalya Serie B": [
        "Verona", "Avellino", "Carrarese", "Catanzaro", "Cesena",
        "Padova", "Virtus Entella", "Empoli", "Juve Stabia", "Mantova",
        "Modena", "Pisa", "LR Vicenza", "Salernitana", "Sampdoria",
        "Benevento", "Arezzo", "Sudtirol"
    ],
    "İtalya Serie A": [
        "AC Milan", "AS Roma", "Atalanta", "Bologna", "Cagliari",
        "Como", "Frosinone", "Fiorentina", "Genoa", "Inter", 
        "Juventus", "Lazio", "Lecce", "Sassuolo", "Napoli", 
        "Palermo", "Parma", "Torino", "Udinese", "Venezia", "Monza"
    ],
    "İspanya Segunda Division": [
        "Albacete", "Almeria", "Burgos", "Cadiz", "Andorra",
        "Castellon", "Cordoba", "Celta Fortuna", "Eibar", "Ceuta",
        "Eldense", "Girona", "Granada", "Las Palmas", "Leganés",
        "Mallorca", "Oviedo", "Real Sociedad B", "Tenerife", "Sabadell",
        "Sporting Gijón","Valladolid"
    ],
    "İspanya La Liga": [
        "Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Celta Vigo",
        "Espanyol", "Getafe", "Elche", "Deportivo La Coruña", "Leganes",
        "Mallorca", "Osasuna", "Racing Santander", "Rayo Vallecano", "Real Betis",
        "Real Madrid", "Real Oviedo", "Real Sociedad", "Sevilla", "Valencia",
        "Levante", "Villarreal"
    ],
    "Almanya Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Münih", "Borussia Dortmund", "Borussia Mönchengladbach",
        "Eintracht Frankfurt", "Freiburg", "Elversberg", "Hoffenheim", "Hamburg",
        "Mainz 05", "RB Leipzig", "Köln", "Paderborn 07", "Stuttgart",
        "Union Berlin", "Schalke 04", "Werder Bremen"
    ],
}

# --- PREMIUM GÖRSEL ARAYÜZ TASARIMI (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #070a13; }
    [data-testid="stSidebar"] { background-color: #0d1324; border-right: 1px solid #1e293b; padding-top: 20px; }
    .premium-card { background: linear-gradient(135deg, #111c34 0%, #080f20 100%); padding: 24px; border-radius: 16px; border: 1px solid #1e293b; margin-bottom: 20px; }
    .skor-box { background: linear-gradient(90deg, #1e1b4b 0%, #311042 100%); border: 2px dashed #8b5cf6; padding: 15px; border-radius: 12px; text-align: center; font-size: 26px; font-weight: bold; color: #f8fafc; margin: 15px 0; letter-spacing: 1px; }
    
    .signal-green { color: #10b981; font-weight: bold; }
    .signal-red { color: #ef4444; font-weight: bold; }
    .badge { background: linear-gradient(90deg, #4c1d95 0%, #2563eb 100%); color: #ffffff; padding: 6px 16px; border-radius: 9999px; font-size: 13px; font-weight: bold; display: inline-block; margin-bottom: 12px; border: 1px solid #3b82f6; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3); }
    .yorum-box { background-color: #0b1329; border-left: 4px solid #8b5cf6; padding: 18px; border-radius: 8px; font-size: 15px; color: #e2e8f0; line-height: 1.6; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

sekme1, sekme2, sekme3 = st.tabs(["📊 Analiz Ekranı", "🗂️ Arşiv Paneli", "🎟️ Kupon Odası"])

# --- SIDEBAR (KULLANICI DOSTU GELİŞMİŞ PANEL) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 20px;'>📋 VERİ SEVİYESİ</h2>", unsafe_allow_html=True)

st.sidebar.markdown("### 🏟️ Müsabaka Seçim Odası")

# 1. Lig Seçimi
lig_listesi = list(LIG_VERITABANI.keys()) + ["🌍 Diğer / Listede Olmayan Lig"]
secilen_lig = st.sidebar.selectbox("Ligi Seç kanka:", lig_listesi)

# 2. Ev Sahibi Alanı
if secilen_lig == "🌍 Diğer / Listede Olmayan Lig":
    ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adı:", "Ev Sahibi")
else:
    ev_secenekleri = ["✍️ Kendim Yazacağım..."] + LIG_VERITABANI[secilen_lig]
    ev_secim = st.sidebar.selectbox("Ev Sahibi Takım:", ev_secenekleri, index=1)
    
    if ev_secim == "✍️ Kendim Yazacağım...":
        ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adını Gir kanka:", "Ev Sahibi")
    else:
        ev_sahibi = ev_secim

# 3. Deplasman Alanı
if secilen_lig == "🌍 Diğer / Listede Olmayan Lig":
    deplasman = st.sidebar.text_input("Deplasman Takım Adı:", "Deplasman")
else:
    dep_secenekleri = ["✍️ Kendim Yazacağım..."] + LIG_VERITABANI[secilen_lig]
    dep_secim = st.sidebar.selectbox("Deplasman Takımı:", dep_secenekleri, index=2)
    
    if dep_secim == "✍️ Kendim Yazacağım...":
        deplasman = st.sidebar.text_input("Deplasman Takım Adını Gir kanka:", "Deplasman")
    else:
        deplasman = dep_secim

st.sidebar.write("---")
st.sidebar.markdown("### 🌍 Lig Karakteristiği")
lig_ort_gol = st.sidebar.slider("Lig Geneli Maç Başına Gol Ortalaması", 1.50, 4.00, 2.80, step=0.05)
st.sidebar.write("---")
st.sidebar.markdown("### 🎛️ Algoritma Taktik Dengesi")
form_agirligi = st.sidebar.slider("Son 5 Maçın (Form) Etki Oranı (%)", 20, 60, 35) / 100
sezon_agirligi = 1.0 - form_agirligi

st.sidebar.write("---")
st.sidebar.markdown("### 🏠 Ev Sahibi İstatistikleri")
ev_ic_mac = st.sidebar.number_input("İç Sahada Oynadığı Maç", min_value=1, value=15)
ev_ic_puan = st.sidebar.number_input("İç Sahada Topladığı Puan", min_value=0, value=34)
ev_toplam_gol = st.sidebar.number_input("İç Sahada Attığı Toplam Gol", min_value=0, value=38)
ev_toplam_yenen = st.sidebar.number_input("İç Sahada Yediği Toplam Gol", min_value=0, value=12)
ev_son5_attigi = st.sidebar.number_input("Son 5 Maçta Attığı Gol (Form)", min_value=0, value=11)
ev_son5_yedigi = st.sidebar.number_input("Son 5 Maçta Yediği Gol (Form)", min_value=0, value=4)
ev_cs = st.sidebar.slider("Ev Sahibi Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 2)
ev_onem = st.sidebar.slider("Ev Sahibinin Maç Önem Derecesi (1-5)", 1, 5, 5)

st.sidebar.write("---")

st.sidebar.markdown("### 🚀 Deplasman İstatistikleri")
dep_dis_mac = st.sidebar.number_input("Dış Sahada Oynadığı Maç", min_value=1, value=15)
dep_dis_puan = st.sidebar.number_input("Dış Sahada Topladığı Puan", min_value=0, value=28)
dep_toplam_gol = st.sidebar.number_input("Dış Sahada Attığı Toplam Gol", min_value=0, value=29)
dep_toplam_yenen = st.sidebar.number_input("Dış Sahada Yediği Toplam Gol", min_value=0, value=18)
dep_son5_attigi = st.sidebar.number_input("Son 5 Maçta Dışarıda Attığı Gol", min_value=0, value=9)
dep_son5_yedigi = st.sidebar.number_input("Son 5 Maçta Dışarıda Yediği Gol", min_value=0, value=6)
dep_cs = st.sidebar.slider("Deplasman Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 1)
dep_onem = st.sidebar.slider("Deplasmanın Maç Önem Derecesi (1-5)", 1, 5, 4)

st.sidebar.markdown("### 🏆 Genel Lig İstatistikleri")
col_genel1, col_genel2 = st.sidebar.columns(2)

with col_genel1:
    st.markdown("**Ev Sahibi (Genel)**")
    ev_genel_mac = st.number_input("Toplam Maç (Ev)", min_value=1, value=20, step=1)
    ev_genel_attigi = st.number_input("Toplam Attığı Gol (Ev)", min_value=0, value=35, step=1)
    ev_genel_yedigi = st.number_input("Toplam Yediği Gol (Ev)", min_value=0, value=20, step=1)

with col_genel2:
    st.markdown("**Deplasman (Genel)**")
    dep_genel_mac = st.number_input("Toplam Maç (Dep)", min_value=1, value=20, step=1)
    dep_genel_attigi = st.number_input("Toplam Attığı Gol (Dep)", min_value=0, value=25, step=1)
    dep_genel_yedigi = st.number_input("Toplam Yediği Gol (Dep)", min_value=0, value=28, step=1)

st.sidebar.write("---")
st.sidebar.markdown("### 🛡️ Sağlık & Kadro Eksik Raporu")
ev_kritik_eksik = st.sidebar.slider("Ev Sahibi Kritik Eksik (As Kaleci, Golcü vb.)", 0, 3, 0)
ev_normal_eksik = st.sidebar.slider("Ev Sahibi Normal Eksik (Rotasyon Oyuncusu)", 0, 5, 1)
dep_kritik_eksik = st.sidebar.slider("Deplasman Kritik Eksik (As Kaleci, Golcü vb.)", 0, 3, 1)
dep_normal_eksik = st.sidebar.slider("Deplasman Normal Eksik (Rotasyon Oyuncusu)", 0, 5, 2)

st.sidebar.write("---")
st.sidebar.markdown("### 📊 Bülten Oran Odası")
b_ms1 = st.sidebar.number_input("Bülten Oranı: MS 1", min_value=1.01, value=1.85)
b_x = st.sidebar.number_input("Bülten Oranı: Beraberlik (X)", min_value=1.01, value=3.60)
b_ms2 = st.sidebar.number_input("Bülten Oranı: MS 2", min_value=1.01, value=3.40)


# --- ⚙️ SEZGİN GÖRMÜŞ MATHEMATICAL MATRIX MOTORU ---
ev_ic_hucum_ort = ev_toplam_gol / ev_ic_mac
ev_ic_savunma_ort = ev_toplam_yenen / ev_ic_mac
dep_dis_hucum_ort = dep_toplam_gol / dep_dis_mac
dep_dis_savunma_ort = dep_toplam_yenen / dep_dis_mac

ev_son5_hucum_ort = ev_son5_attigi / 5
ev_son5_savunma_ort = ev_son5_yedigi / 5
dep_son5_hucum_ort = dep_son5_attigi / 5
dep_son5_savunma_ort = dep_son5_yedigi / 5

ev_genel_hucum_ort = ev_genel_attigi / ev_genel_mac
ev_genel_savunma_ort = ev_genel_yedigi / ev_genel_mac
dep_genel_hucum_ort = dep_genel_attigi / dep_genel_mac
dep_genel_savunma_ort = dep_genel_yedigi / dep_genel_mac

ev_dinamik_ic_hucum = (ev_ic_hucum_ort * sezon_agirligi) + (ev_son5_hucum_ort * form_agirligi)
ev_dinamik_ic_savunma = (ev_ic_savunma_ort * sezon_agirligi) + (ev_son5_savunma_ort * form_agirligi)
dep_dinamik_dis_hucum = (dep_dis_hucum_ort * sezon_agirligi) + (dep_son5_hucum_ort * form_agirligi)
dep_dinamik_dis_savunma = (dep_dis_savunma_ort * sezon_agirligi) + (dep_son5_savunma_ort * form_agirligi)

ev_dinamik_ic_savunma *= (1.0 - (ev_cs * 0.03))
dep_dinamik_dis_savunma *= (1.0 - (dep_cs * 0.03))

yarim_lig_ort = lig_ort_gol / 2
ev_ic_hucum_katsayi = ev_dinamik_ic_hucum / max(yarim_lig_ort, 0.1)
dep_dis_savunma_katsayi = dep_dinamik_dis_savunma / max(yarim_lig_ort, 0.1)
dep_dis_hucum_katsayi = dep_dinamik_dis_hucum / max(yarim_lig_ort, 0.1)
ev_ic_savunma_katsayi = ev_dinamik_ic_savunma / max(yarim_lig_ort, 0.1)

ev_genel_hucum_katsayi = ev_genel_hucum_ort / max(yarim_lig_ort, 0.1)
dep_genel_savunma_katsayi = dep_genel_savunma_ort / max(yarim_lig_ort, 0.1)
dep_genel_hucum_katsayi = dep_genel_hucum_ort / max(yarim_lig_ort, 0.1)
ev_genel_savunma_katsayi = ev_genel_savunma_ort / max(yarim_lig_ort, 0.1)

ev_ppg = ev_ic_puan / ev_ic_mac
dep_ppg = dep_dis_puan / dep_dis_mac
puan_orani = ev_ppg / max((ev_ppg + dep_ppg), 0.1)
puan_denge_carpan = 0.85 + (puan_orani * 0.30)

onem_farki = ev_onem - dep_onem
ev_onem_carpan = 1.0 + (onem_farki * 0.03)
dep_onem_carpan = 1.0 - (onem_farki * 0.03)
ev_kadro_cezasi = (ev_kritik_eksik * 0.22) + (ev_normal_eksik * 0.08)
dep_kadro_cezasi = (dep_kritik_eksik * 0.22) + (dep_normal_eksik * 0.08)

ev_ic_xg = (yarim_lig_ort * ev_ic_hucum_katsayi * dep_dis_savunma_katsayi)
dep_dis_xg = (yarim_lig_ort * dep_dis_hucum_katsayi * ev_ic_savunma_katsayi)

ev_genel_xg = (yarim_lig_ort * ev_genel_hucum_katsayi * dep_genel_savunma_katsayi)
dep_genel_xg = (yarim_lig_ort * dep_genel_hucum_katsayi * ev_genel_savunma_katsayi)

ev_xg = ((ev_ic_xg * 0.5) + (ev_genel_xg * 0.5)) * puan_denge_carpan * ev_onem_carpan - ev_kadro_cezasi
dep_xg = ((dep_dis_xg * 0.5) + (dep_genel_xg * 0.5)) * (2.0 - puan_denge_carpan) * dep_onem_carpan - dep_kadro_cezasi

ev_xg = max(ev_xg, 0.05)
dep_xg = max(dep_xg, 0.05)

# Simülasyon Hesaplamaları
ms1_olasilik, x_olasilik, ms2_olasilik = mac_simule_et(ev_xg, dep_xg)

# En Olası Matris Skor Projeksiyonu
en_yuksek_skor_p, tahmin_ev_skor, tahmin_dep_skor = 0, 0, 0
for i in range(6):
    for j in range(6):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if p > en_yuksek_skor_p:
            en_yuksek_skor_p, tahmin_ev_skor, tahmin_dep_skor = p, i, j

# Gelişmiş Alt/Üst ve KG Var Dağılım Hesaplama
ust_olasilik, kg_var_olasilik = 0.0, 0.0
for i in range(11):
    for j in range(11):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if (i + j) > 2.5: ust_olasilik += p
        if i > 0 and j > 0: kg_var_olasilik += p
ust_olasilik *= 100
kg_var_olasilik *= 100

# Hassas Değer (Value) Filtrelemesi
v_ms1 = ms1_olasilik - (100 / b_ms1)
v_x = x_olasilik - (100 / b_x)
v_ms2 = ms2_olasilik - (100 / b_ms2)

# --- 🎙️ VERİ TABANLI DETAYLI TEKNİK ANALİZ ÖZETİ ---
yorum_cumleleri = []

# 1. xG Yapay Zekâ Güç Dengesi Analizi
yorum_cumleleri.append(
    f"Lig genelindeki {lig_ort_gol:.2f} gol ortalaması parametre alınarak kurgulanan dinamik xG matrisinde; "
    f"{ev_sahibi} ekibinin teorik gol üretkenlik endeksi {ev_xg:.2f}, {deplasman} ekibinin ise {dep_xg:.2f} "
    f"seviyesinde ölçülmüştür. Bu durum, {ev_sahibi if ev_xg > dep_xg else deplasman} takımının hücum varyasyonlarında "
    f"matematiksel olarak daha efektif bir oyun sergileyebileceğine işaret etmektedir."
)

# 2. Kadro Derinliği ve Eksik Oyuncu Etki Korelasyonu
if (ev_kritik_eksik + ev_normal_eksik) > (dep_kritik_eksik + dep_normal_eksik):
    yorum_cumleleri.append(
        f"Ev sahibi ekipte saptanan {ev_kritik_eksik} kritik ve {ev_normal_eksik} rotasyonel eksiklik, "
        f"takımın ideal taktiksel esnekliğini ve saha içi senkronizasyonunu kısıtlama potansiyeline sahip. "
        f"Sistem bu eksikliklerin yaratabileceği zafiyeti ev sahibi xG çıktısında negatif yönde revize etmiştir."
    )
elif (dep_kritik_eksik + dep_normal_eksik) > (ev_kritik_eksik + ev_normal_eksik):
    yorum_cumleleri.append(
        f"Deplasman kadrosundaki {dep_kritik_eksik} kritik ve {dep_normal_eksik} tamamlayıcı oyuncu noksanlığı, "
        f"özellikle dış saha direnci ve geçiş savunması metriklerini olumsuz etkilemektedir. Algoritma, bu kadro "
        f"yıpranmasını deplasmanın direnç katsayısında kırılma olarak hesaplamıştır."
    )
else:
    yorum_cumleleri.append(
        "Her iki takımın eksik oyuncu yükleri ve kadro stabilizasyonları dengeli bir dağılım göstermekte, "
        "müsabakanın taktiksel derinliğinde kadro kaynaklı ekstra bir anomalilik öngörülmemektedir."
    )

# 3. Gol ve Tempo Eğilim Analizi
if ust_olasilik >= 58:
    yorum_cumleleri.append(
        f"Genişletilmiş Poisson havuzunda %{ust_olasilik:.1f} olasılıkla baskın bir 2.5 Üst trendi yakalanmıştır. "
        f"Takımların hücum performans katsayılarının lig ortalamasının üzerinde kalması, geçiş hücumlarının "
        f"yoğun yaşanacağı, yüksek tempolu ve bol pozisyonlu bir 90 dakikayı rasyonel kılmaktadır."
    )
elif ust_olasilik <= 42:
    yorum_cumleleri.append(
        f"Matematiksel modelleme, %{100-ust_olasilik:.1f} ihtimalle savunma katsayılarının hücum hatlarına ağır "
        f"basacağı, düşük tempolu ve kilitli bir oyun yapısına işaret ediyor. Takımların defansif reaksiyonları, "
        f"skorun kontrollü bir düzlemde kalma olasılığını ciddi oranda artırmaktadır."
    )
else:
    yorum_cumleleri.append(
        f"Müsabakanın toplam gol eğilimi %{ust_olasilik:.1f} ihtimalle 2.5 Üst baremine yakınsasa da, orta tempolu "
        f"ve dengeli bir stratejik paylaşıma daha yatkın durmaktadır. İlk golün zamanlaması oyunun karakteristiğini "
        f"doğrudan belirleyecektir."
    )

# 4. Karşılıklı Gol (KG) ve Defansif Sürdürülebilirlik Analizi
if kg_var_olasilik >= 58:
    yorum_cumleleri.append(
        f"Karşılıklı Gol Var (KG Var) ihtimalinin %{kg_var_olasilik:.1f} seviyesinde seyretmesi, iki takımın da "
        f"üçüncü bölgedeki bitiricilik kalitesini doğrular niteliktedir. Defansif zaafiyetler ve hücum iştahı, "
        f"skor tabelasının çift taraflı değişme olasılığını maksimuma yaklaştırıyor."
    )
elif kg_var_olasilik <= 42:
    yorum_cumleleri.append(
        f"Model, %{100-kg_var_olasilik:.1f} gibi yüksek bir oranda takımlardan en az birinin gol yollarında "
        f"tıkanabileceğini veya katı bir savunma bloğu kuracağını öngörüyor. Clean Sheet (Gol yememe) eğilimleri "
        f"bu müsabakada ana belirleyici faktör olabilir."
    )

# 5. Value (Değer) ve Bahis Stratejisi Korelasyonu
en_yuksek_v = max(v_ms1, v_x, v_ms2)
if en_yuksek_v > 0:
    hedef_pazar = "MS 1" if en_yuksek_v == v_ms1 else ("Beraberlik (X)" if en_yuksek_v == v_x else "MS 2")
    yorum_cumleleri.append(
        f"Bülten oranları ile yapay zekâ olasılık matrisi karşılaştırıldığında; piyasa fiyatlamasındaki en net sapma "
        f"ve matematiksel değer (Value: {en_yuksek_v:+.2f}) **{hedef_pazar}** pazarında tespit edilmiştir. "
        f"Uzun vadeli karlılık projeksiyonunda bu tip oran anomalilerine yönelmek kasa yönetimini optimize edecektir."
    )
else:
    yorum_cumleleri.append(
        "Mevcut bülten oranları ile modelin olasılık çıktıları arasında piyasayı manipüle edecek düzeyde radikal "
        "bir 'Value' farkı saptanmamıştır. Oranlar risk dağılımıyla büyük ölçüde örtüşmektedir."
    )

nihai_yorum = " ".join(yorum_cumleleri)

# --- KULLANICI DOSTU ANA PANEL ---
with sekme1:
    st.markdown('<div class="badge">SEZGİN GÖRMÜŞ VERİ ANALİZİ </div>', unsafe_allow_html=True)
    st.title("📊 Sezgin Görmüş Matematiksel Tahmin Robotu")
    st.write(f"Sistem stabilizasyonu tamamlandı. Profesyonel Ev xG: **{ev_xg:.2f}** | Profesyonel Deplasman xG: **{dep_xg:.2f}**")
    
    col_sol, col_sag = st.columns([11, 10])
    
    with col_sol:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎯 Profesyonel Olasılık Dağılımı")
        fig = go.Figure(data=[go.Pie(labels=['Ev Sahibi Galibiyeti (MS1)', 'Beraberlik (X)', 'Deplasman Galibiyeti (MS2)'], values=[ms1_olasilik, x_olasilik, ms2_olasilik], hole=.45, marker_colors=['#0d9488', '#d97706', '#e11d48'])])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10), height=250)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("⚽ Maç Öngörüsü")
        st.markdown(f'<div class="skor-box">🤖 EN YÜKSEK OLASILIKLI SKOR: {tahmin_ev_skor} - {tahmin_dep_skor}</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: st.metric(label="📈 2.5 Üst İhtimali", value=f"%{ust_olasilik:.1f}"); st.progress(int(ust_olasilik))
        with c2: st.metric(label="🔥 Karşılıklı Gol Var İhtimali", value=f"%{kg_var_olasilik:.1f}"); st.progress(int(kg_var_olasilik))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sag:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚨 Sinyal Odası (Value Oran Filtresi)")
        def sinyal_satiri(pazar, model_p, oran, value):
            renk = "signal-green" if value > 0 else "signal-red"
            durum = "DEĞERLİ ORAN 🔥" if value > 0 else "DEĞERSİZ ORAN ❌"
            st.markdown(f'<div style="padding: 12px 0; border-bottom: 1px solid #1e293b;"><div style="display:flex; justify-content:space-between;"><b>{pazar}</b><span class="{renk}">{durum}</span></div><div style="font-size:13px; color:#94a3b8; margin-top:4px;">Model İhtimali: %{model_p:.1f} | Bülten Oranı: {oran:.2f} | <b>Yapay Zekâ Değeri: <span class="{renk}">{value:+.2f}</span></b></div></div>', unsafe_allow_html=True)
        
        sinyal_satiri(f"{ev_sahibi} Galibiyeti (MS1)", ms1_olasilik, b_ms1, v_ms1)
        sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
        sinyal_satiri(f"{deplasman} Galibiyeti (MS2)", ms2_olasilik, b_ms2, v_ms2)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card" style="background: #0f172a;">', unsafe_allow_html=True)
        st.subheader("💡 Sezgin Görmüş Akıllı Strateji Raporu")
        
        en_iyi_bahis = "RİSKLİ MÜSABAKA (PAS) ⚠️"
        pazar_t = "Yok"
        
        if v_ms1 > 0 and ms1_olasilik >= 48: en_iyi_bahis, pazar_t = f"Maç Sonucu 1 ({ev_sahibi})", "Maç Sonucu"
        elif v_ms2 > 0 and ms2_olasilik >= 48: en_iyi_bahis, pazar_t = f"Maç Sonucu 2 ({deplasman})", "Maç Sonucu"
        elif v_x > 0 and x_olasilik >= 32: en_iyi_bahis, pazar_t = "Beraberlik (X)", "Maç Sonucu"
        elif ust_olasilik >= 60: en_iyi_bahis, pazar_t = "2.5 Üst", "2.5 Alt/Üst"
        elif ust_olasilik <= 40: en_iyi_bahis, pazar_t = "2.5 Alt", "2.5 Alt/Üst"
        elif kg_var_olasilik >= 60: en_iyi_bahis, pazar_t = "Karşılıklı Gol Var (KG Var)", "Karşılıklı Gol"
        elif kg_var_olasilik <= 40: en_iyi_bahis, pazar_t = "Karşılıklı Gol Yok (KG Yok)", "Karşılıklı Gol"
        
        st.markdown(f'<div style="color:#f59e0b; font-weight:bold; font-size:17px; margin-bottom: 10px;">🎯 STRATEJİK SEÇİM: {en_iyi_bahis}</div>', unsafe_allow_html=True)
        
        st.markdown("##### 🎙️ Veri Mühendisliği Taktik Analizi")
        st.markdown(f'<div class="yorum-box">{nihai_yorum}</div>', unsafe_allow_html=True)
        st.write("")
        
        if st.button("💾 Kusursuz Analizi Arşive Kaydet"):
            df_logs = pd.read_csv(DB_FILE)
            yeni = {
                "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"), "Ev Sahibi": ev_sahibi, "Deplasman": deplasman, 
                "Model_MS1": round(ms1_olasilik,1), "Model_X": round(x_olasilik,1), "Model_MS2": round(ms2_olasilik,1), 
                "Model_Ust": round(ust_olasilik,1), "Model_KGVar": round(kg_var_olasilik,1),
                "Onerilen_Bahis": en_iyi_bahis, "Pazar_Tipi": pazar_t, "Sonuc": "Bekliyor"
            }
            df_logs = pd.concat([df_logs, pd.DataFrame([yeni])], ignore_index=True)
            df_logs.to_csv(DB_FILE, index=False)
            st.success("Kusursuz veri analizi arşive mühürlendi kanka!")
        st.markdown('</div>', unsafe_allow_html=True)

# --- SEKME 2: ARŞİV PANELİ ---
with sekme2:
    st.markdown('<div class="badge">SEZGİN GÖRMÜŞ ANALİZ MERKEZİ v7.0</div>', unsafe_allow_html=True)
    st.title("🗂️ Profesyonel Analiz Arşivi & Performans Takip Laboratuvarı")
    
    df_logs = pd.read_csv(DB_FILE)
    
    if len(df_logs) == 0:
        st.info("Arşiv veritabanı henüz boş kanka. İlk analizini kaydedince burası şenlenecek!")
    else:
        toplam_mac = len(df_logs)
        kazananlar = len(df_logs[df_logs["Sonuc"] == "KAZANDI ✅"].shape[0]) if "KAZANDI ✅" in df_logs["Sonuc"].values else 0
        kazananlar = len(df_logs[df_logs["Sonuc"] == "KAZANDI ✅"])
        kaybedenler = len(df_logs[df_logs["Sonuc"] == "KAYBETTİ ❌"])
        bekleyenler = len(df_logs[df_logs["Sonuc"] == "Bekliyor"])
        
        sonuclanmis = kazananlar + kaybedenler
        basari_orani = (kazananlar / sonuclanmis * 100) if sonuclanmis > 0 else 0.0
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1: st.metric(label="📊 Toplam Analiz Edilen Müsabaka", value=toplam_mac)
        with m_col2: st.metric(label="🔥 Model Başarı Endeksi", value=f"%{basari_orani:.1f}")
        with m_col3: st.metric(label="⏳ Sonuç Bekleyen Tahminler", value=bekleyenler)
        with m_col4: st.metric(label="🎯 İsabetli / Yatan Müsabaka", value=f"{kazananlar} / {kaybedenler}")
            
        st.write("---")
        st.markdown("### 🔍 Akıllı Filtreleme ve Arama")
        f_c1, f_c2 = st.columns([2, 1])
        with f_c1:
            arama_terimi = st.text_input("Takım Adına Göre Arşivde Ara...", "").lower()
        with f_c2:
            durum_filtresi = st.selectbox("Duruma Göre Süz", ["Hepsi", "KAZANDI ✅", "KAYBETTİ ❌", "Bekliyor"])
            
        df_goster = df_logs.copy()
        if arama_terimi:
            df_goster = df_goster[df_goster["Ev Sahibi"].str.lower().str.contains(arama_terimi) | df_goster["Deplasman"].str.lower().str.contains(arama_terimi)]
        if durum_filtresi != "Hepsi":
            df_goster = df_goster[df_goster["Sonuc"] == durum_filtresi]
            
        st.dataframe(df_goster, use_container_width=True)

        # --- ARŞİV GÜNCELLEME VE SİLME ODASI ---
        st.markdown("### ⚙️ Maç Durumu Güncelleme ve Yönetim")
        if len(df_goster) > 0:
            seçilen_index = st.selectbox("Durumunu Güncellemek İstediğin Maçın Satır Numarası (Index):", df_goster.index)
            yeni_durum = st.selectbox("Yeni Sonuç Değeri:", ["Bekliyor", "KAZANDI ✅", "KAYBETTİ ❌"])
            
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                if st.button("🔄 Durumu Güncelle"):
                    df_logs.at[seçilen_index, "Sonuc"] = yeni_durum
                    df_logs.to_csv(DB_FILE, index=False)
                    st.success("Maç sonucu başarıyla güncellendi kanka!")
                    st.rerun()
            with c_g2:
                if st.button("🗑️ Bu Analizi Veritabanından Sil"):
                    df_logs = df_logs.drop(seçilen_index).reset_index(drop=True)
                    df_logs.to_csv(DB_FILE, index=False)
                    st.warning("Analiz kaydı arşivden tamamen silindi!")
                    st.rerun()

# --- SEKME 3: KUPON ODASI ---
with sekme3:
    st.markdown('<div class="badge">🎟️ KUPON YÖNETİM MERKEZİ v7.0</div>', unsafe_allow_html=True)
    st.title("🎟️ Kupon Odası & Kasa Takip Laboratuvarı")
    st.write("Mühürlenen maç analizlerinizle oluşturduğunuz kuponları buraya kaydederek kasa yönetiminizi profesyonelce takip edin.")
    
    df_kuponlar = pd.read_csv(KUPON_DB_FILE)
    df_logs = pd.read_csv(DB_FILE)
    
    k_col1, k_col2 = st.columns([1, 1])
    
    with k_col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("➕ Yeni Kupon Mühürle")
        
        if len(df_logs) == 0:
            st.warning("Kupona maç ekleyebilmek için önce Analiz Ekranından maç kaydedip arşive eklemelisin kanka.")
        else:
            # Aktif veya bekleyen maç listesinden seçim yaptırıyoruz
            mac_secenekleri = df_logs["Ev Sahibi"] + " vs " + df_logs["Deplasman"] + " (" + df_logs["Onerilen_Bahis"] + ")"
            secilen_maclar = st.multiselect("Kupona Dahil Edilecek Maçları Seç kanka:", mac_secenekleri)
            
            yatirilan_para = st.number_input("Yatırılan Tutar (TL):", min_value=10, value=100, step=10)
            kupon_durumu = st.selectbox("Kupon Güncel Durumu:", ["Bekliyor", "TUTTU 🎉", "YATTI ❌"])
            
            if st.button("🎟️ Kuponu Veritabanına İşle"):
                if len(secilen_maclar) == 0:
                    st.error("En az 1 maç seçmelisin kanka!")
                else:
                    df_kp = pd.read_csv(KUPON_DB_FILE)
                    yeni_id = f"KPN-{len(df_kp) + 1001}"
                    maclar_str = " | ".join(secilen_maclar)
                    
                    yeni_kupon = {
                        "Kupon_ID": yeni_id,
                        "Mac_Detaylari": maclar_str,
                        "Yatirilan_Tutar": yatirilan_para,
                        "Durum": kupon_durumu,
                        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    df_kp = pd.concat([df_kp, pd.DataFrame([yeni_kupon])], ignore_index=True)
                    df_kp.to_csv(KUPON_DB_FILE, index=False)
                    st.success(f"{yeni_id} Kimlikli Kupon Kasaya Mühürlendi!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with k_col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📊 Canlı Kupon Durum Analizi")
        if len(df_kuponlar) == 0:
            st.info("Henüz kasaya mühürlenmiş kupon bulunmuyor kanka.")
        else:
            toplam_k = len(df_kuponlar)
            tutan_k = len(df_kuponlar[df_kuponlar["Durum"] == "TUTTU 🎉"])
            yatan_k = len(df_kuponlar[df_kuponlar["Durum"] == "YATTI ❌"])
            toplam_yatirilan = df_kuponlar["Yatirilan_Tutar"].sum()
            
            st.write(f"📊 Toplam Alınan Kupon: **{toplam_k}**")
            st.write(f"💰 Toplam Yatırılan Finans: **{toplam_yatirilan} TL**")
            st.write(f"✅ Tutan: **{tutan_k}** | ❌ Yatan: **{yatan_k}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.write("---")
    st.markdown("### 📋 Mevcut Kupon Listesi")
    if len(df_kuponlar) > 0:
        st.dataframe(df_kuponlar, use_container_width=True)
        
        st.markdown("#### ⚙️ Kupon Durumu Revize Odası")
        k_secilen_id = st.selectbox("Durumunu Değiştirmek İstediğin Kuponun ID'si:", df_kuponlar["Kupon_ID"].values)
        k_yeni_durum = st.selectbox("Yeni Kupon Durumu Seç:", ["Bekliyor", "TUTTU 🎉", "YATTI ❌"])
        
        ck_g1, ck_g2 = st.columns(2)
        with ck_g1:
            if st.button("🔄 Kupon Durumunu Güncelle"):
                idx = df_kuponlar[df_kuponlar["Kupon_ID"] == k_secilen_id].index[0]
                df_kuponlar.at[idx, "Durum"] = k_yeni_durum
                df_kuponlar.to_csv(KUPON_DB_FILE, index=False)
                st.success("Kupon durumu güncellendi!")
                st.rerun()
        with ck_g2:
            if st.button("🗑️ Bu Kuponu Kasadan Sil"):
                df_kuponlar = df_kuponlar[df_kuponlar["Kupon_ID"] != k_secilen_id].reset_index(drop=True)
                df_kuponlar.to_csv(KUPON_DB_FILE, index=False)
                st.warning("Kupon sistemden imha edildi!")
                st.rerun()
