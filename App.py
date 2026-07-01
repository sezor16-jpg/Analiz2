import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
from datetime import datetime
import numpy as np

# --- SEZGİN GÖRMÜŞ AI PRO v8.0 SAYFA AYARLARI ---
st.set_page_config(page_title="Sezgin Görmüş Veri Analizi v8.0", page_icon="🔮", layout="wide")

DB_FILE = "analiz_gunlugu.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=[
        "Tarih", "Ev Sahibi", "Deplasman", 
        "Model_MS1", "Model_X", "Model_MS2", "Model_Ust", "Model_KGVar",
        "Onerilen_Bahis", "Pazar_Tipi", "Sonuc"
    ])
    df.to_csv(DB_FILE, index=False)

# Yeni Kupon DB Yapısı (Eskisini otomatik uyarlar)
KUPON_DB_FILE = "kuponlar_gunlugu.csv"
if not os.path.exists(KUPON_DB_FILE):
    pd.DataFrame(columns=["Kupon_ID", "Mac_IDleri", "Yatirilan_Tutar", "Durum", "Tarih"]).to_csv(KUPON_DB_FILE, index=False)
else:
    temp_df = pd.read_csv(KUPON_DB_FILE)
    if "Mac_IDleri" not in temp_df.columns:
        pd.DataFrame(columns=["Kupon_ID", "Mac_IDleri", "Yatirilan_Tutar", "Durum", "Tarih"]).to_csv(KUPON_DB_FILE, index=False)

# --- 🤖 KUPON OTOMASYON MOTORU ---
def kuponlari_otomatik_guncelle():
    df_k = pd.read_csv(KUPON_DB_FILE)
    df_l = pd.read_csv(DB_FILE)
    
    if len(df_k) == 0: return
    
    for idx, row in df_k.iterrows():
        if pd.isna(row["Mac_IDleri"]): continue
        
        mac_idler = str(row["Mac_IDleri"]).split(",")
        durumlar = []
        
        for mid in mac_idler:
            try:
                m_idx = int(mid)
                if m_idx in df_l.index:
                    durumlar.append(df_l.at[m_idx, "Sonuc"])
                else:
                    durumlar.append("SİLİNMİŞ MAÇ")
            except:
                pass
                
        if "KAYBETTİ ❌" in durumlar:
            df_k.at[idx, "Durum"] = "YATTI ❌"
        elif all(d == "KAZANDI ✅" for d in durumlar) and len(durumlar) > 0:
            df_k.at[idx, "Durum"] = "TUTTU 🎉"
        else:
            df_k.at[idx, "Durum"] = "Bekliyor"
            
    df_k.to_csv(KUPON_DB_FILE, index=False)

# --- 📐 KUSURSUZ POISSON MOTORU ---
def poisson_olasilik(k, lmbda):
    if lmbda <= 0: return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)
    
def mac_simule_et(ev_gol_beklentisi, dep_gol_beklentisi):
    simulasyon_sayisi = 50000
    ev_goller = np.random.poisson(ev_gol_beklentisi, simulasyon_sayisi)
    dep_goller = np.random.poisson(dep_gol_beklentisi, simulasyon_sayisi)
    
    ms1_sayisi = np.sum(ev_goller > dep_goller)
    beraberlik_sayisi = np.sum(ev_goller == dep_goller)
    ms2_sayisi = np.sum(ev_goller < dep_goller)
    
    ms1_yuzde = (ms1_sayisi / simulasyon_sayisi) * 100
    x_yuzde = (beraberlik_sayisi / simulasyon_sayisi) * 100
    ms2_yuzde = (ms2_sayisi / simulasyon_sayisi) * 100
    
    return ms1_yuzde, x_yuzde, ms2_yuzde


def ai_yorum_uret(ev_sahibi, deplasman, ev_xg, dep_xg, ms1_olasilik, ms2_olasilik, en_iyi_bahis):
    # 1. Hücum Analizi Katmanı
    fark = ev_xg - dep_xg
    if fark > 0.8:
        durum = f"{ev_sahibi} hücumda çok agresif ve dominasyonu elinde tutan bir yapıda."
    elif fark > 0.3:
        durum = f"{ev_sahibi} oyunu kontrol eden ve skor üretmeye yakın taraf."
    elif fark < -0.8:
        durum = f"{deplasman} takımı sahaya çok daha net ve tehlikeli ayaklarla çıkıyor."
    elif fark < -0.3:
        durum = f"{deplasman} oyunu kendi lehine çevirebilecek potansiyele sahip."
    else:
        durum = "İki takım da birbirine karşı üstünlük kurmakta zorlanacak, tam bir satranç maçı."

    # 2. Olasılık Şiddeti Katmanı
    en_yuksek = max(ms1_olasilik, ms2_olasilik)
    if en_yuksek > 65:
        guven_seviyesi = "Modelimizin bu sonuca olan güveni oldukça yüksek."
    elif en_yuksek > 50:
        guven_seviyesi = "Veriler, sonucun bu yönde evrilme ihtimalini güçlü kılıyor."
    else:
        guven_seviyesi = "Maçın sonucu kağıt üzerinde oldukça yakın görünüyor, dikkatli olunmalı."

    # 3. Bahis Tipi Katmanı
    if "MS1" in en_iyi_bahis:
        bahis_yorum = "Ev sahibi avantajı ve güncel form durumu bu tercihi mantıklı kılıyor."
    elif "MS2" in en_iyi_bahis:
        bahis_yorum = "Deplasman ekibinin kontra atak verimliliği bu seçimi öne çıkarıyor."
    elif "Üst" in en_iyi_bahis:
        bahis_yorum = "Savunma disiplinlerinden ziyade, skor odaklı bir mücadele beklentisi var."
    elif "Alt" in en_iyi_bahis:
        bahis_yorum = "Kilitli kalması muhtemel ve taktiksel disiplinin ön planda olduğu bir senaryo."
    elif "KG" in en_iyi_bahis:
        bahis_yorum = "İki ekibin de savunma zafiyetleri göz önüne alındığında gol yollarının açık olması muhtemel."
    else:
        bahis_yorum = "Sistemimiz bu markette bir değer (value) tespit etti."

    return f"🤖 {durum} {guven_seviyesi} {bahis_yorum} ({en_iyi_bahis} seçimi verilerle destekleniyor.)"
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
    "Almanya Bundesliga 2": [
        "Hertha BSC", "Arminia Bielefeld", "VfL Bochum", "Eintracht Braunschweig", "Energie Cottbus",
        "Darmstadt 98", "Dinamo Dresden", "Greuther Fürth", "Hannover 96", "1. FC Heidenheim",
        "1. FC Kaiserslautern", "Karlsruher SC", "Holstein Kiel", "1. FC Magdeburg", "1. FC Nürnberg",
        "VfL Osnabrück", "FC St. Pauli", "VfL Wolfsburg"
    ],
    "Hollanda Eredivisie": [
        "ADO Den Haag", "Ajax", "AZ", "Cambuur", "Excelsior",
        "Feyenoord", "Fortuna Sittard", "Go Ahead Eagles", "Groningen", "Heerenveen",
        "N.E.C.", "PEC Zwolle", "PSV", "Sparta Rotterdam", "Telstar",
        "Twente", "Utrecht", "Willem II"
    ],
    "Hollanda Eerste Divisie": [
        "Jong Ajax", "Almere City", "Jong AZ", "De Graafschap", "Den Bosch",
        "FC Dordrecht", "FC Eindhoven", "Emmen", "Helmond", "Heracles",
        "Maastricht", "NAC Breda", "Jong PSV", "Waalwijk", "Roda",
        "TOP Oss", "Jong Utrecht", "Vitesse","Volendam","VVV Venlo"
    ],
    "Fransa Ligue 1": [
        "Angers", "Brest", "Le Mans", "Lens", "Lille",
        "Lorient", "Lyon", "Marseille", "Monaco", "Paris",
        "Paris Saint-Germain", "Rennes", "Strasbourg", "Toulouse", "Troyes",
        "", "", ""
    ],
    "Fransa Ligue 2": [
        "Annecy", "Boulogne", "Clermont", "Dijon", "Dunkerque",
        "Grenoble", "Guingamp", "Lavallois", "Metz", "Montpellier",
        "Nancy", "Nantes", "Pau", "Red Star", "Reims",
        "Rodez", "St Etienne", "Sochaux"
    ]
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
    
    /* YENİ KUPON KART TASARIMLARI */
    .k-card { border-radius: 12px; padding: 16px; margin-bottom: 16px; background-color: #0f172a; border: 1px solid #334155; position: relative; overflow: hidden; }
    .k-tuttu { background: #10b981; }
    .k-yatti { background: #ef4444; }
    .k-bekliyor { background: #f59e0b; }
    .m-satir { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e293b; font-size: 16px; }
    .m-satir:last-child { border-bottom: none; }
    </style>
""", unsafe_allow_html=True)

sekme1, sekme2, sekme3 = st.tabs(["📊 Analiz Ekranı", "🗂️ Arşiv Paneli", "🎟️ Kupon Odası"])

# --- SIDEBAR (AYNEN KORUNDU) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 20px;'>📋 VERİ SEVİYESİ</h2>", unsafe_allow_html=True)
st.sidebar.markdown("### 🏟️ Müsabaka Seçim Odası")

lig_listesi = list(LIG_VERITABANI.keys()) + ["🌍 Diğer / Listede Olmayan Lig"]
secilen_lig = st.sidebar.selectbox("Ligi Seç kanka:", lig_listesi)

if secilen_lig == "🌍 Diğer / Listede Olmayan Lig":
    ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adı:", "Ev Sahibi")
else:
    ev_secenekleri = ["✍️ Kendim Yazacağım..."] + LIG_VERITABANI[secilen_lig]
    ev_secim = st.sidebar.selectbox("Ev Sahibi Takım:", ev_secenekleri, index=1)
    ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adını Gir kanka:", "Ev Sahibi") if ev_secim == "✍️ Kendim Yazacağım..." else ev_secim

if secilen_lig == "🌍 Diğer / Listede Olmayan Lig":
    deplasman = st.sidebar.text_input("Deplasman Takım Adı:", "Deplasman")
else:
    dep_secenekleri = ["✍️ Kendim Yazacağım..."] + LIG_VERITABANI[secilen_lig]
    dep_secim = st.sidebar.selectbox("Deplasman Takımı:", dep_secenekleri, index=2)
    deplasman = st.sidebar.text_input("Deplasman Takım Adını Gir kanka:", "Deplasman") if dep_secim == "✍️ Kendim Yazacağım..." else dep_secim

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

# Tüm gol ihtimallerini tek seferde hesaplayalım (Daha optimize)
olasiliklar = []
for i in range(6): # Ev sahibi gol sayısı
    for j in range(6): # Deplasman gol sayısı
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        olasiliklar.append({'toplam': i + j, 'p': p})

# Yüzde hesaplamaları
def get_p(alt_limit):
    return sum([item['p'] for item in olasiliklar if item['toplam'] < alt_limit]) * 100

gol_alt_1_5, gol_ust_1_5 = get_p(1.5), 100 - get_p(1.5)
gol_alt_2_5, gol_ust_2_5 = get_p(2.5), 100 - get_p(2.5)
gol_alt_3_5, gol_ust_3_5 = get_p(3.5), 100 - get_p(3.5)

ms1_olasilik, x_olasilik, ms2_olasilik = mac_simule_et(ev_xg, dep_xg)

en_yuksek_skor_p, tahmin_ev_skor, tahmin_dep_skor = 0, 0, 0
for i in range(10):
    for j in range(10):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if p > en_yuksek_skor_p:
            en_yuksek_skor_p, tahmin_ev_skor, tahmin_dep_skor = p, i, j

ust_olasilik, kg_var_olasilik = 0.0, 0.0
for i in range(11):
    for j in range(11):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if (i + j) > 2.5: ust_olasilik += p
        if i > 0 and j > 0: kg_var_olasilik += p
ust_olasilik *= 100
kg_var_olasilik *= 100

v_ms1 = ms1_olasilik - (100 / b_ms1)
v_x = x_olasilik - (100 / b_x)
v_ms2 = ms2_olasilik - (100 / b_ms2)

st.markdown("""
<style>
    /* Premium Kartlar için zorlayıcı stil */
    div[data-testid="stVerticalBlock"] > div:has(.premium-card) {
        background: transparent !important;
    }
    
    .premium-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 20px !important;
        border: 1px solid #334155 !important;
        color: white !important;
    }
    
    .badge {
        background: #3D9DF3 !important;
        color: #fff !important;
        padding: 5px 15px !important;
        border-radius: 20px !important;
        font-weight: bold !important;
        display: inline-block !important;
        margin-bottom: 15px !important;
    }
    
    .signal-green { color: #10b981 !important; font-weight: bold; }
    .signal-red { color: #ef4444 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SEKME 1: ANALİZ EKRANI ---
with sekme1:
    # 1. STRATEJİ MANTIĞINI TANIMLA (NameError'ı engellemek için yukarı taşıdık)
    en_iyi_bahis = "RİSKLİ MÜSABAKA (PAS) ⚠️"
    pazar_t = "Yok"
    if v_ms1 > 0 and ms1_olasilik >= 48: en_iyi_bahis, pazar_t = f"Maç Sonucu 1 ({ev_sahibi})", "Maç Sonucu"
    elif v_ms2 > 0 and ms2_olasilik >= 48: en_iyi_bahis, pazar_t = f"Maç Sonucu 2 ({deplasman})", "Maç Sonucu"
    elif v_x > 0 and x_olasilik >= 32: en_iyi_bahis, pazar_t = "Beraberlik (X)", "Maç Sonucu"
    elif ust_olasilik >= 60: en_iyi_bahis, pazar_t = "2.5 Üst", "2.5 Alt/Üst"
    elif ust_olasilik <= 40: en_iyi_bahis, pazar_t = "2.5 Alt", "2.5 Alt/Üst"
    elif kg_var_olasilik >= 60: en_iyi_bahis, pazar_t = "Karşılıklı Gol Var (KG Var)", "Karşılıklı Gol"
    elif kg_var_olasilik <= 40: en_iyi_bahis, pazar_t = "Karşılıklı Gol Yok (KG Yok)", "Karşılıklı Gol"

    st.markdown('<div class="badge">SEZGİN GÖRMÜŞ VERİ ANALİZİ</div>', unsafe_allow_html=True)
    st.title("📊 Sezgin Görmüş Matematiksel Tahmin Robotu")
    
    # Metrikler artık hata vermez
    m1, m2, m3 = st.columns(3)
    m1.metric("Ev xG", f"{ev_xg:.2f}")
    m2.metric("Dep xG", f"{dep_xg:.2f}")
    m3.metric("Tahmin", en_iyi_bahis.split(" (")[0])
    
    col_sol, col_sag = st.columns([11, 10])
    
    with col_sol:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎯 Profesyonel Olasılık Dağılımı")
        
        grafik_tipi = st.radio("Görünüm:", ["Maç Sonucu", "1.5 Alt/Üst", "2.5 Alt/Üst", "3.5 Alt/Üst", "KG Durumu"], horizontal=True)
        
        if grafik_tipi == "Maç Sonucu":
            fig = go.Figure(data=[go.Pie(labels=['MS1', 'X', 'MS2'], values=[ms1_olasilik, x_olasilik, ms2_olasilik], hole=.4)])
        elif "Alt/Üst" in grafik_tipi:
            if "1.5" in grafik_tipi: vals = [gol_ust_1_5, gol_alt_1_5]
            elif "2.5" in grafik_tipi: vals = [gol_ust_2_5, gol_alt_2_5]
            else: vals = [gol_ust_3_5, gol_alt_3_5]
            fig = go.Figure(data=[go.Pie(labels=['Üst', 'Alt'], values=vals, hole=.4, marker_colors=['#0acc31', '#f53333'])])
        else:
            fig = go.Figure(data=[go.Pie(labels=['KG Var', 'KG Yok'], values=[kg_var_olasilik, 100-kg_var_olasilik], hole=.4, marker_colors=['#0acc31', '#f53333'])])

        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(t=30, b=30, l=30, r=30), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,uirevision='constant'))
        
        # KEY parametresi eklendi
        st.plotly_chart(fig, use_container_width=True, key=f"grafik_{grafik_tipi}")
        st.markdown('</div>', unsafe_allow_html=True)
        
       # OTOMATİK AI YORUM KARTI
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Maç Yorumu")
        
        # Yorumu burada üretiyoruz
        otomatik_yorum = ai_yorum_uret(ev_sahibi, deplasman, ev_xg, dep_xg, ms1_olasilik, ms2_olasilik, en_iyi_bahis)
        
        st.info(otomatik_yorum) # Mavi, şık bir kutuda yorumu basar
        st.markdown('</div>', unsafe_allow_html=True)
   
    with col_sag:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚨 Tahmin ve Sinyal Odası")
        
        # Sinyalleri iyileştiren geliştirilmiş fonksiyon
        def sinyal_satiri(pazar, model_p, oran, value):
            renk = "signal-green" if value > 0 else "signal-red"
            ikon = "✅" if value > 0 else "⚠️"
            # Görseli güçlendirmek için satırları biraz daha belirgin kıldık
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #1e293b;">
                <div>
                    <div style='font-weight: 600; color: #f8fafc;'>{pazar}</div>
                    <div style="font-size: 12px; color: #94a3b8;">Olasılık: %{model_p:.1f} | Oran: {oran}</div>
                </div>
                <div style="text-align: right;">
                    <div class='{renk}' style="font-size: 18px;">{value:+.2f}</div>
                    <div style="font-size: 12px; color: #64748b;">{ikon}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sinyalleri basıyoruz
        sinyal_satiri(f"MS 1 ({ev_sahibi})", ms1_olasilik, b_ms1, v_ms1)
        sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
        sinyal_satiri(f"MS 2 ({deplasman})", ms2_olasilik, b_ms2, v_ms2)
        
        st.markdown('</div>', unsafe_allow_html=True) # Sinyal kartı sonu
        
        # --- STRATEJİ VE KAYIT ---
        st.markdown("---")
        # Butonun daha dikkat çekici olması için kolon yapısı
        if st.button("💾 Analizi Arşive Mühürle 🎯", use_container_width=True):
            try:
                df_logs = pd.read_csv(DB_FILE)
                yeni = {
                    "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"), 
                    "Ev Sahibi": ev_sahibi, 
                    "Deplasman": deplasman,
                    "Sonuc": "Bekliyor",
                    "Model_MS1": round(ms1_olasilik, 1), # <-- EKLENDİ
                    "Model_X": round(x_olasilik, 1),     # <-- EKLENDİ
                    "Model_MS2": round(ms2_olasilik, 1), # <-- EKLENDİ
                    "Model_Ust": round(ust_olasilik, 1), # <-- EKLENDİ
                    "Model_KGVar": round(kg_var_olasilik, 1), # <-- EKLENDİ
                    "Onerilen_Bahis": en_iyi_bahis, 
                    "Pazar_Tipi": pazar_t,               # <-- EKLENDİ
                }
                df_logs = pd.concat([df_logs, pd.DataFrame([yeni])], ignore_index=True)
                df_logs.to_csv(DB_FILE, index=False)
                st.success("Veri başarıyla arşive işlendi kanka!")
            except Exception as e:
                st.error(f"Kayıt hatası: {e}")


# --- SEKME 2: ARŞİV PANELİ ---
with sekme2:
    st.markdown('<div class="badge">SEZGİN GÖRMÜŞ ANALİZ MERKEZİ v8.0</div>', unsafe_allow_html=True)
    st.title("🗂️ Profesyonel Analiz Arşivi")
    
    df_logs = pd.read_csv(DB_FILE)
    
    if len(df_logs) == 0:
        st.info("Arşiv veritabanı boş kanka. Önce maç analiz etmelisin!")
    else:
        kazananlar = len(df_logs[df_logs["Sonuc"] == "KAZANDI ✅"])
        kaybedenler = len(df_logs[df_logs["Sonuc"] == "KAYBETTİ ❌"])
        bekleyenler = len(df_logs[df_logs["Sonuc"] == "Bekliyor"])
        basari_orani = (kazananlar / (kazananlar + kaybedenler) * 100) if (kazananlar + kaybedenler) > 0 else 0.0
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1: st.metric(label="📊 Toplam Maç", value=len(df_logs))
        with m_col2: st.metric(label="🔥 Başarı Oranı", value=f"%{basari_orani:.1f}")
        with m_col3: st.metric(label="⏳ Bekleyenler", value=bekleyenler)
        with m_col4: st.metric(label="🎯 Tutan / Yatan", value=f"{kazananlar} / {kaybedenler}")
            
        st.write("---")
        df_goster = df_logs.copy()
        st.dataframe(df_goster, use_container_width=True)

        st.markdown("### ⚙️ Maç Durumu Güncelleme")
        if len(df_goster) > 0:
            seçilen_index = st.selectbox("Durumunu Güncellemek İstediğin Maçın Numarası (Index):", df_goster.index)
            yeni_durum = st.selectbox("Yeni Sonuç Değeri:", ["Bekliyor", "KAZANDI ✅", "KAYBETTİ ❌"])
            
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                if st.button("🔄 Durumu Güncelle ve Kuponlara İşle"):
                    df_logs.at[seçilen_index, "Sonuc"] = yeni_durum
                    df_logs.to_csv(DB_FILE, index=False)
                    kuponlari_otomatik_guncelle() # OTOMATİK KUPON KONTROLÜ
                    st.success("Maç sonucu güncellendi! Bekleyen kuponlar otomatik denetlendi kanka.")
                    st.rerun()
            with c_g2:
                if st.button("🗑️ Analizi Sil"):
                    df_logs = df_logs.drop(seçilen_index).reset_index(drop=True)
                    df_logs.to_csv(DB_FILE, index=False)
                    st.warning("Analiz kaydı silindi!")
                    st.rerun()


# --- SEKME 3: DİJİTAL KUPON ODASI ---
with sekme3:
    st.markdown('<div class="badge">🎟️ OTOMATİK KUPON MERKEZİ v8.0</div>', unsafe_allow_html=True)
    st.title("🎟️ Akıllı Kupon Odası & Kasa")

    df_kuponlar = pd.read_csv(KUPON_DB_FILE)
    df_logs = pd.read_csv(DB_FILE)

    # --- KRİTİK FİLTRELEME ---
    # Sadece 'Sonuc' sütunu 'Bekliyor' olan maçları filtrele
    aktif_maclar = df_logs[df_logs["Sonuc"] == "Bekliyor"]

    k_col1, k_col2 = st.columns([1, 1])

    with k_col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("➕ Yeni Kupon Mühürle")

        if len(aktif_maclar) == 0:
            st.warning("Şu an analiz edilmiş ve sonucu belli olmayan (Bekleyen) maç yok kanka.")
        else:
            # Filtrelenmiş aktif_maclar üzerinden seçim listesi oluşturduk
            mac_secenekleri = aktif_maclar.index.astype(str) + " - " + aktif_maclar["Ev Sahibi"] + " vs " + aktif_maclar["Deplasman"] + " (" + aktif_maclar["Onerilen_Bahis"] + ")"
            secilen_maclar = st.multiselect("Kupona Eklenecek Maçlar:", mac_secenekleri)

            yatirilan_para = st.number_input("Yatırılan Tutar (TL):", min_value=10, value=100, step=10)

            if st.button("🎟️ Kuponu Sisteme Göm"):
                if len(secilen_maclar) == 0:
                    st.error("En az 1 maç seçmelisin kanka!")
                else:
                    secilen_idler = [m.split(" - ")[0] for m in secilen_maclar]
                    mac_idler_str = ",".join(secilen_idler)

                    yeni_id = f"KPN-{len(df_kuponlar) + 1001}"
                    yeni_kupon = {
                        "Kupon_ID": yeni_id,
                        "Mac_IDleri": mac_idler_str,
                        "Yatirilan_Tutar": yatirilan_para,
                        "Durum": "Bekliyor",
                        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    df_kuponlar = pd.concat([df_kuponlar, pd.DataFrame([yeni_kupon])], ignore_index=True)
                    df_kuponlar.to_csv(KUPON_DB_FILE, index=False)
                    kuponlari_otomatik_guncelle()
                    st.success(f"{yeni_id} Kasaya Mühürlendi!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with k_col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📊 Kasa Analizi")
        if len(df_kuponlar) > 0:
            tutan_k = len(df_kuponlar[df_kuponlar["Durum"] == "TUTTU 🎉"])
            yatan_k = len(df_kuponlar[df_kuponlar["Durum"] == "YATTI ❌"])
            toplam_yatirilan = df_kuponlar["Yatirilan_Tutar"].sum()

            st.metric(label="💰 Toplam Yatırılan Bakiye", value=f"{toplam_yatirilan} TL")
            c_k1, c_k2 = st.columns(2)
            c_k1.metric(label="✅ Tutan Kupon", value=tutan_k)
            c_k2.metric(label="❌ Yatan Kupon", value=yatan_k)
        else:
            st.info("Henüz mühürlenmiş kupon yok.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    st.markdown("### 📋 Aktif Dijital Kupon Fişleri")

    if len(df_kuponlar) > 0:
        satir_sayisi = 0
        kolonlar = st.columns(2)

        for idx, row in df_kuponlar.iloc[::-1].iterrows():
            c = kolonlar[satir_sayisi % 2]
            satir_sayisi += 1

            durum_sinifi = "k-bekliyor"
            renk = "#ffffff"
            if row["Durum"] == "TUTTU 🎉":
                durum_sinifi = "k-tuttu"
            elif row["Durum"] == "YATTI ❌":
                durum_sinifi = "k-yatti"

            with c:
                st.markdown(f'''
                    <div class="k-card {durum_sinifi}" style="padding: 15px; margin-bottom: 15px; border-radius: 8px;">
                        <div style="margin-bottom:10px;">
                            <div style="font-weight:bold; font-size:16px;">{row["Kupon_ID"]}</div>
                            <div style="color:{renk}; font-weight:bold;">{row["Durum"]}</div>
                            <div style="font-size:14px; color:#ffffff; margin-bottom:15px;">Yatırılan Tutar: {row["Yatirilan_Tutar"]} TL</div>
                        </div>
                ''', unsafe_allow_html=True)

                if pd.notna(row["Mac_IDleri"]):
                    idler = str(row["Mac_IDleri"]).split(",")
                    for mid in idler:
                        try:
                            m_idx = int(mid)
                            if m_idx in df_logs.index:
                                m_isim = f"{df_logs.at[m_idx, 'Ev Sahibi']} - {df_logs.at[m_idx, 'Deplasman']}"
                                m_sonuc = df_logs.at[m_idx, 'Sonuc']
                                m_renk = "#e2e8f0"
                                ikon = "⏳"
                                if m_sonuc == "KAZANDI ✅": m_renk = "#10b981"; ikon = "✅"
                                elif m_sonuc == "KAYBETTİ ❌": m_renk = "#ef4444"; ikon = "❌"

                                st.markdown(f'''
                                    <div class="m-satir" style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                        <span>{m_isim}</span><span style="color:{m_renk}; font-weight:bold;">{ikon}</span>
                                    </div>
                                ''', unsafe_allow_html=True)
                        except:
                            pass

                if st.button(f"🗑️ {row['Kupon_ID']} Sil", key=f"sil_{row['Kupon_ID']}"):
                    df_kuponlar = df_kuponlar.drop(idx).reset_index(drop=True)
                    df_kuponlar.to_csv(KUPON_DB_FILE, index=False)
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
