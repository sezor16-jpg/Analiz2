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

# Eğer kupon veritabanı yoksa otomatik oluşturuyoruz
if not os.path.exists(KUPON_DB_FILE):
    pd.DataFrame(columns=["Kupon_ID", "Mac_IDleri", "Mac_Detaylari", "Yatirilan_Tutar", "Durum"]).to_csv(KUPON_DB_FILE, index=False)
# --- 📐 KUSURSUZ POISSON MOTORU (PROFESYONEL LİMİT: 11) ---

def poisson_olasilik(k, lmbda):
    if lmbda <= 0: return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)
    
def mac_simule_et(ev_gol_beklentisi, dep_gol_beklentisi):
    # 100.000 maçlık devasa simülasyon havuzu
    simulasyon_sayisi = 100000
    
    # Bilgisayar, verdiğin xG değerlerine göre arkada rastgele 100.000 maçlık gol sayıları üretiyor
    ev_goller = np.random.poisson(ev_gol_beklentisi, simulasyon_sayisi)
    dep_goller = np.random.poisson(dep_gol_beklentisi, simulasyon_sayisi)
    
    # 100.000 maçın sonuçları çeteleye işleniyor
    ms1_sayisi = np.sum(ev_goller > dep_goller)
    beraberlik_sayisi = np.sum(ev_goller == dep_goller)
    ms2_sayisi = np.sum(ev_goller < dep_goller)
    
    # Yüzdesel oranlar hesaplanıp ana panele gönderiliyor
    ms1_yuzde = (ms1_sayisi / simulasyon_sayisi) * 100
    x_yuzde = (beraberlik_sayisi / simulasyon_sayisi) * 100
    ms2_yuzde = (ms2_sayisi / simulasyon_sayisi) * 100
    
    return ms1_yuzde, x_yuzde, ms2_yuzde

# --- PREMIUM GÖRSEL ARYÜZ TASARIMI (CSS) ---
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

sekme1, sekme2, sekme3 = st.tabs(["📊 Analiz Ekranı", "🗂️ Arşiv Panelini", "🎟️ Kupon Odası"])

# --- SIDEBAR (KULLANICI DOSTU GELİŞMİŞ PANEL) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 20px;'>📋 VERİ SEVİYESİ</h2>", unsafe_allow_html=True)
st.sidebar.markdown("### 🏟️ Müsabaka Bilgileri")
ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adı", "Manchester City")
deplasman = st.sidebar.text_input("Deplasman Takımı Adı", "Liverpool")

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



# --- ⚙️ SEZGİN GÖRMÜŞ KUSURSUZ MATHEMATICAL MATRIX MOTORU ---
# 1. İç/Dış Saha Temel Hücum ve Savunma Metrikleri
ev_ic_hucum_ort = ev_toplam_gol / ev_ic_mac
ev_ic_savunma_ort = ev_toplam_yenen / ev_ic_mac
dep_dis_hucum_ort = dep_toplam_gol / dep_dis_mac
dep_dis_savunma_ort = dep_toplam_yenen / dep_dis_mac

# 2. Form (Son 5 Maç) Metrikleri
ev_son5_hucum_ort = ev_son5_attigi / 5
ev_son5_savunma_ort = ev_son5_yedigi / 5
dep_son5_hucum_ort = dep_son5_attigi / 5
dep_son5_savunma_ort = dep_son5_yedigi / 5

# 3. Genel Lig Performans Metrikleri
ev_genel_hucum_ort = ev_genel_attigi / ev_genel_mac
ev_genel_savunma_ort = ev_genel_yedigi / ev_genel_mac
dep_genel_hucum_ort = dep_genel_attigi / dep_genel_mac
dep_genel_savunma_ort = dep_genel_yedigi / dep_genel_mac

# 4. Dinamik Süreç Entegrasyonu (Sezon vs Form Ağırlığı)
ev_dinamik_ic_hucum = (ev_ic_hucum_ort * sezon_agirligi) + (ev_son5_hucum_ort * form_agirligi)
ev_dinamik_ic_savunma = (ev_ic_savunma_ort * sezon_agirligi) + (ev_son5_savunma_ort * form_agirligi)
dep_dinamik_dis_hucum = (dep_dis_hucum_ort * sezon_agirligi) + (dep_son5_hucum_ort * form_agirligi)
dep_dinamik_dis_savunma = (dep_dis_savunma_ort * sezon_agirligi) + (dep_son5_savunma_ort * form_agirligi)

# 5. Clean Sheet Filtresi
ev_dinamik_ic_savunma *= (1.0 - (ev_cs * 0.03))
dep_dinamik_dis_savunma *= (1.0 - (dep_cs * 0.03))

# 6. Lig Başı Oranlama Katsayıları (İç/Dış Odaklı)
yarim_lig_ort = lig_ort_gol / 2
ev_ic_hucum_katsayi = ev_dinamik_ic_hucum / max(yarim_lig_ort, 0.1)
dep_dis_savunma_katsayi = dep_dinamik_dis_savunma / max(yarim_lig_ort, 0.1)
dep_dis_hucum_katsayi = dep_dinamik_dis_hucum / max(yarim_lig_ort, 0.1)
ev_ic_savunma_katsayi = ev_dinamik_ic_savunma / max(yarim_lig_ort, 0.1)

# 7. Lig Başı Oranlama Katsayıları (Genel Lig Odaklı)
ev_genel_hucum_katsayi = ev_genel_hucum_ort / max(yarim_lig_ort, 0.1)
dep_genel_savunma_katsayi = dep_genel_savunma_ort / max(yarim_lig_ort, 0.1)
dep_genel_hucum_katsayi = dep_genel_hucum_ort / max(yarim_lig_ort, 0.1)
ev_genel_savunma_katsayi = ev_genel_savunma_ort / max(yarim_lig_ort, 0.1)

# 8. Stabilize Edilmiş PPG (Puan) Dengesi
ev_ppg = ev_ic_puan / ev_ic_mac
dep_ppg = dep_dis_puan / dep_dis_mac
puan_orani = ev_ppg / max((ev_ppg + dep_ppg), 0.1)
puan_denge_carpan = 0.85 + (puan_orani * 0.30)

# 9. Motivasyon ve Kadro Eksik Metrisleri
onem_farki = ev_onem - dep_onem
ev_onem_carpan = 1.0 + (onem_farki * 0.03)
dep_onem_carpan = 1.0 - (onem_farki * 0.03)
ev_kadro_cezasi = (ev_kritik_eksik * 0.22) + (ev_normal_eksik * 0.08)
dep_kadro_cezasi = (dep_kritik_eksik * 0.22) + (dep_normal_eksik * 0.08)

# 10. Çift Yönlü xG Modellemesi
# Temel İç/Dış Projeksiyonu
ev_ic_xg = (yarim_lig_ort * ev_ic_hucum_katsayi * dep_dis_savunma_katsayi)
dep_dis_xg = (yarim_lig_ort * dep_dis_hucum_katsayi * ev_ic_savunma_katsayi)

# Temel Genel Lig Projeksiyonu
ev_genel_xg = (yarim_lig_ort * ev_genel_hucum_katsayi * dep_genel_savunma_katsayi)
dep_genel_xg = (yarim_lig_ort * dep_genel_hucum_katsayi * ev_genel_savunma_katsayi)

# 11. Hibrit xG Birleştirme (%50 İç-Dış Etkisi + %50 Genel Lig Karakteri) ve Çarpanlar
ev_xg = ((ev_ic_xg * 0.5) + (ev_genel_xg * 0.5)) * puan_denge_carpan * ev_onem_carpan - ev_kadro_cezasi
dep_xg = ((dep_dis_xg * 0.5) + (dep_genel_xg * 0.5)) * (2.0 - puan_denge_carpan) * dep_onem_carpan - dep_kadro_cezasi

ev_xg = max(ev_xg, 0.05)
dep_xg = max(dep_xg, 0.05)



# Simülasyon Hesaplamaları (Genişletilmiş Havuz)

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

# 3. Gol ve Tempo Eğilim Analizi (Genişletilmiş Poisson Dağılımı)
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

        

        # Filtreleri daha profesyonel ve güvenli sınırlara çektik

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
    st.write("Veritabanına mühürlenmiş geçmiş tahminlerinizi yönetin, sistem verimliliğini ve başarı indeksini analiz edin.")
    
    df_logs = pd.read_csv(DB_FILE)
    
    if len(df_logs) == 0:
        st.info("Arşiv veritabanı henüz boş kanka. İlk analizini kaydedince burası şenlenecek!")
    else:
        # --- 1. MİLYON DOLARLIK KPI METRİKLERİ (PREMIUM CARDS) ---
        toplam_mac = len(df_logs)
        kazananlar = len(df_logs[df_logs["Sonuc"] == "KAZANDI ✅"])
        kaybedenler = len(df_logs[df_logs["Sonuc"] == "KAYBETTİ ❌"])
        bekleyenler = len(df_logs[df_logs["Sonuc"] == "Bekliyor"])
        
        # Başarı Oranı Hesaplama (Bekleyenler hariç gerçek performans)
        sonuclanmis = kazananlar + kaybedenler
        basari_orani = (kazananlar / sonuclanmis * 100) if sonuclanmis > 0 else 0.0
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label="📊 Toplam Analiz Edilen Müsabaka", value=toplam_mac)
        with m_col2:
            st.metric(label="🔥 Model Başarı Endeksi", value=f"%{basari_orani:.1f}", delta=f"+%{basari_orani:.1f}" if basari_orani > 50 else f"-%{100-basari_orani:.1f}")
        with m_col3:
            st.metric(label="⏳ Sonuç Bekleyen Tahminler", value=bekleyenler)
        with m_col4:
            st.metric(label="🎯 İsabetli Kupon / Yatan", value=f"{kazananlar} / {kaybedenler}")
            
        st.write("---")
        
        # --- 2. Gelişmiş Filtreleme ve Arama Barı ---
        st.markdown("### 🔍 Akıllı Filtreleme ve Arama")
        f_c1, f_c2 = st.columns([2, 1])
        with f_c1:
            arama_terimi = st.text_input("Takım Adına Göre Arşivde Ara...", "").lower()
        with f_c2:
            durum_filtresi = st.selectbox("Duruma Göre Süz", ["Hepsi", "KAZANDI ✅", "KAYBETTİ ❌", "Bekliyor"])
            
        # Filtreleri Uygula
        df_goster = df_logs.copy()
        if arama_terimi:
            df_goster = df_goster[df_goster["Ev Sahibi"].str.lower().str.contains(arama_terimi) | df_goster["Deplasman"].str.lower().str.contains(arama_terimi)]
        if durum_filtresi != "Hepsi":
            df_goster = df_goster[df_goster["Sonuc"] == durum_filtresi]

        # --- 3. MAÇ DURUMU GÜNCELLEME VE GELİŞMİŞ SİLME PANELİ ---
        st.markdown("### 🛠️ Veritabanı Yönetim ve Operasyon Odası")
        
        # İşlemler için yan yana iki sütün: Biri Güncelleme, Biri Silme
        tablo_c1, tablo_c2 = st.columns([1, 1])
        
        with tablo_c1:
            st.markdown("<div style='background: #0f172a; padding:15px; border-radius:10px; border:1px solid #1e293b;'>", unsafe_allow_html=True)
            st.markdown("##### ✏️ Maç Sonucu Güncelle")
            secilen_index = st.selectbox(
                "İşlem Yapılacak Maçı Seç", 
                df_goster.index, 
                format_func=lambda x: f"[{x}] {df_goster.loc[x, 'Ev Sahibi']} - {df_goster.loc[x, 'Deplasman']} ({df_goster.loc[x, 'Onerilen_Bahis']})"
            )
            
            pazar_tipi = df_logs.loc[secilen_index, "Pazar_Tipi"]
            if pazar_tipi == "Maç Sonucu":
                secenekler = ["Bekliyor", "MS1 Bitti ✅", "X Bitti ✅", "MS2 Bitti ✅", "KAYBETTİ ❌"]
            elif pazar_tipi == "2.5 Alt/Üst":
                secenekler = ["Bekliyor", "2.5 Üst Bitti ✅", "2.5 Alt Bitti ✅", "KAYBETTİ ❌"]
            elif pazar_tipi == "Karşılıklı Gol":
                secenekler = ["Bekliyor", "KG Var Bitti ✅", "KG Yok Bitti ✅", "KAYBETTİ ❌"]
            else:
                secenekler = ["Bekliyor", "KAZANDI ✅", "KAYBETTİ ❌"]
                
            sonuc = st.selectbox("Gerçek Maç Sonucu Nedir?", secenekler, key="guncelle_select")
            
            if st.button("🔄 Sonucu İşle ve İstatistikleri Yenile", use_container_width=True):
                onerilen = df_logs.loc[secilen_index, "Onerilen_Bahis"]
                
                # Kesin Kontrol Mekanizması (Hatalı yeşil tikleri önleyen kusursuz mantık)
                if "MS1" in onerilen and sonuc == "MS1 Bitti ✅": final_durum = "KAZANDI ✅"
                elif "MS2" in onerilen and sonuc == "MS2 Bitti ✅": final_durum = "KAZANDI ✅"
                elif "Beraberlik" in onerilen and sonuc == "X Bitti ✅": final_durum = "KAZANDI ✅"
                elif "2.5 Üst" in onerilen and sonuc == "2.5 Üst Bitti ✅": final_durum = "KAZANDI ✅"
                elif "2.5 Alt" in onerilen and sonuc == "2.5 Alt Bitti ✅": final_durum = "KAZANDI ✅"
                elif "KG Var" in onerilen and sonuc == "KG Var Bitti ✅": final_durum = "KAZANDI ✅"
                elif "KG Yok" in onerilen and sonuc == "KG Yok Bitti ✅": final_durum = "KAZANDI ✅"
                elif sonuc == "Bekliyor": final_durum = "Bekliyor"
                else: final_durum = "KAYBETTİ ❌"  # Şartlar uyuşmuyorsa kupon net yatmıştır.
                
                df_logs.loc[secilen_index, "Sonuc"] = final_durum
                df_logs.to_csv(DB_FILE, index=False)
                st.success(f"[{secilen_index}] ID'li müsabaka başarıyla '{final_durum}' olarak güncellendi!")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tablo_c2:
            st.markdown("<div style='background: #180f14; padding:15px; border-radius:10px; border:1px solid #3f1616;'>", unsafe_allow_html=True)
            st.markdown("<h5 style='color:#ef4444;'>🗑️ Veritabanından Analiz Sil</h5>", unsafe_allow_html=True)
            silinecek_index = st.selectbox(
                "Arşivden Tamamen Silinecek Maçı Seç", 
                df_goster.index, 
                format_func=lambda x: f"[{x}] {df_goster.loc[x, 'Ev Sahibi']} - {df_goster.loc[x, 'Deplasman']}"
            )
            
            st.write("")
            st.write(f"⚠️ **DİKKAT:** **[{silinecek_index}]** ID'li maç veritabanından kalıcı olarak silinecektir. Bu işlem geri alınamaz.")
            st.write("")
            
            if st.button("💥 Seçilen Analizi Kalıcı Olarak Sil", use_container_width=True):
                df_logs = df_logs.drop(silinecek_index).reset_index(drop=True)
                df_logs.to_csv(DB_FILE, index=False)
                st.error("Analiz kaydı arşivden söküldü ve imha edildi!")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.write("---")
        
        # --- 4. RENKLENDİRİLMİŞ PREMIUM DATA FRAME GÖRÜNÜMÜ ---
        st.markdown("### 📋 Güncel Analiz Log Dosyası Matrisi")
        
        # Renklendirme fonksiyonu (Milyon dolarlık finans şirketleri tasarımı)
        def satır_renklendir(val):
            if val == "KAZANDI ✅":
                return "background-color: #064e3b; color: #10b981; font-weight: bold;"
            elif val == "KAYBETTİ ❌":
                return "background-color: #7f1d1d; color: #f87171; font-weight: bold;"
            elif val == "Bekliyor":
                return "background-color: #78350f; color: #fbbf24; font-weight: bold;"
            return ""
            
        # Sadece Görsel Tabloyu Şıklaştırmak İçin Stil Uygulama
        styled_df = df_goster.style.map(satır_renklendir, subset=["Sonuc"])
        
        st.dataframe(styled_df, use_container_width=True, height=350)

        st.dataframe(df_logs, use_container_width=True) 


# --- SEKME 3: AKILLI KUPON KOMBİNASYON ODASI ---
with sekme3:
    st.markdown('<div class="badge">FINANSAL PORTFÖY YÖNETİMİ v1.0</div>', unsafe_allow_html=True)
    st.title("🎟️ Akıllı Kupon Kombinasyon Odası")
    st.write("Arşivdeki 'Bekliyor' durumundaki maçları kombine edin, kasanızı profesyonelce yönetin.")
    
    df_logs = pd.read_csv(DB_FILE)
    df_kuponlar = pd.read_csv(KUPON_DB_FILE)
    
    # --- 🔄 OTOMATİK KUPON SONUÇLANDIRICI (KUPON MOTORU) ---
    if len(df_kuponlar) > 0 and len(df_logs) > 0:
        kupon_guncellendi_mi = False
        for idx, kupon in df_kuponlar.iterrows():
            if kupon["Durum"] == "Bekliyor":
                # Kupondaki maç ID'lerini listeye çevir
                mac_idleri = [int(x) for x in str(kupon["Mac_IDleri"]).split(",")]
                
                mac_durumlari = []
                for m_id in list(mac_idleri):
                    if m_id in df_logs.index:
                        mac_durumlari.append(df_logs.loc[m_id, "Sonuc"])
                
                # Otomatik Değerlendirme Mantığı
                if "KAYBETTİ ❌" in mac_durumlari:
                    df_kuponlar.loc[idx, "Durum"] = "KAYBETTİ ❌"
                    kupon_guncellendi_mi = True
                elif len(mac_durumlari) == len(mac_idleri) and all(durum == "KAZANDI ✅" for durum in mac_durumlari):
                    df_kuponlar.loc[idx, "Durum"] = "KAZANDI ✅"
                    kupon_guncellendi_mi = True
                    
        if kupon_guncellendi_mi:
            df_kuponlar.to_csv(KUPON_DB_FILE, index=False)
            df_kuponlar = pd.read_csv(KUPON_DB_FILE) # Güncel halini tekrar çek
            
    # --- 📊 KUPON BAŞARI METRİKLERİ ---
    if len(df_kuponlar) > 0:
        k_toplam = len(df_kuponlar)
        k_kazanan = len(df_kuponlar[df_kuponlar["Durum"] == "KAZANDI ✅"])
        k_kaybeden = len(df_kuponlar[df_kuponlar["Durum"] == "KAYBETTİ ❌"])
        k_bekleyen = len(df_kuponlar[df_kuponlar["Durum"] == "Bekliyor"])
        
        k_sonuclanmis = k_kazanan + k_kaybeden
        k_basari = (k_kazanan / k_sonuclanmis * 100) if k_sonuclanmis > 0 else 0.0
        
        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            st.metric(label="🎟️ Toplam Yapılan Kupon", value=k_toplam)
        with kc2:
            st.metric(label="🎯 Kupon Başarı Yüzdesi", value=f"%{k_basari:.1f}")
        with kc3:
            st.metric(label="💰 Toplam Yatırılan (Bekleyen)", value=f"{df_kuponlar[df_kuponlar['Durum']=='Bekliyor']['Yatirilan_Tutar'].sum()} TL")
    st.write("---")

    # --- ➕ YENİ KUPON OLUŞTURMA ALANI ---
    st.markdown("### 🛠️ Yeni Kupon Kombinasyonu Oluştur")
    
    # Sadece durumu "Bekliyor" olan maçları süzüyoruz
    bekleyen_maclar = df_logs[df_logs["Sonuc"] == "Bekliyor"]
    
    if len(bekleyen_maclar) == 0:
        st.warning("Kupon yapabilmek için arşivde durumu 'Bekliyor' olan en az 1 maç olmalı kanka. Önce analiz yapıp kaydet!")
    else:
        # Multiselect ile çoklu maç seçimi
        secilen_mac_gosterim = st.multiselect(
            "Kupona Eklenecek Maçları Seç kanka:",
            options=bekleyen_maclar.index,
            format_func=lambda x: f"ID [{x}] {bekleyen_maclar.loc[x, 'Ev Sahibi']} - {bekleyen_maclar.loc[x, 'Deplasman']} ({bekleyen_maclar.loc[x, 'Onerilen_Bahis']})"
        )
        
        # 1000 TL Limitli Tutar Girişi
        yatirilacak_para = st.number_input("Kupon Tutarı (TL):", min_value=10, max_value=1000, value=100, step=10, help="Maksimum 1000 TL yatırabilirsin kanka.")
        
        if st.button("🚀 Kuponu Kilitle ve Gönder", use_container_width=True):
            if len(secilen_mac_gosterim) == 0:
                st.error("En az 1 maç seçmeden kuponu kilitleyemezsin!")
            else:
                # Seçilen maçların özet metnini oluştur
                detay_listesi = []
                for m_id in secilen_mac_gosterim:
                    ev = bekleyen_maclar.loc[m_id, 'Ev Sahibi']
                    dep = bekleyen_maclar.loc[m_id, 'Deplasman']
                    tahmin = bekleyen_maclar.loc[m_id, 'Onerilen_Bahis']
                    detay_listesi.append(f"{ev}-{dep} ({tahmin})")
                
                mac_detaylari_str = " | ".join(detay_listesi)
                mac_idleri_str = ",".join([str(x) for x in secilen_mac_gosterim])
                new_id = len(df_kuponlar) + 1
                
                # Yeni kupon satırını ekle
                yeni_kupon = pd.DataFrame([{
                    "Kupon_ID": new_id,
                    "Mac_IDleri": mac_idleri_str,
                    "Mac_Detaylari": mac_detaylari_str,
                    "Yatirilan_Tutar": yatirilacak_para,
                    "Durum": "Bekliyor"
                }])
                
                df_kuponlar = pd.concat([df_kuponlar, yeni_kupon], ignore_index=True)
                df_kuponlar.to_csv(KUPON_DB_FILE, index=False)
                st.success(f"🎟️ {new_id} numaralı kupon başarıyla kilitlendi! Sistem takibe aldı.")
                st.rerun()

    # --- 📋 KUPON ARŞİVİ TABLOSU ---
    st.write("---")
    st.markdown("### 📜 Güncel Kupon Portföyü")
    if len(df_kuponlar) > 0:
        def kupon_renklendir(val):
            if val == "KAZANDI ✅": return "background-color: #064e3b; color: #10b981; font-weight: bold;"
            elif val == "KAYBETTİ ❌": return "background-color: #7f1d1d; color: #f87171; font-weight: bold;"
            elif val == "Bekliyor": return "background-color: #78350f; color: #fbbf24; font-weight: bold;"
            return ""
        
        styled_kupon = df_kuponlar.style.map(kupon_renklendir, subset=["Durum"])
        st.dataframe(styled_kupon, use_container_width=True)
