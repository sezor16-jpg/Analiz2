import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
from datetime import datetime
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

# --- 📐 KUSURSUZ POISSON MOTORU (PROFESYONEL LİMİT: 11) ---

def poisson_olasilik(k, lmbda):
    if lmbda <= 0: return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)
    
def mac_simule_et(ev_gol_beklentisi, dep_gol_beklentisi):
    ms1, beraberlik, ms2 = 0.0, 0.0, 0.0
    # range(11) ile ekstrem ve çılgın tüm skor varyasyonları (Örn: 5-4, 7-1) olasılık havuzuna dahil edildi.
    for i in range(11):
        for j in range(11):
            p_ev = poisson_olasilik(i, ev_gol_beklentisi)
            p_dep = poisson_olasilik(j, dep_gol_beklentisi)
            p_skor = p_ev * p_dep
            if i > j: ms1 += p_skor
            elif i == j: beraberlik += p_skor
            else: ms2 += p_skor
    toplam = ms1 + beraberlik + ms2
    if toplam == 0: return 33.3, 33.3, 33.3
    return (ms1/toplam)*100, (beraberlik/toplam)*100, (ms2/toplam)*100

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

sekme1, sekme2 = st.tabs(["🔮 Gelişmiş Veri Laboratuvarı", "🗂️ Sezgin Görmüş Analiz Arşivi"])

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

# 1. Temel Ortalama Hesaplamaları

ev_genel_hucum = ev_toplam_gol / ev_ic_mac

ev_genel_savunma = ev_toplam_yenen / ev_ic_mac

dep_genel_hucum = dep_toplam_gol / dep_dis_mac

dep_genel_savunma = dep_toplam_yenen / dep_dis_mac



ev_son5_hucum = ev_son5_attigi / 5

ev_son5_savunma = ev_son5_yedigi / 5

dep_son5_hucum = dep_son5_attigi / 5

dep_son5_savunma = dep_son5_yedigi / 5



# 2. Dinamik Süreç Entegrasyonu (Kullanıcı Ayarlı Ağırlık Dengelemesi)

ev_dinamik_hucum = (ev_genel_hucum * sezon_agirligi) + (ev_son5_hucum * form_agirligi)

ev_dinamik_savunma = (ev_genel_savunma * sezon_agirligi) + (ev_son5_savunma * form_agirligi)

dep_dinamik_hucum = (dep_genel_hucum * sezon_agirligi) + (dep_son5_hucum * form_agirligi)

dep_dinamik_savunma = (dep_genel_savunma * sezon_agirligi) + (dep_son5_savunma * form_agirligi)



# 3. Clean Sheet Çift Sayım Filtresi (Yumuşatılmış Etki)

ev_dinamik_savunma *= (1.0 - (ev_cs * 0.03))

dep_dinamik_savunma *= (1.0 - (dep_cs * 0.03))



# 4. Profesyonel Lig Başı Oranlama ve Ters Metrik Optimizasyonu

yarim_lig_ort = lig_ort_gol / 2

ev_hucum_katsayi = ev_dinamik_hucum / max(yarim_lig_ort, 0.1)

dep_savunma_katsayi = dep_dinamik_savunma / max(yarim_lig_ort, 0.1)



dep_hucum_katsayi = dep_dinamik_hucum / max(yarim_lig_ort, 0.1)

ev_savunma_katsayi = ev_dinamik_savunma / max(yarim_lig_ort, 0.1)



# 5. Stabilize Edilmiş PPG (Puan) Dengesi (Maksimum %15 Etki Marjı ile Yumuşatıldı)

ev_ppg = ev_ic_puan / ev_ic_mac

dep_ppg = dep_dis_puan / dep_dis_mac

puan_orani = ev_ppg / max((ev_ppg + dep_ppg), 0.1)

puan_denge_carpan = 0.85 + (puan_orani * 0.30)  # [0.85 - 1.15] Aralığında dengeli etki



# 6. Motivasyon ve Önem Derecesi Katsayısı

onem_farki = ev_onem - dep_onem

ev_onem_carpan = 1.0 + (onem_farki * 0.03)

dep_onem_carpan = 1.0 - (onem_farki * 0.03)



# 7. Kadro Eksik Metrisi (Kritik ve Normal Eksikler Ayrı Ağırlıklandırıldı)

ev_kadro_cezasi = (ev_kritik_eksik * 0.22) + (ev_normal_eksik * 0.08)

dep_kadro_cezasi = (dep_kritik_eksik * 0.22) + (dep_normal_eksik * 0.08)



# 8. Nihai Profesyonel xG Tahmin Çıktıları

ev_xg = (yarim_lig_ort * ev_hucum_katsayi * dep_savunma_katsayi) * puan_denge_carpan * ev_onem_carpan - ev_kadro_cezasi

dep_xg = (yarim_lig_ort * dep_hucum_katsayi * ev_savunma_katsayi) * (2.0 - puan_denge_carpan) * dep_onem_carpan - dep_kadro_cezasi



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

        st.subheader("⚽ Kusursuz Skor & Gol Projeksiyonu")

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

    st.title("🗂️ Sezgin Görmüş Analiz Günlüğü Veritabanı")

    df_logs = pd.read_csv(DB_FILE)

    

    if len(df_logs) == 0:

        st.info("Arşiv veritabanı henüz boş kanka.")

    else:

        st.markdown("### 🛠️ Maç Durumu Güncelleme Paneli")

        c_m, c_s = st.columns(2)

        with c_m:

            secilen = st.selectbox("Sonuçlandırılacak Maçı Seç", df_logs.index, format_func=lambda x: f"{df_logs.loc[x, 'Ev Sahibi']} - {df_logs.loc[x, 'Deplasman']} ({df_logs.loc[x, 'Onerilen_Bahis']})")

        with c_s:

            pazar_tipi = df_logs.loc[secilen, "Pazar_Tipi"]

            if pazar_tipi == "Maç Sonucu":

                secenekler = ["Bekliyor", "MS1 Bitti ✅", "X Bitti ✅", "MS2 Bitti ✅", "KAYBETTİ ❌"]

            elif pazar_tipi == "2.5 Alt/Üst":

                secenekler = ["Bekliyor", "2.5 Üst Bitti ✅", "2.5 Alt Bitti ✅", "KAYBETTİ ❌"]

            elif pazar_tipi == "Karşılıklı Gol":

                secenekler = ["Bekliyor", "KG Var Bitti ✅", "KG Yok Bitti ✅", "KAYBETTİ ❌"]

            else:

                secenekler = ["Bekliyor", "KAZANDI ✅", "KAYBETTİ ❌"]

                

            sonuc = st.selectbox("Gerçek Maç Sonucu Nedir?", secenekler)

            

        if st.button("Sonucu Kaydet ve Veritabanını Güncelle"):

            onerilen = df_logs.loc[secilen, "Onerilen_Bahis"]

            if "MS1" in onerilen and sonuc == "MS1 Bitti ✅": final_durum = "KAZANDI ✅"

            elif "MS2" in onerilen and sonuc == "MS2 Bitti ✅": final_durum = "KAZANDI ✅"

            elif "Beraberlik" in onerilen and sonuc == "X Bitti ✅": final_durum = "KAZANDI ✅"

            elif "2.5 Üst" in onerilen and sonuc == "2.5 Üst Bitti ✅": final_durum = "KAZANDI ✅"

            elif "2.5 Alt" in onerilen and sonuc == "2.5 Alt Bitti ✅": final_durum = "KAZANDI ✅"

            elif "KG Var" in onerilen and sonuc == "KG Var Bitti ✅": final_durum = "KAZANDI ✅"

            elif "KG Yok" in onerilen and sonuc == "KG Yok Bitti ✅": final_durum = "KAZANDI ✅"

            elif sonuc == "Bekliyor": final_durum = "Bekliyor"

            elif sonuc == "KAYBETTİ ❌": final_durum = "KAYBETTİ ❌"

            else: final_durum = sonuc

            

            df_logs.loc[secilen, "Sonuc"] = final_durum

            df_logs.to_csv(DB_FILE, index=False)

            st.success("Maç sonucu başarıyla arşive işlendi!")

            st.rerun()

            

        st.write("---")

        st.markdown("### 📋 Güncel Analiz Geçmişi")

        st.dataframe(df_logs, use_container_width=True) 

