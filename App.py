import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
from datetime import datetime

# --- KULLANICI DOSTU SAYFA AYARLARI ---
st.set_page_config(page_title="Sezgin Görmüş Veri Analizi AI PRO", page_icon="🔮", layout="wide")

DB_FILE = "analiz_gunlugu.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=[
        "Tarih", "Ev Sahibi", "Deplasman", 
        "Model_MS1", "Model_X", "Model_MS2", "Model_Ust", "Model_KGVar",
        "Onerilen_Bahis", "Pazar_Tipi", "Sonuc"
    ])
    df.to_csv(DB_FILE, index=False)

# --- 📐 MATEMATİKSEL POISSON MOTORU ---
def poisson_olasilik(k, lmbda):
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def mac_simule_et(ev_gol_beklentisi, dep_gol_beklentisi):
    ms1, beraberlik, ms2 = 0.0, 0.0, 0.0
    for i in range(7):
        for j in range(7):
            p_ev = poisson_olasilik(i, ev_gol_beklentisi)
            p_dep = poisson_olasilik(j, dep_gol_beklentisi)
            p_skor = p_ev * p_dep
            if i > j: ms1 += p_skor
            elif i == j: beraberlik += p_skor
            else: ms2 += p_skor
    toplam = ms1 + beraberlik + ms2
    return (ms1/toplam)*100, (beraberlik/toplam)*100, (ms2/toplam)*100

# --- PREMIUM & KULLANICI DOSTU GÖRSEL MOTOR (CSS) ---
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

# --- SIDEBAR (KULLANICI DOSTU VERİ GİRİŞ ALANI) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 20px;'>📋 VERİ PANELİ</h2>", unsafe_allow_html=True)

st.sidebar.markdown("### 🏟️ Müsabaka Bilgileri")
ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adı", "Dalian")
deplasman = st.sidebar.text_input("Deplasman Takımı Adı", "Shanghai")

st.sidebar.write("---")
st.sidebar.markdown("### 🏠 Ev Sahibi İstatistikleri")
ev_ic_mac = st.sidebar.number_input("İç Sahada Oynadığı Maç", min_value=1, value=11)
ev_ic_puan = st.sidebar.number_input("İç Sahada Topladığı Puan", min_value=0, value=15)
ev_toplam_gol = st.sidebar.number_input("İç Sahada Attığı Toplam Gol", min_value=0, value=13)
ev_toplam_yenen = st.sidebar.number_input("İç Sahada Yediği Toplam Gol", min_value=0, value=14)
ev_son5_attigi = st.sidebar.number_input("Son 5 Maçta Attığı Gol (Form)", min_value=0, value=8)
ev_son5_yedigi = st.sidebar.number_input("Son 5 Maçta Yediği Gol (Form)", min_value=0, value=9)
ev_cs = st.sidebar.slider("Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 1)
ev_onem = st.sidebar.slider("Ev Sahibinin Maç Önem Derecesi (1-5)", 1, 5, 4)

st.sidebar.write("---")
st.sidebar.markdown("### 🚀 Deplasman İstatistikleri")
dep_dis_mac = st.sidebar.number_input("Dış Sahada Oynadığı Maç", min_value=1, value=10)
dep_dis_puan = st.sidebar.number_input("Dış Sahada Topladığı Puan", min_value=0, value=21)
dep_toplam_gol = st.sidebar.number_input("Dış Sahada Attığı Toplam Gol", min_value=0, value=15)
dep_toplam_yenen = st.sidebar.number_input("Dış Sahada Yediği Toplam Gol", min_value=0, value=7)
dep_son5_attigi = st.sidebar.number_input("Son 5 Maçta Dışarıda Attığı Gol", min_value=0, value=12)
dep_son5_yedigi = st.sidebar.number_input("Son 5 Maçta Dışarıda Yediği Gol", min_value=0, value=4)
dep_cs = st.sidebar.slider("Deplasmanın Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 3)
dep_onem = st.sidebar.slider("Deplasmanın Maç Önem Derecesi (1-5)", 1, 5, 5)

st.sidebar.write("---")
st.sidebar.markdown("### 🛡️ Kadro & Bülten Oranları")
ev_eksik = st.sidebar.slider("Ev Sahibi Kritik Eksik Etkisi (0-5)", 0, 5, 1)
dep_eksik = st.sidebar.slider("Deplasman Kritik Eksik Etkisi (0-5)", 0, 5, 0)
b_ms1 = st.sidebar.number_input("Bülten Oranı: MS 1", min_value=1.01, value=4.80)
b_x = st.sidebar.number_input("Bülten Oranı: Beraberlik (X)", min_value=1.01, value=3.80)
b_ms2 = st.sidebar.number_input("Bülten Oranı: MS 2", min_value=1.01, value=1.45)


# --- ⚙️ SEZGİN GÖRMÜŞ MATRIX HESAPLAMA MOTORU ---
ev_genel_hucum = ev_toplam_gol / ev_ic_mac
ev_genel_savunma = ev_toplam_yenen / ev_ic_mac
dep_genel_hucum = dep_toplam_gol / dep_dis_mac
dep_genel_savunma = dep_toplam_yenen / dep_dis_mac

ev_son5_hucum = ev_son5_attigi / 5
ev_son5_savunma = ev_son5_yedigi / 5
dep_son5_hucum = dep_son5_attigi / 5
dep_son5_savunma = dep_son5_yedigi / 5

# %50 Tarih + %50 Trend Dengesi
ev_dinamik_hucum = (ev_genel_hucum * 0.5) + (ev_son5_hucum * 0.5)
ev_dinamik_savunma = (ev_genel_savunma * 0.5) + (ev_son5_savunma * 0.5)
dep_dinamik_hucum = (dep_genel_hucum * 0.5) + (dep_son5_hucum * 0.5)
dep_dinamik_savunma = (dep_genel_savunma * 0.5) + (dep_son5_savunma * 0.5)

# Savunma Duvarı ve Puan Denge Çarpanları
ev_dinamik_savunma *= (1.0 - (ev_cs * 0.06))
dep_dinamik_savunma *= (1.0 - (dep_cs * 0.06))
ev_ppg = ev_ic_puan / ev_ic_mac
dep_ppg = dep_dis_puan / dep_dis_mac
puan_denge = ev_ppg / (ev_ppg + dep_ppg)

# Motivasyon Dengesi
onem_farki = ev_onem - dep_onem
ev_onem_carpan = 1.0 + (onem_farki * 0.04)
dep_onem_carpan = 1.0 - (onem_farki * 0.04)

# Nihai Akıllı xG Değerleri
ev_xg = (ev_dinamik_hucum * dep_dinamik_savunma) * (puan_denge * 2) * ev_onem_carpan - (ev_eksik * 0.1)
dep_xg = (dep_dinamik_hucum * ev_dinamik_savunma) * ((1 - puan_denge) * 2) * dep_onem_carpan - (dep_eksik * 0.1)
ev_xg = max(ev_xg, 0.1)
dep_xg = max(dep_xg, 0.1)

# Simülasyon Çalıştırılması
ms1_olasilik, x_olasilik, ms2_olasilik = mac_simule_et(ev_xg, dep_xg)

# En Olası Skor Projeksiyonu
en_yuksek_skor_p, tahmin_ev_skor, tahmin_dep_skor = 0, 0, 0
for i in range(5):
    for j in range(5):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if p > en_yuksek_skor_p:
            en_yuksek_skor_p, tahmin_ev_skor, tahmin_dep_skor = p, i, j

# Alt/Üst ve KG Var Algoritması
ust_olasilik, kg_var_olasilik = 0.0, 0.0
for i in range(7):
    for j in range(7):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if (i + j) > 2.5: ust_olasilik += p
        if i > 0 and j > 0: kg_var_olasilik += p
ust_olasilik *= 100
kg_var_olasilik *= 100

# Value (Değer) Hesaplamaları
v_ms1 = ms1_olasilik - (100 / b_ms1)
v_x = x_olasilik - (100 / b_x)
v_ms2 = ms2_olasilik - (100 / b_ms2)


# --- 🧠 AKILLI YORUM JENERATÖRÜ ---
yorum_cumleleri = []
if abs(ev_xg - dep_xg) < 0.3:
    yorum_cumleleri.append(f"Veri analizi motoru, iki takım arasında dengeli bir taktik savaş okuyor (Ev xG: {ev_xg:.2f} vs Dep xG: {dep_xg:.2f}). Sahada kontrollü bir oyun tercih edilecektir.")
elif ev_xg > dep_xg:
    fark_onem = "belirgin bir baskı" if (ev_xg - dep_xg) > 0.8 else "hafif bir saha avantajı"
    yorum_cumleleri.append(f"{ev_sahibi}, iç saha dinamikleri ve {ev_xg:.2f}'lik hücum beklentisiyle sahaya {fark_onem} taşıyor. {deplasman} savunmasının işi hiç kolay olmayacaktır.")
else:
    fark_onem = "ezici bir oyun üstünlüğü" if (dep_xg - ev_xg) > 0.8 else "net bir performans avantajı"
    yorum_cumleleri.append(f"{deplasman} takımı, dış sahada vites artırma ve şablonu dikte etme becerisine sahip. Ürettiği {dep_xg:.2f}'lik güvenli xG değeri sahadaki {fark_onem} göstergesidir.")

if dep_cs >= 3 and ev_xg < 1.2:
    yorum_cumleleri.append(f"{deplasman} takımının geride kurduğu savunma duvarı (Son 5 maçta {dep_cs} gol yememe) ev sahibinin hücum ritmini tamamen bozabilir.")
if ev_cs >= 3 and dep_xg < 1.2:
    yorum_cumleleri.append(f"{ev_sahibi} ekibinin arkasındaki savunma direnci oldukça yüksek. Maçın düğümünü çözmek ekstra bireysel yetenek gerektirebilir.")

if ust_olasilik >= 60:
    yorum_cumleleri.append(f"İki takımın da yakın dönem skor trendleri yüksek tempolu ve bol pozisyonlu bir 90 dakikaya işaret ediyor. %{ust_olasilik:.1f}'lik 2.5 Üst ihtimali bunu destekler nitelikte.")
elif ust_olasilik <= 42:
    yorum_cumleleri.append(f"Simülasyon, savunma öncelikli oyun planlarının devrede olacağını gösteriyor. %{100-ust_olasilik:.1f}'lik 2.5 Alt ağırlığı ile kilit bir maç bizi bekliyor.")

nihai_yorum = " ".join(yorum_cumleleri)


# --- KULLANICI DOSTU ANA PANEL ---
with sekme1:
    st.markdown('<div class="badge">SEZGİN GÖRMÜŞ VERİ ANALİZİ AI v6.8</div>', unsafe_allow_html=True)
    st.title("📊 Sezgin Görmüş Matematiksel Tahmin & Risk Robotu")
    st.write(f"Sistem başarıyla optimize edildi. Ev Güvenli xG: **{ev_xg:.2f}** | Deplasman Güvenli xG: **{dep_xg:.2f}**")
    
    col_sol, col_sag = st.columns([11, 10])
    
    with col_sol:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎯 Olasılık Dağılım Grafiği")
        fig = go.Figure(data=[go.Pie(labels=['Ev Sahibi Galibiyeti (MS1)', 'Beraberlik (X)', 'Deplasman Galibiyeti (MS2)'], values=[ms1_olasilik, x_olasilik, ms2_olasilik], hole=.45, marker_colors=['#0d9488', '#d97706', '#e11d48'])])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10), height=250)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("⚽ Skor & Gol Projeksiyon Odası")
        st.markdown(f'<div class="skor-box">🤖 YAPAY ZEKÂ SKOR TAHMİNİ: {tahmin_ev_skor} - {tahmin_dep_skor}</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: st.metric(label="📈 2.5 Üst İhtimali", value=f"%{ust_olasilik:.1f}"); st.progress(int(ust_olasilik))
        with c2: st.metric(label="🔥 Karşılıklı Gol Var İhtimali", value=f"%{kg_var_olasilik:.1f}"); st.progress(int(kg_var_olasilik))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sag:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚨 Sinyal Odası (Value Oran Kontrolü)")
        def sinyal_satiri(pazar, model_p, oran, value):
            renk = "signal-green" if value > 0 else "signal-red"
            durum = "DEĞERLİ ORAN 🔥" if value > 0 else "DEĞERSİZ ORAN ❌"
            st.markdown(f'<div style="padding: 12px 0; border-bottom: 1px solid #1e293b;"><div style="display:flex; justify-content:space-between;"><b>{pazar}</b><span class="{renk}">{durum}</span></div><div style="font-size:13px; color:#94a3b8; margin-top:4px;">Model İhtimali: %{model_p:.1f} | Bülten Oranı: {oran:.2f} | <b>Yapay Zekâ Değeri: <span class="{renk}">{value:+.2f}</span></b></div></div>', unsafe_allow_html=True)
        
        sinyal_satiri(f"{ev_sahibi} Galibiyeti (MS1)", ms1_olasilik, b_ms1, v_ms1)
        sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
        sinyal_satiri(f"{deplasman} Galibiyeti (MS2)", ms2_olasilik, b_ms2, v_ms2)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card" style="background: #0f172a;">', unsafe_allow_html=True)
        st.subheader("💡 Sezgin Görmüş Akıllı Bahis Raporu")
        
        en_iyi_bahis = "RİSKLİ MÜSABAKA (PAS) ⚠️"
        pazar_t = "Yok"
        
        if v_ms1 > 0 and ms1_olasilik >= 50: en_iyi_bahis, pazar_t = f"Maç Sonucu 1 ({ev_sahibi})", "Maç Sonucu"
        elif v_ms2 > 0 and ms2_olasilik >= 50: en_iyi_bahis, pazar_t = f"Maç Sonucu 2 ({deplasman})", "Maç Sonucu"
        elif v_x > 0 and x_olasilik >= 32: en_iyi_bahis, pazar_t = "Beraberlik (X)", "Maç Sonucu"
        elif ust_olasilik >= 62: en_iyi_bahis, pazar_t = "2.5 Üst", "2.5 Alt/Üst"
        elif ust_olasilik <= 38: en_iyi_bahis, pazar_t = "2.5 Alt", "2.5 Alt/Üst"
        elif kg_var_olasilik >= 60: en_iyi_bahis, pazar_t = "Karşılıklı Gol Var (KG Var)", "Karşılıklı Gol"
        elif kg_var_olasilik <= 40: en_iyi_bahis, pazar_t = "Karşılıklı Gol Yok (KG Yok)", "Karşılıklı Gol"
        
        st.markdown(f'<div style="color:#f59e0b; font-weight:bold; font-size:17px; margin-bottom: 10px;">🎯 STRATEJİK SEÇİM: {en_iyi_bahis}</div>', unsafe_allow_html=True)
        
        st.markdown("##### 🎙️ Yapay Zekâ Taktik Analiz Özeti")
        st.markdown(f'<div class="yorum-box">{nihai_yorum}</div>', unsafe_allow_html=True)
        st.write("")
        
        if st.button("💾 Bu Analizi Arşive Kaydet"):
            df_logs = pd.read_csv(DB_FILE)
            yeni = {
                "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"), "Ev Sahibi": ev_sahibi, "Deplasman": deplasman, 
                "Model_MS1": round(ms1_olasilik,1), "Model_X": round(x_olasilik,1), "Model_MS2": round(ms2_olasilik,1), 
                "Model_Ust": round(ust_olasilik,1), "Model_KGVar": round(kg_var_olasilik,1),
                "Onerilen_Bahis": en_iyi_bahis, "Pazar_Tipi": pazar_t, "Sonuc": "Bekliyor"
            }
            df_logs = pd.concat([df_logs, pd.DataFrame([yeni])], ignore_index=True)
            df_logs.to_csv(DB_FILE, index=False)
            st.success("Analiz başarıyla arşiv veritabanına mühürlendi kanka!")
        st.markdown('</div>', unsafe_allow_html=True)

# --- SEKME 2: GELİŞMİŞ ARŞİV PANELİ ---
with sekme2:
    st.title("🗂️ Sezgin Görmüş Analiz Günlüğü Veritabanı")
    df_logs = pd.read_csv(DB_FILE)
    
    if "Pazar_Tipi" not in df_logs.columns: df_logs["Pazar_Tipi"] = "Maç Sonucu"
    if "Model_Ust" not in df_logs.columns: df_logs["Model_Ust"] = 0.0
    if "Model_KGVar" not in df_logs.columns: df_logs["Model_KGVar"] = 0.0
    df_logs.to_csv(DB_FILE, index=False)
    
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
