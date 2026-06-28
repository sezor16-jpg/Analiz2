import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
from datetime import datetime

# --- SEZGİN GÖRMÜŞ AI PRO v7.5 SAYFA AYARLARI ---
st.set_page_config(page_title="Sezgin Görmüş Veri Analizi AI PRO v7.5", page_icon="🔮", layout="wide")

DB_FILE = "analiz_gunlugu.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=[
        "Tarih", "Ev Sahibi", "Deplasman", 
        "Model_MS1", "Model_X", "Model_MS2", "Model_Ust", "Model_KGVar",
        "Onerilen_Bahis", "Pazar_Tipi", "Oran", "Guven_Skoru", "Sonuc"
    ])
    df.to_csv(DB_FILE, index=False)

# --- 📐 KUSURSUZ POISSON MOTORU ---
def poisson_olasilik(k, lmbda):
    if lmbda <= 0: return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def mac_simule_et(ev_gol_beklentisi, dep_gol_beklentisi):
    ms1, beraberlik, ms2 = 0.0, 0.0, 0.0
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

# --- PREMIUM GÖRSEL ARYÜZ TASARIMI ---
st.markdown("""
    <style>
    .main { background-color: #070a13; }
    [data-testid="stSidebar"] { background-color: #0d1324; border-right: 1px solid #1e293b; padding-top: 20px; }
    .premium-card { background: linear-gradient(135deg, #111c34 0%, #080f20 100%); padding: 24px; border-radius: 16px; border: 1px solid #1e293b; margin-bottom: 20px; }
    .skor-box { background: linear-gradient(90deg, #1e1b4b 0%, #311042 100%); border: 2px dashed #8b5cf6; padding: 15px; border-radius: 12px; text-align: center; font-size: 26px; font-weight: bold; color: #f8fafc; margin: 15px 0; letter-spacing: 1px; }
    .signal-green { color: #10b981; font-weight: bold; }
    .signal-red { color: #ef4444; font-weight: bold; }
    .badge { background: linear-gradient(90deg, #4c1d95 0%, #2563eb 100%); color: #ffffff; padding: 6px 16px; border-radius: 9999px; font-size: 13px; font-weight: bold; display: inline-block; margin-bottom: 12px; border: 1px solid #3b82f6; }
    .yorum-box { background-color: #0b1329; border-left: 4px solid #8b5cf6; padding: 18px; border-radius: 8px; font-size: 15px; color: #e2e8f0; line-height: 1.6; margin-top: 15px; }
    .kasa-card { background: #0d1527; border: 1px solid #10b981; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

sekme1, sekme2 = st.tabs(["🔮 Gelişmiş Veri Laboratuvarı", "🗂️ Backtest & Analiz Arşivi"])

# --- SIDEBAR PANELİ ---
st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 20px;'>📋 VERİ SEVİYESİ</h2>", unsafe_allow_html=True)
ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adı", "Barcelona")
deplasman = st.sidebar.text_input("Deplasman Takımı Adı", "Real Madrid")

st.sidebar.write("---")
lig_ort_gol = st.sidebar.slider("Lig Geneli Maç Başına Gol Ortalaması", 1.50, 4.00, 2.80, step=0.05)
form_agirligi = st.sidebar.slider("Son 5 Maçın (Form) Etki Oranı (%)", 20, 60, 35) / 100
sezon_agirligi = 1.0 - form_agirligi

st.sidebar.write("---")
st.sidebar.markdown("### 🏠 Ev Sahibi İstatistikleri")
ev_ic_mac = st.sidebar.number_input("İç Sahada Oynadığı Maç", min_value=1, value=15)
ev_ic_puan = st.sidebar.number_input("İç Sahada Topladığı Puan", min_value=0, value=36)
ev_toplam_gol = st.sidebar.number_input("İç Sahada Attığı Toplam Gol", min_value=0, value=34)
ev_toplam_yenen = st.sidebar.number_input("İç Sahada Yediği Toplam Gol", min_value=0, value=11)
ev_son5_attigi = st.sidebar.number_input("Son 5 Maçta Attığı Gol (Form)", min_value=0, value=12)
ev_son5_yedigi = st.sidebar.number_input("Son 5 Maçta Yediği Gol (Form)", min_value=0, value=3)
ev_cs = st.sidebar.slider("Ev Sahibi Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 3)
ev_onem = st.sidebar.slider("Ev Sahibinin Maç Önem Derecesi (1-5)", 1, 5, 5)

st.sidebar.write("---")
st.sidebar.markdown("### 🚀 Deplasman İstatistikleri")
dep_dis_mac = st.sidebar.number_input("Dış Sahada Oynadığı Maç", min_value=1, value=15)
dep_dis_puan = st.sidebar.number_input("Dış Sahada Topladığı Puan", min_value=0, value=30)
dep_toplam_gol = st.sidebar.number_input("Dış Sahada Attığı Toplam Gol", min_value=0, value=28)
dep_toplam_yenen = st.sidebar.number_input("Dış Sahada Yediği Toplam Gol", min_value=0, value=14)
dep_son5_attigi = st.sidebar.number_input("Son 5 Maçta Dışarıda Attığı Gol", min_value=0, value=10)
dep_son5_yedigi = st.sidebar.number_input("Son 5 Maçta Dışarıda Yediği Gol", min_value=0, value=5)
dep_cs = st.sidebar.slider("Deplasman Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 2)
dep_onem = st.sidebar.slider("Deplasmanın Maç Önem Derecesi (1-5)", 1, 5, 5)

st.sidebar.write("---")
st.sidebar.markdown("### 🛡️ Sağlık & Kadro Eksik Raporu")
ev_kritik_eksik = st.sidebar.slider("Ev Sahibi Kritik Eksik", 0, 3, 0)
ev_normal_eksik = st.sidebar.slider("Ev Sahibi Normal Eksik", 0, 5, 1)
dep_kritik_eksik = st.sidebar.slider("Deplasman Kritik Eksik", 0, 3, 0)
dep_normal_eksik = st.sidebar.slider("Deplasman Normal Eksik", 0, 5, 1)

st.sidebar.write("---")
st.sidebar.markdown("### 📊 Bülten Oran Odası")
b_ms1 = st.sidebar.number_input("Bülten Oranı: MS 1", min_value=1.01, value=2.10)
b_x = st.sidebar.number_input("Bülten Oranı: Beraberlik (X)", min_value=1.01, value=3.40)
b_ms2 = st.sidebar.number_input("Bülten Oranı: MS 2", min_value=1.01, value=2.90)
b_ust = st.sidebar.number_input("Bülten Oranı: 2.5 Üst", min_value=1.01, value=1.75)


# --- ⚙️ SEZGİN GÖRMÜŞ MATHEMATICAL MATRIX MOTORU (YILDIZLAR TEMİZLENDİ) ---
ev_genel_hucum = ev_toplam_gol / ev_ic_mac
ev_genel_savunma = ev_toplam_yenen / ev_ic_mac
dep_genel_hucum = dep_toplam_gol / dep_dis_mac
dep_genel_savunma = dep_toplam_yenen / dep_dis_mac

ev_son5_hucum = ev_son5_attigi / 5
ev_son5_savunma = ev_son5_yedigi / 5
dep_son5_hucum = dep_son5_attigi / 5
dep_son5_savunma = dep_son5_yedigi / 5

# Hatalı olan ** işaretleri kod alanından tamamen kaldırıldı
ev_dinamik_hucum = (ev_genel_hucum * ocean_agirligi if 'ocean_agirligi' in locals() else ev_genel_hucum * sezon_agirligi) + (ev_son5_hucum * form_agirligi)
ev_dinamik_savunma = (ev_genel_savunma * sezon_agirligi) + (ev_son5_savunma * form_agirligi)
dep_dinamik_hucum = (dep_genel_hucum * sezon_agirligi) + (dep_son5_hucum * form_agirligi)
dep_dinamik_savunma = (dep_genel_savunma * sezon_agirligi) + (dep_son5_savunma * form_agirligi)

ev_dinamik_savunma *= (1.0 - (ev_cs * 0.03))
dep_dinamik_savunma *= (1.0 - (dep_cs * 0.03))

yarim_lig_ort = lig_ort_gol / 2
ev_hucum_katsayi = ev_dinamik_hucum / max(yarim_lig_ort, 0.1)
dep_savunma_katsayi = dep_dinamik_savunma / max(yarim_lig_ort, 0.1)
dep_hucum_katsayi = dep_dinamik_hucum / max(yarim_lig_ort, 0.1)
ev_savunma_katsayi = ev_dinamik_savunma / max(yarim_lig_ort, 0.1)

ev_ppg = ev_ic_puan / ev_ic_mac
dep_ppg = dep_dis_puan / dep_dis_mac
puan_orani = ev_ppg / max((ev_ppg + dep_ppg), 0.1)
puan_denge_carpan = 0.85 + (puan_orani * 0.30)

onem_farki = ev_onem - dep_onem
ev_onem_carpan = 1.0 + (onem_farki * 0.03)
dep_onem_carpan = 1.0 - (onem_farki * 0.03)

ev_kadro_cezasi = (ev_kritik_eksik * 0.22) + (ev_normal_eksik * 0.08)
dep_kadro_cezasi = (dep_kritik_eksik * 0.22) + (dep_normal_eksik * 0.08)

ev_xg = (yarim_lig_ort * ev_hucum_katsayi * dep_savunma_katsayi) * puan_denge_carpan * ev_onem_carpan - ev_kadro_cezasi
dep_xg = (yarim_lig_ort * dep_hucum_katsayi * ev_savunma_katsayi) * (2.0 - puan_denge_carpan) * dep_onem_carpan - dep_kadro_cezasi
ev_xg = max(ev_xg, 0.05)
dep_xg = max(dep_xg, 0.05)

ms1_olasilik, x_olasilik, ms2_olasilik = mac_simule_et(ev_xg, dep_xg)

en_yuksek_skor_p, tahmin_ev_skor, tahmin_dep_skor = 0, 0, 0
for i in range(6):
    for j in range(6):
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
v_ust = ust_olasilik - (100 / b_ust)

# --- 🎯 GÜVEN SKORU ALGORİTMASI ---
xg_farki = abs(ev_xg - dep_xg)
if xg_farki >= 1.20: guven_metni, guven_renk = "SÜPER GÜVENLİ 🔥", "#10b981"
elif xg_farki >= 0.70: guven_metni, guven_renk = "YÜKSEK GÜVEN 📈", "#34d399"
elif xg_farki >= 0.35: guven_metni, guven_renk = "ORTA GÜVEN ⚖️", "#f59e0b"
else: guven_metni, guven_renk = "DÜŞÜK GÜVEN (RİSKLİ) ⚠️", "#ef4444"

# --- STRATEJİK SEÇİM BELİRLEME ---
en_iyi_bahis = "PAS (RİSKLİ MÜSABAKA)"
pazar_t = "Yok"
secilen_oran = 1.00

if v_ms1 > 0 and ms1_olasilik >= 45: en_iyi_bahis, pazar_t, secilen_oran = f"MS 1 ({ev_sahibi})", "Maç Sonucu", b_ms1
elif v_ms2 > 0 and ms2_olasilik >= 45: en_iyi_bahis, pazar_t, secilen_oran = f"MS 2 ({deplasman})", "Maç Sonucu", b_ms2
elif v_ust > 0 and ust_olasilik >= 58: en_iyi_bahis, pazar_t, secilen_oran = "2.5 Üst", "2.5 Alt/Üst", b_ust
elif v_x > 0 and x_olasilik >= 32: en_iyi_bahis, pazar_t, secilen_oran = "Beraberlik (X)", "Maç Sonucu", b_x
elif kg_var_olasilik >= 60: en_iyi_bahis, pazar_t, secilen_oran = "KG Var", "Karşılıklı Gol", 1.60

# --- ANA PANEL ARABİRİMİ ---
with sekme1:
    st.markdown('<div class="badge">SEZGİN GÖRMÜŞ VERİ ANALİZİ AI v7.5 ULTIMATE</div>', unsafe_allow_html=True)
    st.title("🔮 Gelişmiş Model Laboratuvarı & Güven Sinyali")
    st.write(f"Sistem Kararlılığı: **Maksimum** | Projeksiyon: **{ev_sahibi} ({ev_xg:.2f}) - ({dep_xg:.2f}) {deplasman}**")
    
    col_sol, col_sag = st.columns([11, 10])
    
    with col_sol:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎯 Olasılık Matrisi")
        fig = go.Figure(data=[go.Pie(labels=['MS1', 'X', 'MS2'], values=[ms1_olasilik, x_olasilik, ms2_olasilik], hole=.45, marker_colors=['#0d9488', '#d97706', '#e11d48'])])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10), height=240)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("⚽ Skor & Gol Projeksiyon Odası")
        st.markdown(f'<div class="skor-box">🤖 YAPAY ZEKÂ SKOR TAHMİNİ: {tahmin_ev_skor} - {tahmin_dep_skor}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.metric(label="📈 2.5 Üst Olasılığı", value=f"%{ust_olasilik:.1f}")
        with c2: st.metric(label="🔥 KG Var Olasılığı", value=f"%{kg_var_olasilik:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sag:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚨 Sinyal Odası (Value Oran Filtresi)")
        def sinyal_satiri(pazar, model_p, oran, value):
            renk = "signal-green" if value > 0 else "signal-red"
            durum = "DEĞERLİ 🔥" if value > 0 else "DEĞERSİZ ❌"
            st.markdown(f'<div style="padding: 10px 0; border-bottom: 1px solid #1e293b;"><div style="display:flex; justify-content:space-between;"><b>{pazar}</b><span class="{renk}">{durum}</span></div><div style="font-size:13px; color:#94a3b8;">İhtimal: %{model_p:.1f} | Oran: {oran:.2f} | Değer: <span class="{renk}">{value:+.2f}</span></div></div>', unsafe_allow_html=True)
        
        sinyal_satiri("MS 1", ms1_olasilik, b_ms1, v_ms1)
        sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
        sinyal_satiri("MS 2", ms2_olasilik, b_ms2, v_ms2)
        sinyal_satiri("2.5 Üst", ust_olasilik, b_ust, v_ust)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card" style="background: #0f172a;">', unsafe_allow_html=True)
        st.subheader("💡 Sezgin Görmüş Akıllı Strateji Raporu")
        st.markdown(f'<div style="color:#f59e0b; font-weight:bold; font-size:18px;">🎯 ÖNERİLEN BAHİS: {en_iyi_bahis}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:5px; font-size:16px;">🛡️ SİSTEM GÜVEN SKORU: <span style="color:{guven_renk}; font-weight:bold;">{guven_metni}</span></div>', unsafe_allow_html=True)
        
        st.write("")
        if st.button("💾 Analizi Arşive Gönder"):
            df_logs = pd.read_csv(DB_FILE)
            yeni = {
                "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"), "Ev Sahibi": ev_sahibi, "Deplasman": deplasman, 
                "Model_MS1": round(ms1_olasilik,1), "Model_X": round(x_olasilik,1), "Model_MS2": round(ms2_olasilik,1), 
                "Model_Ust": round(ust_olasilik,1), "Model_KGVar": round(kg_var_olasilik,1),
                "Onerilen_Bahis": en_iyi_bahis, "Pazar_Tipi": pazar_t, "Oran": secilen_oran, "Guven_Skoru": guven_metni, "Sonuc": "Bekliyor"
            }
            df_logs = pd.concat([df_logs, pd.DataFrame([yeni])], ignore_index=True)
            df_logs.to_csv(DB_FILE, index=False)
            st.success("Analiz başarıyla kilitlendi kanka!")
        st.markdown('</div>', unsafe_allow_html=True)

# --- SEKME 2: BACKTEST & ARŞİV PANELİ ---
with sekme2:
    st.title("🗂️ Backtest Performansı & Kar/Zarar İstasyon Odası")
    df_logs = pd.read_csv(DB_FILE)
    
    if len(df_logs) > 0:
        toplam_kupon = len(df_logs[df_logs["Sonuc"] != "Bekliyor"])
        kazananlar = len(df_logs[df_logs["Sonuc"] == "KAZANDI ✅"])
        kaybedenler = len(df_logs[df_logs["Sonuc"] == "KAYBETTİ ❌"])
        
        basari_orani = (kazananlar / toplam_kupon * 100) if toplam_kupon > 0 else 0.0
        
        sabit_bet = 100
        toplam_yatirilan = toplam_kupon * sabit_bet
        toplam_kasa = 0.0
        
        for idx, row in df_logs.iterrows():
            if row["Sonuc"] == "KAZANDI ✅":
                toplam_kasa += (sabit_bet * float(row["Oran"])) - sabit_bet
            elif row["Sonuc"] == "KAYBETTİ ❌":
                toplam_kasa -= sabit_bet
                
        roi = (toplam_kasa / toplam_yatirilan * 100) if toplam_yatirilan > 0 else 0.0
        
        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1: st.metric("📋 Değerlendirilen Maç", f"{toplam_kupon} Maç")
        with cf2: st.metric("🎯 Model Başarı Yüzdesi", f"%{basari_orani:.1f}")
        with cf3: 
            renk_kasa = "+" if toplam_kasa >= 0 else ""
            st.metric("💰 Net Kasa Değişimi (100TL Sabit)", f"{renk_kasa}{toplam_kasa:.2f} TL")
        with cf4: st.metric("📊 Finansal ROI (Yatırım Getirisi)", f"%{roi:+.1f}")
        
        st.write("---")
        st.markdown("### 🛠️ Maç Sonuçlandırma İstasyonu")
        c_m, c_s = st.columns(2)
        with c_m:
            secilen = st.selectbox("Sonuçlandırılacak Maçı Seç", df_logs.index, format_func=lambda x: f"{df_logs.loc[x, 'Ev Sahibi']} - {df_logs.loc[x, 'Deplasman']} -> [{df_logs.loc[x, 'Onerilen_Bahis']}]")
        with c_s:
            sonuc = st.selectbox("Maç Sonucu?", ["Bekliyor", "KAZANDI ✅", "KAYBETTİ ❌"])
            
        if st.button("Sonucu Arşive Mühürle"):
            df_logs.loc[secilen, "Sonuc"] = sonuc
            df_logs.to_csv(DB_FILE, index=False)
            st.success("Veritabanı güncellendi!")
            st.rerun()
            
        st.write("---")
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("Finansal backtest raporu üretilebilmesi için önce arşive maç kaydetmelisin kanka.")
