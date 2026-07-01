import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
from datetime import datetime
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Sezgin Görmüş Veri Analizi v8.0", page_icon="🔮", layout="wide")

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
    pd.DataFrame(columns=["Kupon_ID", "Mac_IDleri", "Yatirilan_Tutar", "Durum", "Tarih"]).to_csv(KUPON_DB_FILE, index=False)
else:
    temp_df = pd.read_csv(KUPON_DB_FILE)
    if "Mac_IDleri" not in temp_df.columns:
        pd.DataFrame(columns=["Kupon_ID", "Mac_IDleri", "Yatirilan_Tutar", "Durum", "Tarih"]).to_csv(KUPON_DB_FILE, index=False)

# --- POISSON MOTORU ---
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

def ai_yorum_uret(ev_sahibi, deplasman, ev_xg, dep_xg, ms1_olasilik, ms2_olasilik, en_iyi_bahis):
    fark = ev_xg - dep_xg
    if fark > 0.8: durum = f"{ev_sahibi} hücumda çok agresif ve dominasyonu elinde tutan bir yapıda."
    elif fark > 0.3: durum = f"{ev_sahibi} oyunu kontrol eden ve skor üretmeye yakın taraf."
    elif fark < -0.8: durum = f"{deplasman} takımı sahaya çok daha net ve tehlikeli ayaklarla çıkıyor."
    elif fark < -0.3: durum = f"{deplasman} oyunu kendi lehine çevirebilecek potansiyele sahip."
    else: durum = "İki takım da birbirine karşı üstünlük kurmakta zorlanacak, tam bir satranç maçı."

    en_yuksek = max(ms1_olasilik, ms2_olasilik)
    if en_yuksek > 65: guven_seviyesi = "Modelimizin bu sonuca olan güveni oldukça yüksek."
    elif en_yuksek > 50: guven_seviyesi = "Veriler, sonucun bu yönde evrilme ihtimalini güçlü kılıyor."
    else: guven_seviyesi = "Maçın sonucu kağıt üzerinde oldukça yakın görünüyor, dikkatli olunmalı."

    if "MS1" in en_iyi_bahis: bahis_yorum = "Ev sahibi avantajı ve güncel form durumu bu tercihi mantıklı kılıyor."
    elif "MS2" in en_iyi_bahis: bahis_yorum = "Deplasman ekibinin kontra atak verimliliği bu seçimi öne çıkarıyor."
    elif "Üst" in en_iyi_bahis: bahis_yorum = "Savunma disiplinlerinden ziyade, skor odaklı bir mücadele beklentisi var."
    elif "Alt" in en_iyi_bahis: bahis_yorum = "Kilitli kalması muhtemel ve taktiksel disiplinin ön planda olduğu bir senaryo."
    elif "KG" in en_iyi_bahis: bahis_yorum = "İki ekibin de savunma zafiyetleri göz önüne alındığında gol yollarının açık olması muhtemel."
    else: bahis_yorum = "Sistemimiz bu markette bir değer (value) tespit etti."

    return f"🤖 {durum} {guven_seviyesi} {bahis_yorum} ({en_iyi_bahis} seçimi verilerle destekleniyor.)"

LIG_VERITABANI = {
    "Türkiye Trendyol Süper Lig": ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir", "Kasımpaşa", "Göztepe", "Samsunspor", "Alanyaspor", "Rizespor", "Çorum FK", "Amed SK", "Erzurumspor FK", "Eyüpspor", "Gençlerbirliği", "Gaziantep FK", "Kocaelispor", "Konyaspor"],
    "İngiltere Premier Lig": ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool", "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest", "Tottenham Hotspur"]
}

# --- CSS VE ARAYÜZ ---
st.markdown("""
    <style>
    .badge { background: #3D9DF3; color: #fff; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; margin-bottom: 15px; }
    .signal-green { color: #10b981 !important; font-weight: bold; }
    .signal-red { color: #ef4444 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

sekme1, sekme2, sekme3 = st.tabs(["📊 Analiz Ekranı", "🗂️ Arşiv Paneli", "🎟️ Kupon Odası"])

st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 20px;'>📋 VERİ SEVİYESİ</h2>", unsafe_allow_html=True)
st.sidebar.markdown("### 🏟️ Müsabaka Seçim Odası")

lig_listesi = list(LIG_VERITABANI.keys()) + ["🌍 Diğer / Listede Olmayan Lig"]
secilen_lig = st.sidebar.selectbox("Ligi Seç kanka:", lig_listesi)

if secilen_lig == "🌍 Diğer / Listede Olmayan Lig":
    ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adı:", "Ev Sahibi")
    deplasman = st.sidebar.text_input("Deplasman Takım Adı:", "Deplasman")
else:
    ev_secenekleri = ["✍️ Kendim Yazacağım..."] + LIG_VERITABANI[secilen_lig]
    ev_secim = st.sidebar.selectbox("Ev Sahibi Takım:", ev_secenekleri, index=1)
    ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım Adını Gir kanka:", "Ev Sahibi") if ev_secim == "✍️ Kendim Yazacağım..." else ev_secim

    dep_secenekleri = ["✍️ Kendim Yazacağım..."] + LIG_VERITABANI[secilen_lig]
    dep_secim = st.sidebar.selectbox("Deplasman Takımı:", dep_secenekleri, index=2)
    deplasman = st.sidebar.text_input("Deplasman Takım Adını Gir kanka:", "Deplasman") if dep_secim == "✍️ Kendim Yazacağım..." else dep_secim

st.sidebar.markdown("### 🌍 Lig Karakteristiği")
lig_ort_gol = st.sidebar.slider("Lig Geneli Maç Başına Gol Ort", 1.50, 4.00, 2.80, step=0.05)
form_agirligi = st.sidebar.slider("Son 5 Maçın Etki Oranı (%)", 20, 60, 35) / 100
sezon_agirligi = 1.0 - form_agirligi

# ÖRNEK DEĞERLER (Tüm değişkenlerin hatasız çalışması için kısaltılmış örnek blok)
ev_ic_puan = 34; ev_ic_mac = 15; ev_toplam_gol = 38; ev_toplam_yenen = 12
ev_son5_attigi = 11; ev_son5_yedigi = 4; ev_cs = 2; ev_onem = 5
dep_dis_puan = 28; dep_dis_mac = 15; dep_toplam_gol = 29; dep_toplam_yenen = 18
dep_son5_attigi = 9; dep_son5_yedigi = 6; dep_cs = 1; dep_onem = 4
ev_genel_mac = 20; ev_genel_attigi = 35; ev_genel_yedigi = 20
dep_genel_mac = 20; dep_genel_attigi = 25; dep_genel_yedigi = 28
ev_kritik_eksik = 0; ev_normal_eksik = 1; dep_kritik_eksik = 1; dep_normal_eksik = 2

b_ms1 = st.sidebar.number_input("Bülten Oranı: MS 1", min_value=1.01, value=1.85)
b_x = st.sidebar.number_input("Bülten Oranı: Beraberlik (X)", min_value=1.01, value=3.60)
b_ms2 = st.sidebar.number_input("Bülten Oranı: MS 2", min_value=1.01, value=3.40)

# KATSAYI HESAPLARI
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
ev_dinamik_ic_savunma = (ev_ic_savunma_ort * sezon_agirligi) + (ev_son5_savunma_ort * form_agirligi) * (1.0 - (ev_cs * 0.03))
dep_dinamik_dis_hucum = (dep_dis_hucum_ort * sezon_agirligi) + (dep_son5_hucum_ort * form_agirligi)
dep_dinamik_dis_savunma = (dep_dis_savunma_ort * sezon_agirligi) + (dep_son5_savunma_ort * form_agirligi) * (1.0 - (dep_cs * 0.03))

yarim_lig_ort = lig_ort_gol / 2
ev_xg = max(((yarim_lig_ort * (ev_dinamik_ic_hucum / max(yarim_lig_ort, 0.1)) * (dep_dinamik_dis_savunma / max(yarim_lig_ort, 0.1))) * 0.5 + 
             (yarim_lig_ort * (ev_genel_hucum_ort / max(yarim_lig_ort, 0.1)) * (dep_genel_savunma_ort / max(yarim_lig_ort, 0.1))) * 0.5) * 
             (0.85 + ((ev_ic_puan / ev_ic_mac) / max(((ev_ic_puan / ev_ic_mac) + (dep_dis_puan / dep_dis_mac)), 0.1)) * 0.30) * 
             (1.0 + ((ev_onem - dep_onem) * 0.03)) - ((ev_kritik_eksik * 0.22) + (ev_normal_eksik * 0.08)), 0.05)
dep_xg = max(((yarim_lig_ort * (dep_dinamik_dis_hucum / max(yarim_lig_ort, 0.1)) * (ev_dinamik_ic_savunma / max(yarim_lig_ort, 0.1))) * 0.5 + 
              (yarim_lig_ort * (dep_genel_hucum_ort / max(yarim_lig_ort, 0.1)) * (ev_genel_savunma_ort / max(yarim_lig_ort, 0.1))) * 0.5) * 
              (2.0 - (0.85 + ((ev_ic_puan / ev_ic_mac) / max(((ev_ic_puan / ev_ic_mac) + (dep_dis_puan / dep_dis_mac)), 0.1)) * 0.30)) * 
              (1.0 - ((ev_onem - dep_onem) * 0.03)) - ((dep_kritik_eksik * 0.22) + (dep_normal_eksik * 0.08)), 0.05)

ms1_olasilik, x_olasilik, ms2_olasilik = mac_simule_et(ev_xg, dep_xg)

olasiliklar = []
ust_olasilik, kg_var_olasilik = 0.0, 0.0
for i in range(11):
    for j in range(11):
        p = poisson_olasilik(i, ev_xg) * poisson_olasilik(j, dep_xg)
        if i < 6 and j < 6: olasiliklar.append({'toplam': i + j, 'p': p})
        if (i + j) > 2.5: ust_olasilik += p
        if i > 0 and j > 0: kg_var_olasilik += p
ust_olasilik *= 100; kg_var_olasilik *= 100

def get_p(alt_limit): return sum([item['p'] for item in olasiliklar if item['toplam'] < alt_limit]) * 100
gol_alt_1_5, gol_ust_1_5 = get_p(1.5), 100 - get_p(1.5)
gol_alt_2_5, gol_ust_2_5 = get_p(2.5), 100 - get_p(2.5)
gol_alt_3_5, gol_ust_3_5 = get_p(3.5), 100 - get_p(3.5)

v_ms1, v_x, v_ms2 = ms1_olasilik - (100 / b_ms1), x_olasilik - (100 / b_x), ms2_olasilik - (100 / b_ms2)

with sekme1:
    en_iyi_bahis = "RİSKLİ MÜSABAKA (PAS) ⚠️"
    if v_ms1 > 0 and ms1_olasilik >= 48: en_iyi_bahis = f"Maç Sonucu 1 ({ev_sahibi})"
    elif v_ms2 > 0 and ms2_olasilik >= 48: en_iyi_bahis = f"Maç Sonucu 2 ({deplasman})"
    elif v_x > 0 and x_olasilik >= 32: en_iyi_bahis = "Beraberlik (X)"
    elif ust_olasilik >= 60: en_iyi_bahis = "2.5 Üst"
    elif ust_olasilik <= 40: en_iyi_bahis = "2.5 Alt"
    elif kg_var_olasilik >= 60: en_iyi_bahis = "Karşılıklı Gol Var (KG Var)"

    st.markdown('<div class="badge">SEZGİN GÖRMÜŞ VERİ ANALİZİ</div>', unsafe_allow_html=True)
    st.title("📊 Sezgin Görmüş Matematiksel Tahmin Robotu")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Ev xG", f"{ev_xg:.2f}")
    m2.metric("Dep xG", f"{dep_xg:.2f}")
    m3.metric("Tahmin", en_iyi_bahis.split(" (")[0])
    
    col_sol, col_sag = st.columns([11, 10])
    
    with col_sol:
        # HATA DÜZELTME: RAW HTML sarmalaması yerine st.container kullandık
        with st.container(border=True):
            st.subheader("🎯 Profesyonel Olasılık Dağılımı")
            grafik_tipi = st.radio("Görünüm:", ["Maç Sonucu", "1.5 Alt/Üst", "2.5 Alt/Üst", "3.5 Alt/Üst", "KG Durumu"], horizontal=True)
            
            if grafik_tipi == "Maç Sonucu": fig = go.Figure(data=[go.Pie(labels=['MS1', 'X', 'MS2'], values=[ms1_olasilik, x_olasilik, ms2_olasilik], hole=.4)])
            elif "1.5" in grafik_tipi: fig = go.Figure(data=[go.Pie(labels=['Üst', 'Alt'], values=[gol_ust_1_5, gol_alt_1_5], hole=.4, marker_colors=['#0acc31', '#f53333'])])
            elif "2.5" in grafik_tipi: fig = go.Figure(data=[go.Pie(labels=['Üst', 'Alt'], values=[gol_ust_2_5, gol_alt_2_5], hole=.4, marker_colors=['#0acc31', '#f53333'])])
            elif "3.5" in grafik_tipi: fig = go.Figure(data=[go.Pie(labels=['Üst', 'Alt'], values=[gol_ust_3_5, gol_alt_3_5], hole=.4, marker_colors=['#0acc31', '#f53333'])])
            else: fig = go.Figure(data=[go.Pie(labels=['KG Var', 'KG Yok'], values=[kg_var_olasilik, 100-kg_var_olasilik], hole=.4, marker_colors=['#0acc31', '#f53333'])])

            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(t=30, b=30, l=30, r=30), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,uirevision='constant'))
            st.plotly_chart(fig, use_container_width=True, key=f"grafik_{grafik_tipi}")
            
        with st.container(border=True):
            st.subheader("🤖 AI Maç Yorumu")
            st.info(ai_yorum_uret(ev_sahibi, deplasman, ev_xg, dep_xg, ms1_olasilik, ms2_olasilik, en_iyi_bahis))
    
    with col_sag:
        with st.container(border=True):
            st.subheader("🚨 Tahmin ve Sinyal Odası")
            def sinyal_satiri(pazar, model_p, oran, value):
                renk = "signal-green" if value > 0 else "signal-red"
                ikon = "✅" if value > 0 else "⚠️"
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
            
            sinyal_satiri(f"MS 1 ({ev_sahibi})", ms1_olasilik, b_ms1, v_ms1)
            sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
            sinyal_satiri(f"MS 2 ({deplasman})", ms2_olasilik, b_ms2, v_ms2)

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
