import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime

# --- PREMIUM SAYFA AYARLARI ---
st.set_page_config(page_title="Sezgin Grms Enterprise", page_icon="📈", layout="wide")

# --- VERİ TABANI ALTYAPISI (CSV GÜNLÜĞÜ) ---
DB_FILE = "analiz_gunlugu.csv"

def veri_tabani_hazirla():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=[
            "Tarih", "Ev Sahibi", "Deplasman", "Model_MS1", "Model_X", "Model_MS2", 
            "Oran_MS1", "Oran_X", "Oran_MS2", "Onerilen_Bahis", "Value_Degeri", "Sonuc"
        ])
        df.to_csv(DB_FILE, index=False)

veri_tabani_hazirla()

# --- GÜNLÜĞE MAÇ KAYDETME ---
def mac_kaydet(ev, dep, m_ms1, m_x, m_ms2, o_ms1, o_x, o_ms2, oneri, val):
    df = pd.read_csv(DB_FILE)
    yeni_kayit = {
        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ev Sahibi": ev,
        "Deplasman": dep,
        "Model_MS1": round(m_ms1, 1),
        "Model_X": round(m_x, 1),
        "Model_MS2": round(m_ms2, 1),
        "Oran_MS1": o_ms1,
        "Oran_X": o_x,
        "Oran_MS2": o_ms2,
        "Onerilen_Bahis": oneri,
        "Value_Degeri": round(val, 2),
        "Sonuc": "Bekliyor"
    }
    # pandas concat ile ekleme
    df = pd.concat([df, pd.DataFrame([yeni_kayit])], ignore_index=True)
    df.to_csv(DB_FILE, index=False)
    return True

# --- KURUMSAL CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    .premium-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px; border-radius: 16px; border: 1px solid #334155; margin-bottom: 20px;
    }
    .commentary-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        padding: 24px; border-radius: 16px; border: 1px solid #4338ca; margin-bottom: 20px;
    }
    .signal-green { color: #10b981; font-weight: bold; }
    .signal-red { color: #ef4444; font-weight: bold; }
    .badge { background-color: #312e81; color: #c7d2fe; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; display: inline-block; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ÜST SEKMELER (TABS) - Ana Panel ve Günlük Arasında Geçiş
sekme1, sekme2 = st.tabs(["📊 Canlı Maç Analiz Laboratuvarı", "🗂️ Geçmiş Analiz Günlüğü & Performans"])

# --- YAN PANEL: VERİ GİRİŞ ODASI ---
st.sidebar.markdown("### 🏟️ 1. Müsabaka Tanımı")
ev_sahibi = st.sidebar.text_input("Ev Sahibi Takım", "Fenerbahçe")
deplasman = st.sidebar.text_input("Deplasman Takımı", "Rakip x")

st.sidebar.write("---")
st.sidebar.markdown("### 📊 2. Ev Sahibi Performans Verileri")
ev_ic_mac = st.sidebar.number_input("İç Saha Maç Sayısı", min_value=1, value=10)
ev_ic_puan = st.sidebar.number_input("İç Sahada Toplanan Puan", min_value=0, value=25)
ev_toplam_gol = st.sidebar.number_input("İç Sahada Attığı Gol", min_value=0, value=22)
ev_toplam_yenen = st.sidebar.number_input("İç Sahada Yediği Gol", min_value=0, value=8)
ev_son5 = st.sidebar.slider("Son 5 Maç Performansı (Puan)", 0, 15, 13)

st.sidebar.write("---")
st.sidebar.markdown("### 📉 3. Deplasman Performans Verileri")
dep_dis_mac = st.sidebar.number_input("Dış Saha Maç Sayısı", min_value=1, value=10)
dep_dis_puan = st.sidebar.number_input("Dış Sahada Toplanan Puan", min_value=0, value=19)
dep_toplam_gol = st.sidebar.number_input("Dış Sahada Attığı Gol", min_value=0, value=16)
dep_toplam_yenen = st.sidebar.number_input("Dış Sahada Yediği Gol", min_value=0, value=11)
dep_son5 = st.sidebar.slider("Son 5 Maç Performansı (Puan )", 0, 15, 10)

st.sidebar.write("---")
st.sidebar.markdown("### 🛡️ 4. Kritik Faktörler")
ev_eksik = st.sidebar.slider("Ev Sahibi Eksik/Cezalı Etkisi", 0, 5, 0)
dep_eksik = st.sidebar.slider("Deplasman Eksik/Cezalı Etkisi", 0, 5, 2)
ev_mot = st.sidebar.slider("Ev Sahibi Motivasyon Endeksi (1-5)", 1, 5, 5)
dep_mot = st.sidebar.slider("Deplasman Motivasyon Endeksi (1-5)", 1, 5, 4)

st.sidebar.write("---")
st.sidebar.markdown("### 💰 5. Piyasa Oranları")
b_ms1 = st.sidebar.number_input("Piyasa Oranı: MS1", min_value=1.01, value=1.75, step=0.05)
b_x = st.sidebar.number_input("Piyasa Oranı: X", min_value=1.01, value=3.50, step=0.05)
b_ms2 = st.sidebar.number_input("Piyasa Oranı: MS2", min_value=1.01, value=3.20, step=0.05)

# --- ⚙️ DÜZELTİLMİŞ MATEMATİK MOTORU ---
ev_ppg = ev_ic_puan / ev_ic_mac
dep_ppg = dep_dis_puan / dep_dis_mac
ev_gol_ort = ev_toplam_gol / ev_ic_mac
dep_gol_ort = dep_toplam_gol / dep_dis_mac
ev_yenen_ort = ev_toplam_yenen / ev_ic_mac
dep_yenen_ort = dep_toplam_yenen / dep_dis_mac

# Güç katsayıları
ev_toplam = (ev_son5 * 2) + (ev_ppg * 5) + (ev_gol_ort * 10) + ((3 - ev_yenen_ort) * 10) + (-ev_eksik * 5) + (ev_mot * 5)
dep_toplam = (dep_son5 * 2) + (dep_ppg * 5) + (dep_gol_ort * 10) + ((3 - dep_yenen_ort) * 10) + (-dep_eksik * 5) + (dep_mot * 5)

ev_toplam, dep_toplam = max(ev_toplam, 1), max(dep_toplam, 1)

# [YENİ] Beraberlik İhtimali Hesaplama Mantığı
# İki takımın güç farkı yüzdesi ne kadar azsa, beraberlik o kadar kuvvetlidir (Max %35, Min %10)
guc_farki_orani = abs(ev_toplam - dep_toplam) / (ev_toplam + dep_toplam)
x_olasilik = max(35 - (guc_farki_orani * 50), 10)

# Kalan yüzdeyi MS1 ve MS2 arasında adilce dağıtma
kalan_yuzde = 100 - x_olasilik
ms1_olasilik = (ev_toplam / (ev_toplam + dep_toplam)) * kalan_yuzde
ms2_olasilik = (dep_toplam / (ev_toplam + dep_toplam)) * kalan_yuzde

# Gol olasılıkları
ust_olasilik = min(max(((ev_gol_ort + dep_gol_ort) / 4) * 100, 0), 100)
kg_var_olasilik = min(max(((ev_gol_ort + dep_gol_ort) / (ev_yenen_ort + dep_yenen_ort + 1)) * 50, 0), 100)

# Value (Değer) Hesaplama
v_ms1 = ms1_olasilik - (100 / b_ms1)
v_x = x_olasilik - (100 / b_x)
v_ms2 = ms2_olasilik - (100 / b_ms2)

# Dinamik Yorum ve En İyi Öneri Seçimi
onerilen_bahis_adi = "Çifte Şans 1-X"
en_yuksek_value = max(v_ms1, v_x, v_ms2)
if en_yuksek_value == v_ms1 and v_ms1 > 0: onerilen_bahis_adi = "MS1"
elif en_yuksek_value == v_ms2 and v_ms2 > 0: onerilen_bahis_adi = "MS2"
elif en_yuksek_value == v_x and v_x > 0: onerilen_bahis_adi = "X (Beraberlik)"
elif ust_olasilik > 60: onerilen_bahis_adi = "2.5 Üst"


# --- SEKME 1: ANA ANALİZ EKRANI ---
with sekme1:
    st.markdown('<div class="badge">Sezgin Görmüş Veri Analiz V3.0</div>', unsafe_allow_html=True)
    st.title("📈 İleri Düzey Algoritmik Maç Analiz Platformu")
    
    col_sol, col_sag = st.columns([11, 10])
    with col_sol:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📊 Olasılık ve Dağılım Matrisi")
        fig = go.Figure(data=[go.Pie(labels=['MS1', 'X', 'MS2'], values=[ms1_olasilik, x_olasilik, ms2_olasilik], hole=.45, marker_colors=['#0f766e', '#d97706', '#be123c'])])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10), height=260)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎯 Gol ve Alternatif Pazarlar")
        c1, c2 = st.columns(2)
        with c1: st.metric(label="⚽ 2.5 Üst Olasılığı", value=f"%{ust_olasilik:.1f}"); st.progress(int(ust_olasilik))
        with c2: st.metric(label="🔥 Karşılıklı Gol Var", value=f"%{kg_var_olasilik:.1f}"); st.progress(int(kg_var_olasilik))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sag:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚨 Algoritmik Sinyal Merkezi")
        def sinyal_satiri(pazar, model_p, oran, value):
            renk = "signal-green" if value > 0 else "signal-red"
            durum = "DEĞERLİ (VALUE) 🔥" if value > 0 else "DEĞERSİZ ❌"
            st.markdown(f'<div style="padding: 10px 0; border-bottom: 1px solid #334155;"><div style="display:flex; justify-content:space-between;"><b>{pazar}</b><span class="{renk}">{durum}</span></div><div style="font-size:13px; color:#94a3b8; margin-top:4px;">Model: %{model_p:.1f} | Oran: {oran:.2f} | <b>Net Avantaj: <span class="{renk}">{value:+.2f}</span></b></div></div>', unsafe_allow_html=True)
        sinyal_satiri(f"{ev_sahibi} Galibiyeti (MS1)", ms1_olasilik, b_ms1, v_ms1)
        sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
        sinyal_satiri(f"{deplasman} Galibiyeti (MS2)", ms2_olasilik, b_ms2, v_ms2)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ANALİZİ KAYDETME BUTONU
        st.write("")
        if st.button("💾 Bu Analizi Arşive Kaydet"):
            if mac_kaydet(ev_sahibi, deplasman, ms1_olasilik, x_olasilik, ms2_olasilik, b_ms1, b_x, b_ms2, onerilen_bahis_adi, en_yuksek_value):
                st.success(f"✅ {ev_sahibi} vs {deplasman} analizi başarıyla günlüğe kaydedildi!")

    # Yapay Zeka Yorumu
    st.markdown('<div class="commentary-card">', unsafe_allow_html=True)
    st.subheader("🤖 Yapay Zeka & Algoritma Maç Yorumu")
    yorum_metni = f"**{ev_sahibi}** ({ev_ppg:.2f} iç saha PPG) ve **{deplasman}** ({dep_ppg:.2f} dış saha PPG) mücadelesinde modelimiz takımların son form durumlarını ve gol güçlerini hesapladı. "
    if en_yuksek_value > 0:
        yorum_metni += f"Piyasa oranları incelendiğinde, bu maçta bahis şirketinin oran hatası yaptığı bir **Value** fırsatı göze çarpıyor."
    else:
        yorum_metni += "Oranlar matematiksel sınırda, risk dağılımı dengeli görünüyor."
    
    st.write(yorum_metni)
    st.markdown(f'<div style="background-color: #1e1b4b; padding:12px; border-radius:10px; margin-top:10px;">🎯 <b>KUPON TAVSİYESİ:</b> {onerilen_bahis_adi} (Değer: {en_yuksek_value:+.2f})</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- SEKME 2: GEÇMİŞ ANALİZ GÜNLÜĞÜ VE TAKİP ---
with sekme2:
    st.title("🗂️ Analiz Günlüğü ve Performans Takip Odası")
    st.write("Kaydettiğin maçları buradan inceleyebilir, maçlar bittikçe sonuçlarını güncelleyebilirsin.")
    
    # Veriyi Oku
    df_logs = pd.read_csv(DB_FILE)
    
    if len(df_logs) == 0:
        st.info("Henüz kaydedilmiş bir analiz bulunmuyor. Ana panelden maç kaydedebilirsin kanka.")
    else:
        # Sonuç Güncelleme Alanı
        st.subheader("🔄 Maç Sonuçlarını Güncelle")
        col_m, col_s = st.columns(2)
        with col_m:
            secilen_mac_indeksi = st.selectbox("Sonucunu Değiştirmek İstediğin Maç", df_logs.index, format_func=lambda x: f"{df_logs.loc[x, 'Ev Sahibi']} vs {df_logs.loc[x, 'Deplasman']} ({df_logs.loc[x, 'Tarih']})")
        with col_s:
            yeni_sonuc = st.selectbox("Maçın Sonucu Ne Oldu?", ["Bekliyor", "KAZANDI ✅", "KAYBETTİ ❌"])
        
        if st.button("💾 Sonucu Onayla ve Güncelle"):
            df_logs.loc[secilen_mac_indeksi, "Sonuc"] = yeni_sonuc
            df_logs.to_csv(DB_FILE, index=False)
            st.success("Maç sonucu başarıyla güncellendi!")
            st.rerun()
            
        st.write("---")
        st.subheader("📊 Güncel Arşiv Listesi")
        st.dataframe(df_logs, use_container_width=True)
        
        # Küçük Bir Başarı İstatistiği Grafiği
        st.write("---")
        st.subheader("📉 Sistem Başarı Metrikleri")
        kazananlar = len(df_logs[df_logs["Sonuc"] == "KAZANDI ✅"])
        kaybedenler = len(df_logs[df_logs["Sonuc"] == "KAYBETTİ ❌"])
        toplam_sonuclanan = kazananlar + kaybedenler
        
        if toplam_sonuclanan > 0:
            basari_orani = (kazananlar / toplam_sonuclanan) * 100
            st.metric(label="🏆 Toplam Sistem Başarı Yüzdesi", value=f"%{basari_orani:.1f}")
            
            # Bar Grafiği
            fig_perf = go.Figure([go.Bar(x=['Kazanan Tahminler', 'Kaybeden Tahminler'], values=[kazananlar, kaybedenler], marker_color=['#10b981', '#ef4444'])])
            fig_perf.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Sistem başarı oranının hesaplanması için en az 1 maçın sonucunu 'KAZANDI' veya 'KAYBETTİ' olarak güncellemelisin kanka.")
