import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# "anthropic" paketi opsiyoneldir — sadece kullanıcı gerçek AI yorumunu açarsa gerekir.
# Kurulu değilse uygulama hiçbir şekilde bozulmaz, sadece o özellik devre dışı kalır.
try:
    import anthropic
    ANTHROPIC_PAKETI_MEVCUT = True
except ImportError:
    ANTHROPIC_PAKETI_MEVCUT = False

# =====================================================================================
# SEZGİN GÖRMÜŞ VERİ ANALİZİ v10.0 — SADE ANALİZ SÜRÜMÜ
# -------------------------------------------------------------------------------------
# v9.0'a göre değişen: Arşiv Paneli ve Kupon Odası tamamen kaldırıldı (kullanıcı isteği
# üzerine — artık maç takibine ihtiyaç yok, sadece tek seferlik analiz gerekiyor).
# CSV veritabanı, migration, Mac_ID sistemi ve kupon otomasyonu bu yüzden bu sürümde
# yok. Uygulama artık tamamen durumsuz (stateless) tek ekranlık bir analiz aracı.
# Matematik motoru (Poisson + Dixon-Coles, overround normalizasyonu, value hesabı,
# zenginleştirilmiş yorum motoru) v9.0 ile birebir aynı.
# =====================================================================================

st.set_page_config(page_title="Sezgin Görmüş Veri Analizi v10.0", page_icon="🔮", layout="wide")


# --------------------------------------------------------------------------------
# 📐 POISSON + DIXON-COLES SKOR MATRİSİ MOTORU
# --------------------------------------------------------------------------------
def poisson_olasilik(k, lmbda):
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)


def dixon_coles_tau(x, y, lam, mu, rho):
    """Düşük skorlu hücrelerde (0-0,0-1,1-0,1-1) bağımsız Poisson varsayımını hafifçe
    düzeltir. rho negatif olduğunda 0-0 ve 1-1 biraz daha olası, 1-0 ve 0-1 biraz daha
    az olası hale gelir — literatürde gözlenen gerçek maç davranışına daha yakın."""
    if x == 0 and y == 0:
        return max(1e-6, 1 - (lam * mu * rho))
    if x == 0 and y == 1:
        return 1 + (lam * rho)
    if x == 1 and y == 0:
        return 1 + (mu * rho)
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def skor_matrisi_olustur(ev_xg, dep_xg, rho=-0.08, max_gol=9):
    """Tek bir (max_gol+1)x(max_gol+1) olasılık matrisi üretir. Tüm pazarlar
    (1X2, alt/üst, KG, en olası skor) bu TEK matristen türetilir."""
    grid = np.zeros((max_gol + 1, max_gol + 1))
    for i in range(max_gol + 1):
        p_i = poisson_olasilik(i, ev_xg)
        for j in range(max_gol + 1):
            p_j = poisson_olasilik(j, dep_xg)
            tau = dixon_coles_tau(i, j, ev_xg, dep_xg, rho)
            grid[i, j] = p_i * p_j * tau
    toplam = grid.sum()
    if toplam > 0:
        grid = grid / toplam  # kesme (truncation) + tau düzeltmesinden doğan sapmayı normalize et
    return grid


def matris_metrikleri(grid):
    max_gol = grid.shape[0] - 1
    ms1 = x = ms2 = 0.0
    ust15 = ust25 = ust35 = 0.0
    kg_var = 0.0
    en_iyi_p, en_iyi_i, en_iyi_j = 0.0, 0, 0

    for i in range(max_gol + 1):
        for j in range(max_gol + 1):
            p = grid[i, j]
            if i > j:
                ms1 += p
            elif i == j:
                x += p
            else:
                ms2 += p

            toplam_gol = i + j
            if toplam_gol > 1:
                ust15 += p
            if toplam_gol > 2:
                ust25 += p
            if toplam_gol > 3:
                ust35 += p
            if i > 0 and j > 0:
                kg_var += p

            if p > en_iyi_p:
                en_iyi_p, en_iyi_i, en_iyi_j = p, i, j

    return {
        "ms1": ms1 * 100, "x": x * 100, "ms2": ms2 * 100,
        "ust15": ust15 * 100, "alt15": (1 - ust15) * 100,
        "ust25": ust25 * 100, "alt25": (1 - ust25) * 100,
        "ust35": ust35 * 100, "alt35": (1 - ust35) * 100,
        "kg_var": kg_var * 100, "kg_yok": (1 - kg_var) * 100,
        "en_iyi_skor": (en_iyi_i, en_iyi_j), "en_iyi_skor_p": en_iyi_p * 100,
    }


# --------------------------------------------------------------------------------
# 💰 ORAN → ADİL OLASILIK (OVERROUND NORMALİZASYONU)
# --------------------------------------------------------------------------------
def adil_olasiliklar(oranlar):
    """Bülten oranlarının kâr marjını (overround) çıkarıp, olasılıkları
    toplamı %100 olacak şekilde normalize eder. Bunu yapmadan ham 1/oran ile
    model olasılığını kıyaslamak sistematik olarak yanıltıcıdır."""
    imp = [1.0 / o if o > 0 else 0.0 for o in oranlar]
    toplam = sum(imp)
    if toplam <= 0:
        return [0.0 for _ in oranlar]
    return [(i / toplam) * 100 for i in imp]


# --------------------------------------------------------------------------------
# 🧠 STRATEJİ MOTORU: EN İYİ BAHSİ BELİRLE (value + olasılık eşiği birlikte)
# --------------------------------------------------------------------------------
def en_iyi_bahsi_belirle(metrics, ev_sahibi, deplasman, b_ms1, b_x, b_ms2,
                          diger_pazar_aktif, b_ust25=None, b_alt25=None,
                          b_kgvar=None, b_kgyok=None,
                          min_value_puani=2.0, min_olasilik=35.0):
    """Dönüş: (bahis_etiketi, pazar_tipi, value_puani veya None, model_olasiligi veya None)"""
    fair_ms1, fair_x, fair_ms2 = adil_olasiliklar([b_ms1, b_x, b_ms2])

    adaylar = [
        (metrics["ms1"] - fair_ms1, f"Maç Sonucu 1 ({ev_sahibi})", "Maç Sonucu", metrics["ms1"]),
        (metrics["x"] - fair_x, "Beraberlik (X)", "Maç Sonucu", metrics["x"]),
        (metrics["ms2"] - fair_ms2, f"Maç Sonucu 2 ({deplasman})", "Maç Sonucu", metrics["ms2"]),
    ]

    if diger_pazar_aktif and b_ust25 and b_alt25 and b_kgvar and b_kgyok:
        fair_ust25, fair_alt25 = adil_olasiliklar([b_ust25, b_alt25])
        fair_kgvar, fair_kgyok = adil_olasiliklar([b_kgvar, b_kgyok])
        adaylar += [
            (metrics["ust25"] - fair_ust25, "2.5 Üst", "2.5 Alt/Üst", metrics["ust25"]),
            (metrics["alt25"] - fair_alt25, "2.5 Alt", "2.5 Alt/Üst", metrics["alt25"]),
            (metrics["kg_var"] - fair_kgvar, "Karşılıklı Gol Var (KG Var)", "Karşılıklı Gol", metrics["kg_var"]),
            (metrics["kg_yok"] - fair_kgyok, "Karşılıklı Gol Yok (KG Yok)", "Karşılıklı Gol", metrics["kg_yok"]),
        ]
        gecerli = [a for a in adaylar if a[3] >= min_olasilik and a[0] >= min_value_puani]
        if gecerli:
            en_iyi = max(gecerli, key=lambda a: a[0])
            return en_iyi[1], en_iyi[2], en_iyi[0], en_iyi[3]
        return "RİSKLİ MÜSABAKA (PAS) ⚠️", "Yok", None, None

    # Diğer pazarlar için oran girilmemişse: 1X2'de value ara, yoksa ham olasılık eşiği kullan
    ms_gecerli = [a for a in adaylar if a[3] >= min_olasilik and a[0] >= min_value_puani]
    if ms_gecerli:
        en_iyi = max(ms_gecerli, key=lambda a: a[0])
        return en_iyi[1], en_iyi[2], en_iyi[0], en_iyi[3]

    if metrics["ust25"] >= 60:
        return "2.5 Üst", "2.5 Alt/Üst", None, metrics["ust25"]
    if metrics["alt25"] >= 60:
        return "2.5 Alt", "2.5 Alt/Üst", None, metrics["alt25"]
    if metrics["kg_var"] >= 60:
        return "Karşılıklı Gol Var (KG Var)", "Karşılıklı Gol", None, metrics["kg_var"]
    if metrics["kg_yok"] >= 60:
        return "Karşılıklı Gol Yok (KG Yok)", "Karşılıklı Gol", None, metrics["kg_yok"]

    return "RİSKLİ MÜSABAKA (PAS) ⚠️", "Yok", None, None


# --------------------------------------------------------------------------------
# 🧾 ZENGİNLEŞTİRİLMİŞ KURAL TABANLI YORUM MOTORU
# -------------------------------------------------------------------------------
# Bu bir dil modeli değildir — internet gerektirmez, her zaman çalışır. Ama artık tek
# cümle değil; hücum/savunma dengesi, form, sakatlık durumu, value mantığı ve en olası
# skor gibi gerçekten hesaplanan verileri harmanlayan 4 paragraflık bir "maç önizlemesi"
# üretir. Kullanıcı isterse (aşağıdaki gerçek AI seçeneğiyle) bunun yerine kendi API
# anahtarıyla Claude'dan daha akıcı bir yorum da ürettirebilir.
# --------------------------------------------------------------------------------
def kural_tabanli_yorum_uret(ev_sahibi, deplasman, ev_xg, dep_xg, ev_ppg, dep_ppg,
                              ev_cs, dep_cs, ev_kritik_eksik, dep_kritik_eksik,
                              ev_normal_eksik, dep_normal_eksik,
                              metrics, en_iyi_bahis, pazar_t, value_puani):

    # --- 1) GİRİŞ + GENEL GÜÇ DENGESİ ---
    girisler = [
        f"Analiz motorumuz {ev_sahibi} ve {deplasman} mücadelesini derinlemesine inceledi.",
        f"{ev_sahibi} ile {deplasman} arasındaki bu kapışmada veriler ilginç bir tablo sunuyor.",
        f"Sistemimiz, {ev_sahibi}'nin evindeki istatistiklerini ve {deplasman}'ın deplasman karnesini kıyasladı.",
        f"{ev_sahibi} - {deplasman} maçına dair modelin ortaya çıkardığı tablo şöyle:",
    ]

    fark = ev_xg - dep_xg
    if fark > 0.7:
        guc_yorumu = (f"Ev sahibi {ev_sahibi}, xG verilerine göre {deplasman} savunmasını zorlayabilecek "
                      f"belirgin bir hücum üstünlüğüne sahip (Ev xG: {ev_xg:.2f} — Dep xG: {dep_xg:.2f}).")
    elif fark > 0.2:
        guc_yorumu = (f"İki takımın hücum hatları birbirine yakın olsa da {ev_sahibi} saha avantajıyla "
                      f"hafif önde görünüyor (Ev xG: {ev_xg:.2f} — Dep xG: {dep_xg:.2f}).")
    elif fark < -0.7:
        guc_yorumu = (f"Dikkat çekici bir veri var: {deplasman} deplasmanda olmasına rağmen xG bazında "
                      f"maçı domine etme potansiyeli taşıyor (Ev xG: {ev_xg:.2f} — Dep xG: {dep_xg:.2f}).")
    elif fark < -0.2:
        guc_yorumu = (f"{deplasman}, dış saha performansına rağmen gol beklentisinde hafif önde "
                      f"(Ev xG: {ev_xg:.2f} — Dep xG: {dep_xg:.2f}).")
    else:
        guc_yorumu = (f"Maçın genel karakteri dengeli; iki takım da birbirine karşı belirgin bir "
                      f"üstünlük kurmakta zorlanabilir (Ev xG: {ev_xg:.2f} — Dep xG: {dep_xg:.2f}).")

    if ev_ppg > dep_ppg * 1.3:
        puan_yorumu = f"Puan ortalamaları da bu tabloyu destekliyor: {ev_sahibi} maç başına {ev_ppg:.2f} puan toplarken {deplasman} sadece {dep_ppg:.2f} puanda kalıyor."
    elif dep_ppg > ev_ppg * 1.3:
        puan_yorumu = f"Puan ortalamalarına bakınca tablo tersine dönüyor: {deplasman} maç başına {dep_ppg:.2f} puan alırken {ev_sahibi} {ev_ppg:.2f} puanda kalıyor — sürpriz ihtimali göz ardı edilmemeli."
    else:
        puan_yorumu = f"Puan ortalamaları da ({ev_sahibi}: {ev_ppg:.2f}, {deplasman}: {dep_ppg:.2f}) iki takımın birbirine yakın seviyede olduğunu gösteriyor."

    paragraf_1 = f"{random.choice(girisler)} {guc_yorumu} {puan_yorumu}"

    # --- 2) FORM & KADRO DURUMU ---
    def savunma_notu(takim, cs):
        if cs >= 4:
            return f"{takim} son 5 maçının {cs} tanesinde kalesini gole kapattı; savunma organizasyonu bu aralıkta oldukça sağlam görünüyor."
        if cs >= 2:
            return f"{takim} son 5 maçta {cs} kez kalesini gole kapatarak ortalamanın üstünde bir savunma istikrarı sergiledi."
        return f"{takim} son 5 maçta sadece {cs} kez gol yemedi; savunmada zaman zaman açıklar verdiği söylenebilir."

    def sakatlik_notu(takim, kritik, normal):
        if kritik >= 2:
            return f"{takim} kadrosunda {kritik} kritik eksik bulunması (olası ilk 11 oyuncusu) hücum ve savunma gücünü belirgin şekilde düşürüyor."
        if kritik == 1:
            return f"{takim} kadrosunda 1 kritik eksik var; bu tek başına belirleyici olmasa da göz ardı edilmemeli."
        if normal >= 3:
            return f"{takim} kadrosunda kritik bir eksik yok ama {normal} rotasyon oyuncusunun yokluğu derinliği bir miktar zayıflatıyor."
        return f"{takim} kadrosunda dikkat çekici bir eksik görünmüyor; bu tahmini güçlendiren bir faktör."

    paragraf_2 = (
        f"{savunma_notu(ev_sahibi, ev_cs)} {savunma_notu(deplasman, dep_cs)} "
        f"{sakatlik_notu(ev_sahibi, ev_kritik_eksik, ev_normal_eksik)} "
        f"{sakatlik_notu(deplasman, dep_kritik_eksik, dep_normal_eksik)}"
    )

    # --- 3) VALUE / ORAN MANTIĞI ---
    bahis_mantigi = {
        "MS1": "Ev sahibi avantajı ve gol beklentisi (xG) verileri bu tercihi öne çıkarıyor.",
        "MS2": "Deplasmanın hücum verimliliği ve rakip savunmasındaki zafiyetler bu seçimi destekliyor.",
        "Beraberlik": "İki takımın gücünün birbirine çok yakın çıkması modelin beraberlik ihtimaline daha fazla ağırlık vermesine yol açıyor.",
        "Üst": "İki ekibin de skor odaklı bir oyun anlayışına sahip olması gol bareminin aşılma ihtimalini artırıyor.",
        "Alt": "Her iki takımın da savunma disiplini, düşük skorlu bir maç ihtimalini güçlendiriyor.",
        "KG Var": "Savunma istatistiklerindeki zayıflıklar her iki kalenin de gol görme ihtimalini artırıyor.",
        "KG Yok": "En az bir tarafın sağlam savunma verileri, karşılıklı gol ihtimalini düşürüyor.",
    }
    secilen_mantik = next((v for k, v in bahis_mantigi.items() if k in en_iyi_bahis),
                          "Sistemimiz bu markette olasılık/oran dengesizliği tespit etti.")

    if value_puani is not None:
        if value_puani >= 5:
            value_yorumu = (f"Bülten oranlarına göre bu seçimde belirgin bir değer (value) var: model, oranın "
                            f"ima ettiği adil olasılıktan yaklaşık {value_puani:.1f} puan daha yüksek bir ihtimal hesaplıyor.")
        elif value_puani > 0:
            value_yorumu = (f"Küçük bir value avantajı görünüyor (~{value_puani:.1f} puan); bu tek başına "
                            f"güçlü bir sinyal sayılmaz ama diğer verilerle birlikte tercihi destekliyor.")
        else:
            value_yorumu = "Bu seçimde belirgin bir oran/value avantajı yok; tercih daha çok ham olasılığa dayanıyor."
    elif pazar_t == "Yok":
        value_yorumu = "Model, hiçbir pazarda yeterince güçlü bir sinyal ya da value bulamadı; bu yüzden temkinli bir duruş öneriliyor."
    else:
        value_yorumu = "Bu pazar için oran girilmediğinden value hesaplanmadı; tahmin sadece model olasılığına dayanıyor."

    en_iyi_skor = metrics.get("en_iyi_skor", (0, 0))
    skor_yorumu = (f"Modelin en olası gördüğü skor {ev_sahibi} {en_iyi_skor[0]}-{en_iyi_skor[1]} {deplasman} "
                   f"(~%{metrics.get('en_iyi_skor_p', 0):.1f} olasılıkla — tek bir skor için bu normal bir orandır).")

    paragraf_3 = f"{secilen_mantik} {value_yorumu} {skor_yorumu}"

    # --- 4) KAPANIŞ / SORUMLULUK NOTU ---
    kapanislar = [
        "Yine de unutma: bu bir olasılık tahminidir, kesinlik iddiası taşımaz.",
        "Bahis kararların her zaman kendi risk toleransına bağlı olmalı; bu yalnızca istatistiksel bir görüş.",
        "Futbolun sürpriz doğası gereği, düşük olasılıklı sonuçlar da gerçekleşebilir — bankoya oynama.",
        "Model geçmiş verilere dayanır; sakatlık, taktik değişikliği gibi son dakika gelişmelerini her zaman yakalayamayabilir.",
    ]
    paragraf_4 = f"{random.choice(kapanislar)} (Tahmin: {en_iyi_bahis})"

    return [paragraf_1, paragraf_2, paragraf_3, paragraf_4]


# --------------------------------------------------------------------------------
# 🧠 GERÇEK AI İLE ZENGİNLEŞTİRME (OPSİYONEL — kullanıcının kendi API anahtarıyla)
# --------------------------------------------------------------------------------
def claude_ile_yorum_uret(api_anahtari, ev_sahibi, deplasman, ev_xg, dep_xg, metrics,
                           en_iyi_bahis, pazar_t, value_puani, b_ms1, b_x, b_ms2):
    """Kullanıcının kendi Anthropic API anahtarıyla gerçek bir Claude modelinden
    doğal, bağlama duyarlı bir maç yorumu üretir. Anahtar hiçbir yere kaydedilmez,
    sadece bu tek istekte kullanılır. Başarısız olursa çağıran taraf kural tabanlı
    yoruma otomatik olarak geri döner."""
    client = anthropic.Anthropic(api_key=api_anahtari)

    en_iyi_skor = metrics.get("en_iyi_skor", (0, 0))
    value_metni = f"{value_puani:.1f} puan" if value_puani is not None else "hesaplanmadı"

    prompt = f"""Sen deneyimli, ölçülü ve dürüst bir futbol veri analistisin. Aşağıdaki istatistiksel
model çıktılarına dayanarak Türkçe, akıcı, 3-4 cümlelik tek paragraflık bir maç önizleme yorumu yaz.

Maç: {ev_sahibi} (ev sahibi) - {deplasman} (deplasman)
Ev xG: {ev_xg:.2f} | Deplasman xG: {dep_xg:.2f}
MS1 olasılığı: %{metrics['ms1']:.1f} | Beraberlik: %{metrics['x']:.1f} | MS2 olasılığı: %{metrics['ms2']:.1f}
2.5 Üst olasılığı: %{metrics['ust25']:.1f} | KG Var olasılığı: %{metrics['kg_var']:.1f}
En olası skor: {en_iyi_skor[0]}-{en_iyi_skor[1]}
Bülten oranları: MS1 {b_ms1} / X {b_x} / MS2 {b_ms2}
Modelin önerdiği bahis: {en_iyi_bahis} ({pazar_t})
Value (oran karşısında model avantajı): {value_metni}

Kurallar:
- Kesin kazanma garantisi verme, "kesin", "banko" gibi ifadeler kullanma.
- Sayısal verileri doğal bir dille bağlama otur, listeleme yapma.
- Value düşükse veya yoksa bunu açıkça belirt, abartma.
- Sadece yorumu yaz, başlık veya giriş cümlesi ekleme."""

    yanit = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in yanit.content if hasattr(block, "text")).strip()


# --------------------------------------------------------------------------------
# 🏆 LİG VE TAKIM VERİTABANI (Ligue 1'deki boş string artıkları temizlendi)
# --------------------------------------------------------------------------------
LIG_VERITABANI = {
    "Türkiye Trendyol Süper Lig": [
        "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir",
        "Kasımpaşa", "Göztepe", "Samsunspor", "Alanyaspor", "Rizespor", "Çorum FK",
        "Amed SK", "Erzurumspor FK", "Eyüpspor", "Gençlerbirliği", "Gaziantep FK",
        "Kocaelispor", "Konyaspor",
    ],
    "Türkiye Trendyol 1. Lig": [
        "Ankara Keçiörengücü", "Antalyaspor", "Bandırmaspor", "Batman Petrolspor",
        "Bodrum", "Boluspor", "Bursaspor", "Esenler Erokspor", "Fatih Karagümrük",
        "Iğdır", "İstanbulspor", "Kayserispor", "Manisa", "Mardin 1969", "Muğlaspor",
        "Pendikspor", "Sarıyer", "Sivasspor", "Ümraniyespor", "Vanspor",
    ],
    "İngiltere Premier Lig": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion",
        "Chelsea", "Coventry City", "Crystal Palace", "Everton", "Fulham",
        "Hull City", "Ipswich Town", "Leeds United", "Liverpool", "Manchester City",
        "Manchester United", "Newcastle United", "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
    ],
    "İngiltere Championship": [
        "Blackburn Rovers", "Bristol City", "Burnley", "Cardiff City", "Birmingham City",
        "Derby County", "Lincoln City", "Bolton Wanderers", "Charlton Athletic",
        "Middlesbrough", "Millwall", "Norwich City", "Portsmouth", "Preston North End",
        "Queens Park Rangers", "Sheffield United", "Stoke City", "Swansea City",
        "Watford", "West Bromwich Albion", "Wolverhampton Wanderers", "West Ham United",
        "Wrexham", "Southampton",
    ],
    "İtalya Serie B": [
        "Verona", "Avellino", "Carrarese", "Catanzaro", "Cesena", "Padova",
        "Virtus Entella", "Empoli", "Juve Stabia", "Mantova", "Modena", "Pisa",
        "LR Vicenza", "Salernitana", "Sampdoria", "Benevento", "Arezzo", "Sudtirol",
    ],
    "İtalya Serie A": [
        "AC Milan", "AS Roma", "Atalanta", "Bologna", "Cagliari", "Como", "Frosinone",
        "Fiorentina", "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Sassuolo",
        "Napoli", "Palermo", "Parma", "Torino", "Udinese", "Venezia", "Monza",
    ],
    "İspanya Segunda Division": [
        "Albacete", "Almeria", "Burgos", "Cadiz", "Andorra", "Castellon", "Cordoba",
        "Celta Fortuna", "Eibar", "Ceuta", "Eldense", "Girona", "Granada", "Las Palmas",
        "Leganés", "Mallorca", "Oviedo", "Real Sociedad B", "Tenerife", "Sabadell",
        "Sporting Gijón", "Valladolid",
    ],
    "İspanya La Liga": [
        "Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Celta Vigo",
        "Espanyol", "Getafe", "Elche", "Deportivo La Coruña", "Leganes", "Mallorca",
        "Osasuna", "Racing Santander", "Rayo Vallecano", "Real Betis", "Real Madrid",
        "Real Oviedo", "Real Sociedad", "Sevilla", "Valencia", "Levante", "Villarreal",
    ],
    "Almanya Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Münih", "Borussia Dortmund",
        "Borussia Mönchengladbach", "Eintracht Frankfurt", "Freiburg", "Elversberg",
        "Hoffenheim", "Hamburg", "Mainz 05", "RB Leipzig", "Köln", "Paderborn 07",
        "Stuttgart", "Union Berlin", "Schalke 04", "Werder Bremen",
    ],
    "Almanya Bundesliga 2": [
        "Hertha BSC", "Arminia Bielefeld", "VfL Bochum", "Eintracht Braunschweig",
        "Energie Cottbus", "Darmstadt 98", "Dinamo Dresden", "Greuther Fürth",
        "Hannover 96", "1. FC Heidenheim", "1. FC Kaiserslautern", "Karlsruher SC",
        "Holstein Kiel", "1. FC Magdeburg", "1. FC Nürnberg", "VfL Osnabrück",
        "FC St. Pauli", "VfL Wolfsburg",
    ],
    "Hollanda Eredivisie": [
        "ADO Den Haag", "Ajax", "AZ", "Cambuur", "Excelsior", "Feyenoord",
        "Fortuna Sittard", "Go Ahead Eagles", "Groningen", "Heerenveen", "N.E.C.",
        "PEC Zwolle", "PSV", "Sparta Rotterdam", "Telstar", "Twente", "Utrecht", "Willem II",
    ],
    "Hollanda Eerste Divisie": [
        "Jong Ajax", "Almere City", "Jong AZ", "De Graafschap", "Den Bosch",
        "FC Dordrecht", "FC Eindhoven", "Emmen", "Helmond", "Heracles", "Maastricht",
        "NAC Breda", "Jong PSV", "Waalwijk", "Roda", "TOP Oss", "Jong Utrecht",
        "Vitesse", "Volendam", "VVV Venlo",
    ],
    "Fransa Ligue 1": [
        "Angers", "Brest", "Le Mans", "Lens", "Lille", "Lorient", "Lyon", "Marseille",
        "Monaco", "Paris", "Paris Saint-Germain", "Rennes", "Strasbourg", "Toulouse", "Troyes",
    ],
    "Fransa Ligue 2": [
        "Annecy", "Boulogne", "Clermont", "Dijon", "Dunkerque", "Grenoble", "Guingamp",
        "Lavallois", "Metz", "Montpellier", "Nancy", "Nantes", "Pau", "Red Star",
        "Reims", "Rodez", "St Etienne", "Sochaux",
    ],
}

# --------------------------------------------------------------------------------
# 🎨 CSS (tek sefer tanımlı — v8.0'daki tekrar eden blok kaldırıldı)
# --------------------------------------------------------------------------------
st.markdown("""
    <style>
    .main { background-color: #070a13; }
    [data-testid="stSidebar"] { background-color: #0d1324; border-right: 1px solid #1e293b; padding-top: 20px; }
    .premium-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border-radius: 15px !important; padding: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 20px !important; border: 1px solid #334155 !important; color: white !important;
    }
    .badge {
        background: #3D9DF3 !important; color: #fff !important; padding: 5px 15px !important;
        border-radius: 20px !important; font-weight: bold !important; display: inline-block !important;
        margin-bottom: 15px !important;
    }
    .signal-green { color: #10b981 !important; font-weight: bold; }
    .signal-red { color: #ef4444 !important; font-weight: bold; }
    .yorum-box { background-color: #0b1329; border-left: 4px solid #8b5cf6; padding: 18px; border-radius: 8px;
        font-size: 15px; color: #e2e8f0; line-height: 1.6; margin-top: 15px; }
    .k-card { border-radius: 12px; padding: 16px; margin-bottom: 16px; background-color: #0f172a;
        border: 1px solid #334155; position: relative; overflow: hidden; }
    .k-tuttu { background: #10b981; }
    .k-yatti { background: #ef4444; }
    .k-bekliyor { background: #f59e0b; }
    .m-satir { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e293b; font-size: 16px; }
    .m-satir:last-child { border-bottom: none; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 📋 SIDEBAR — katlanabilir bölümlere ayrıldı
# --------------------------------------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 10px;'>📋 Veri Girişi</h2>",
                     unsafe_allow_html=True)

with st.sidebar.expander("🏟️ Müsabaka Seçimi", expanded=True):
    lig_listesi = list(LIG_VERITABANI.keys()) + ["🌍 Diğer / Listede Olmayan Lig"]
    secilen_lig = st.selectbox("Ligi Seç:", lig_listesi)

    if secilen_lig == "🌍 Diğer / Listede Olmayan Lig":
        ev_sahibi = st.text_input("Ev Sahibi Takım Adı:", "Ev Sahibi")
        deplasman = st.text_input("Deplasman Takım Adı:", "Deplasman")
    else:
        takimlar = LIG_VERITABANI[secilen_lig]
        ev_secenekleri = ["✍️ Kendim Yazacağım..."] + takimlar
        varsayilan_ev_idx = 1 if len(takimlar) >= 1 else 0
        varsayilan_dep_idx = 2 if len(takimlar) >= 2 else 0
        ev_secim = st.selectbox("Ev Sahibi Takım:", ev_secenekleri, index=varsayilan_ev_idx)
        ev_sahibi = st.text_input("Ev Sahibi Takım Adını Gir:", "Ev Sahibi") if ev_secim == "✍️ Kendim Yazacağım..." else ev_secim

        dep_secenekleri = ["✍️ Kendim Yazacağım..."] + takimlar
        dep_secim = st.selectbox("Deplasman Takımı:", dep_secenekleri, index=varsayilan_dep_idx)
        deplasman = st.text_input("Deplasman Takım Adını Gir:", "Deplasman") if dep_secim == "✍️ Kendim Yazacağım..." else dep_secim

with st.sidebar.expander("🌍 Lig & Ağırlıklandırma", expanded=False):
    lig_ort_gol = st.slider("Lig Geneli Maç Başına Gol Ortalaması", 1.50, 4.00, 2.80, step=0.05)
    form_agirligi = st.slider("Son 5 Maçın (Form) Etki Oranı (%)", 20, 60, 35) / 100
    sezon_agirligi = 1.0 - form_agirligi
    rho_dixon_coles = st.slider(
        "Dixon-Coles Korelasyon Katsayısı (ρ)", -0.20, 0.00, -0.08, step=0.01,
        help="Düşük skorlu maçlardaki (0-0, 1-0, 0-1, 1-1) gerçek hayat korelasyonunu modele ekler. "
             "0'a yakın değer, bağımsız Poisson varsayımına (v8.0 davranışı) döner."
    )

with st.sidebar.expander("🏠 Ev Sahibi İstatistikleri", expanded=False):
    ev_ic_mac = st.number_input("İç Sahada Oynadığı Maç", min_value=1, value=1)
    ev_ic_puan = st.number_input("İç Sahada Topladığı Puan", min_value=0, value=1)
    ev_toplam_gol = st.number_input("İç Sahada Attığı Toplam Gol", min_value=0, value=1)
    ev_toplam_yenen = st.number_input("İç Sahada Yediği Toplam Gol", min_value=0, value=1)
    ev_son5_attigi = st.number_input("Son 5 Maçta Attığı Gol (Form)", min_value=0, value=1)
    ev_son5_yedigi = st.number_input("Son 5 Maçta Yediği Gol (Form)", min_value=0, value=1)
    ev_cs = st.slider("Ev Sahibi Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 2)
    ev_onem = st.slider("Ev Sahibinin Maç Önem Derecesi (1-5)", 1, 5, 5)

with st.sidebar.expander("🚀 Deplasman İstatistikleri", expanded=False):
    dep_dis_mac = st.number_input("Dış Sahada Oynadığı Maç", min_value=1, value=1)
    dep_dis_puan = st.number_input("Dış Sahada Topladığı Puan", min_value=0, value=1)
    dep_toplam_gol = st.number_input("Dış Sahada Attığı Toplam Gol", min_value=0, value=1)
    dep_toplam_yenen = st.number_input("Dış Sahada Yediği Toplam Gol", min_value=0, value=1)
    dep_son5_attigi = st.number_input("Son 5 Maçta Dışarıda Attığı Gol", min_value=0, value=1)
    dep_son5_yedigi = st.number_input("Son 5 Maçta Dışarıda Yediği Gol", min_value=0, value=1)
    dep_cs = st.slider("Deplasman Son 5 Maçta Gol Yemediği Maç Sayısı", 0, 5, 1)
    dep_onem = st.slider("Deplasmanın Maç Önem Derecesi (1-5)", 1, 5, 4)

with st.sidebar.expander("🏆 Genel Lig İstatistikleri", expanded=False):
    col_genel1, col_genel2 = st.columns(2)
    with col_genel1:
        st.markdown("**Ev Sahibi (Genel)**")
        ev_genel_mac = st.number_input("Toplam Maç (Ev)", min_value=1, value=1, step=1)
        ev_genel_attigi = st.number_input("Toplam Attığı Gol (Ev)", min_value=0, value=1, step=1)
        ev_genel_yedigi = st.number_input("Toplam Yediği Gol (Ev)", min_value=0, value=1, step=1)
    with col_genel2:
        st.markdown("**Deplasman (Genel)**")
        dep_genel_mac = st.number_input("Toplam Maç (Dep)", min_value=1, value=1, step=1)
        dep_genel_attigi = st.number_input("Toplam Attığı Gol (Dep)", min_value=0, value=1, step=1)
        dep_genel_yedigi = st.number_input("Toplam Yediği Gol (Dep)", min_value=0, value=1, step=1)

with st.sidebar.expander("🛡️ Sağlık & Kadro Eksik Raporu", expanded=False):
    ev_kritik_eksik = st.slider("Ev Sahibi Kritik Eksik (As Kaleci, Golcü vb.)", 0, 3, 0)
    ev_normal_eksik = st.slider("Ev Sahibi Normal Eksik (Rotasyon Oyuncusu)", 0, 5, 1)
    dep_kritik_eksik = st.slider("Deplasman Kritik Eksik (As Kaleci, Golcü vb.)", 0, 3, 1)
    dep_normal_eksik = st.slider("Deplasman Normal Eksik (Rotasyon Oyuncusu)", 0, 5, 2)

with st.sidebar.expander("📊 Bülten Oranları (Maç Sonucu)", expanded=True):
    b_ms1 = st.number_input("Bülten Oranı: MS 1", min_value=1.01, value=1.85)
    b_x = st.number_input("Bülten Oranı: Beraberlik (X)", min_value=1.01, value=3.60)
    b_ms2 = st.number_input("Bülten Oranı: MS 2", min_value=1.01, value=3.40)

with st.sidebar.expander("📊 Diğer Pazar Oranları (Opsiyonel)", expanded=False):
    diger_pazar_aktif = st.checkbox(
        "2.5 Alt/Üst ve KG pazarları için de value hesapla",
        value=False,
        help="Açarsan bu pazarlar için de bahis şirketinin marjından arındırılmış 'adil olasılık' "
             "kıyaslaması yapılır. Kapalıysa bu pazarlarda sadece ham model olasılığı (%60 eşiği) kullanılır."
    )
    b_ust25 = b_alt25 = b_kgvar = b_kgyok = None
    if diger_pazar_aktif:
        b_ust25 = st.number_input("Bülten Oranı: 2.5 Üst", min_value=1.01, value=1.90)
        b_alt25 = st.number_input("Bülten Oranı: 2.5 Alt", min_value=1.01, value=1.90)
        b_kgvar = st.number_input("Bülten Oranı: KG Var", min_value=1.01, value=1.80)
        b_kgyok = st.number_input("Bülten Oranı: KG Yok", min_value=1.01, value=2.00)

with st.sidebar.expander("🧠 Gerçek AI ile Zenginleştir (Opsiyonel)", expanded=False):
    st.caption(
        "Varsayılan yorum kural tabanlıdır, internet gerektirmez ve her zaman çalışır. "
        "İstersen kendi Anthropic API anahtarınla gerçek bir Claude modelinden daha akıcı, "
        "bağlama duyarlı bir yorum ürettirebilirsin. Anahtarın hiçbir yere kaydedilmez, "
        "sadece bu oturumda ve tek bir istek için kullanılır."
    )
    ai_yorum_aktif = st.checkbox("Claude ile yorum üret", value=False)
    ai_api_anahtari = ""
    if ai_yorum_aktif:
        if not ANTHROPIC_PAKETI_MEVCUT:
            st.warning("`anthropic` paketi kurulu değil. Terminalde şunu çalıştır: `pip install anthropic`")
        ai_api_anahtari = st.text_input(
            "Anthropic API Anahtarı", type="password",
            help="console.anthropic.com üzerinden alabilirsin. Boş bırakırsan kural tabanlı yoruma dönülür.",
        )


# --------------------------------------------------------------------------------
# ⚠️ AYNI TAKIM KONTROLÜ
# --------------------------------------------------------------------------------
takim_ayni_mi = (
    ev_sahibi.strip().lower() == deplasman.strip().lower()
    and ev_sahibi.strip().lower() not in ("", "ev sahibi", "deplasman")
)


# --------------------------------------------------------------------------------
# ⚙️ xG HESAP MOTORU (v8.0 ile aynı mantık, sadeleştirilmiş)
# --------------------------------------------------------------------------------
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

ev_ic_xg = yarim_lig_ort * ev_ic_hucum_katsayi * dep_dis_savunma_katsayi
dep_dis_xg = yarim_lig_ort * dep_dis_hucum_katsayi * ev_ic_savunma_katsayi
ev_genel_xg = yarim_lig_ort * ev_genel_hucum_katsayi * dep_genel_savunma_katsayi
dep_genel_xg = yarim_lig_ort * dep_genel_hucum_katsayi * ev_genel_savunma_katsayi

ev_hucum_cezasi = min((ev_kritik_eksik * 0.06) + (ev_normal_eksik * 0.02), 0.25)
dep_hucum_cezasi = min((dep_kritik_eksik * 0.06) + (dep_normal_eksik * 0.02), 0.25)
ev_savunma_zafiyeti = min((ev_kritik_eksik * 0.04) + (ev_normal_eksik * 0.015), 0.20)
dep_savunma_zafiyeti = min((dep_kritik_eksik * 0.04) + (dep_normal_eksik * 0.015), 0.20)

ev_xg = ((ev_ic_xg * 0.5) + (ev_genel_xg * 0.5))
dep_xg = ((dep_dis_xg * 0.5) + (dep_genel_xg * 0.5))

ev_xg *= puan_denge_carpan * ev_onem_carpan
dep_xg *= (2.0 - puan_denge_carpan) * dep_onem_carpan

ev_xg *= (1 - ev_hucum_cezasi)
dep_xg *= (1 - dep_hucum_cezasi)

ev_xg *= (1 + dep_savunma_zafiyeti)
dep_xg *= (1 + ev_savunma_zafiyeti)

ev_xg = max(ev_xg, 0.05)
dep_xg = max(dep_xg, 0.05)

# --------------------------------------------------------------------------------
# 🎲 TEK SKOR MATRİSİ → TÜM PAZARLAR BURADAN TÜRETİLİYOR
# --------------------------------------------------------------------------------
grid = skor_matrisi_olustur(ev_xg, dep_xg, rho=rho_dixon_coles, max_gol=9)
metrics = matris_metrikleri(grid)

ms1_olasilik, x_olasilik, ms2_olasilik = metrics["ms1"], metrics["x"], metrics["ms2"]
gol_alt_1_5, gol_ust_1_5 = metrics["alt15"], metrics["ust15"]
gol_alt_2_5, gol_ust_2_5 = metrics["alt25"], metrics["ust25"]
gol_alt_3_5, gol_ust_3_5 = metrics["alt35"], metrics["ust35"]
ust_olasilik, kg_var_olasilik = metrics["ust25"], metrics["kg_var"]
tahmin_ev_skor, tahmin_dep_skor = metrics["en_iyi_skor"]

fair_ms1, fair_x, fair_ms2 = adil_olasiliklar([b_ms1, b_x, b_ms2])
v_ms1 = ms1_olasilik - fair_ms1
v_x = x_olasilik - fair_x
v_ms2 = ms2_olasilik - fair_ms2

en_iyi_bahis, pazar_t, value_puani, secim_olasiligi = en_iyi_bahsi_belirle(
    metrics, ev_sahibi, deplasman, b_ms1, b_x, b_ms2,
    diger_pazar_aktif, b_ust25, b_alt25, b_kgvar, b_kgyok,
)


# --------------------------------------------------------------------------------
# 📊 ANALİZ EKRANI
# --------------------------------------------------------------------------------
st.markdown('<div class="badge">SEZGİN GÖRMÜŞ VERİ ANALİZİ v10.0</div>', unsafe_allow_html=True)
st.title("📊 Matematiksel Tahmin Robotu")

if takim_ayni_mi:
    st.error("Ev sahibi ve deplasman takımı aynı görünüyor. Lütfen farklı iki takım seç.")
    st.stop()

# --------------------------------------------------------------------------------
# 🔢 KPI ÖZET SATIRI
# --------------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ev xG", f"{ev_xg:.2f}")
k2.metric("Deplasman xG", f"{dep_xg:.2f}")
k3.metric("En Olası Skor", f"{tahmin_ev_skor}-{tahmin_dep_skor}",
          f"%{metrics['en_iyi_skor_p']:.1f} olasılık", delta_color="off")
k4.metric("Model Tahmini", en_iyi_bahis.split(" (")[0],
          f"+{value_puani:.1f} puan value" if value_puani is not None else None)

# --------------------------------------------------------------------------------
# 📐 TAKIM KARŞILAŞTIRMA SKORLARI (radar grafiği için 0-100 ölçekli)
# --------------------------------------------------------------------------------
def olcek_0_100(deger, min_v, max_v):
    if max_v == min_v:
        return 50.0
    return max(0.0, min(100.0, (deger - min_v) / (max_v - min_v) * 100))

hucum_ev = olcek_0_100(ev_xg, 0, 3.0)
hucum_dep = olcek_0_100(dep_xg, 0, 3.0)
savunma_ev = 100 - olcek_0_100(ev_ic_savunma_ort, 0, 2.5)
savunma_dep = 100 - olcek_0_100(dep_dis_savunma_ort, 0, 2.5)
form_ev = olcek_0_100(ev_ppg, 0, 3.0)
form_dep = olcek_0_100(dep_ppg, 0, 3.0)
kadro_ev = 100 - olcek_0_100(ev_kritik_eksik * 2 + ev_normal_eksik, 0, 10)
kadro_dep = 100 - olcek_0_100(dep_kritik_eksik * 2 + dep_normal_eksik, 0, 10)

radar_kategorileri = ["Hücum", "Savunma", "Form (PPG)", "Kadro Sağlığı"]
radar_kategorileri_kapali = radar_kategorileri + [radar_kategorileri[0]]
ev_radar_kapali = [hucum_ev, savunma_ev, form_ev, kadro_ev] + [hucum_ev]
dep_radar_kapali = [hucum_dep, savunma_dep, form_dep, kadro_dep] + [hucum_dep]

# --------------------------------------------------------------------------------
# 🎲 SKOR TOPLAMI DAĞILIMI (0,1,2,3,4,5,6,7+ gol olasılıkları — tek grid'den)
# --------------------------------------------------------------------------------
_max_toplam = grid.shape[0] + grid.shape[1] - 1
_dagilim_ham = np.zeros(_max_toplam)
for _i in range(grid.shape[0]):
    for _j in range(grid.shape[1]):
        _dagilim_ham[_i + _j] += grid[_i, _j]
_esik = 7
gol_dagilimi_etiket = [str(k) for k in range(_esik)] + [f"{_esik}+"]
gol_dagilimi_deger = list(_dagilim_ham[:_esik] * 100) + [_dagilim_ham[_esik:].sum() * 100]

st.write("")
tab_genel, tab_matris, tab_value, tab_yorum = st.tabs(
    ["📊 Genel Bakış", "🔥 Skor Matrisi", "💰 Value Analizi", "🧠 Yorum"]
)

# =================================================================================
# TAB 1 — GENEL BAKIŞ: takım karşılaştırma radarı + 1X2 model/adil olasılık barı
# =================================================================================
with tab_genel:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🧭 Takım Güç Karşılaştırması")
        st.caption("0-100 ölçeğinde göreceli güç skoru (xG, savunma, form ve kadro sağlığından türetilir).")

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=ev_radar_kapali, theta=radar_kategorileri_kapali, fill='toself',
            name=ev_sahibi, line_color='#8b5cf6', fillcolor='rgba(139,92,246,0.25)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=dep_radar_kapali, theta=radar_kategorileri_kapali, fill='toself',
            name=deplasman, line_color='#f59e0b', fillcolor='rgba(245,158,11,0.20)',
        ))
        fig_radar.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#334155', showticklabels=True),
                angularaxis=dict(gridcolor='#334155'),
            ),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=380, margin=dict(t=20, b=20, l=50, r=50),
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_karsilastirma")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("⚖️ Maç Sonucu: Model vs. Bülten")
        st.caption("Model olasılığı ile bülten oranının marjdan arındırılmış (adil) olasılığı yan yana.")

        fig_1x2 = go.Figure()
        fig_1x2.add_trace(go.Bar(
            name='Model Olasılığı', x=['MS1', 'X', 'MS2'],
            y=[ms1_olasilik, x_olasilik, ms2_olasilik],
            marker_color='#8b5cf6', text=[f"%{v:.1f}" for v in [ms1_olasilik, x_olasilik, ms2_olasilik]],
            textposition='outside',
        ))
        fig_1x2.add_trace(go.Bar(
            name='Adil Olasılık (Oran)', x=['MS1', 'X', 'MS2'],
            y=[fair_ms1, fair_x, fair_ms2],
            marker_color='#475569', text=[f"%{v:.1f}" for v in [fair_ms1, fair_x, fair_ms2]],
            textposition='outside',
        ))
        fig_1x2.update_layout(
            barmode='group', template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=380, yaxis_title="Olasılık (%)", yaxis_range=[0, 100],
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=20, b=20, l=40, r=20),
        )
        st.plotly_chart(fig_1x2, use_container_width=True, key="bar_1x2")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("🥅 Maçtaki Toplam Gol Sayısı Dağılımı")
    st.caption("Skor matrisinden türetilmiştir; her çubuk o sayıda toplam gol olma olasılığını gösterir.")
    fig_dagilim = go.Figure(go.Bar(
        x=gol_dagilimi_etiket, y=gol_dagilimi_deger, marker_color='#3b82f6',
        text=[f"%{v:.1f}" for v in gol_dagilimi_deger], textposition='outside',
    ))
    fig_dagilim.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300, xaxis_title="Toplam Gol Sayısı", yaxis_title="Olasılık (%)",
        margin=dict(t=20, b=20, l=40, r=20),
    )
    st.plotly_chart(fig_dagilim, use_container_width=True, key="bar_gol_dagilimi")
    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================================
# TAB 2 — SKOR MATRİSİ: ısı haritası
# =================================================================================
with tab_matris:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("🔥 Skor Olasılık Isı Haritası")
    st.caption("Her hücre, o kesin skorun gerçekleşme olasılığını gösterir (Dixon-Coles düzeltmeli Poisson matrisi).")

    _boyut = 6
    _z = grid[:_boyut, :_boyut] * 100
    fig_heat = go.Figure(data=go.Heatmap(
        z=_z,
        x=[str(j) for j in range(_boyut)],
        y=[str(i) for i in range(_boyut)],
        colorscale=[[0, '#0b1329'], [0.5, '#6d28d9'], [1, '#22c55e']],
        text=np.round(_z, 1),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "#f8fafc"},
        hovertemplate=f"{ev_sahibi}: %{{y}} — {deplasman}: %{{x}}<br>Olasılık: %{{z:.1f}}%<extra></extra>",
        colorbar=dict(title="Olasılık (%)"),
    ))
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
        height=440,
        xaxis_title=f"{deplasman} — Gol Sayısı",
        yaxis_title=f"{ev_sahibi} — Gol Sayısı",
        margin=dict(t=20, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_skor")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.caption("En olası skor")
    st.markdown(
        f'<div style="text-align:center; font-size:28px; font-weight:bold; color:#f8fafc;">'
        f'{ev_sahibi} {tahmin_ev_skor} - {tahmin_dep_skor} {deplasman}</div>'
        f'<div style="text-align:center; font-size:13px; color:#94a3b8; margin-top:4px;">'
        f'~%{metrics["en_iyi_skor_p"]:.1f} olasılıkla (tek bir skor için bu normal bir orandır)</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================================
# TAB 3 — VALUE ANALİZİ: sinyal kartları + seçilen bahsin güven göstergesi
# =================================================================================
with tab_value:
    col_c, col_d = st.columns([1, 1])

    with col_c:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚨 Value Sinyalleri")
        st.caption("Value = Model olasılığı − bülten oranının marjdan arındırılmış (adil) olasılığı.")

        def sinyal_satiri(pazar, model_p, oran, value):
            renk = "#10b981" if value > 0 else "#ef4444"
            yuzde = min(abs(value) * 10, 100)
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.5); padding: 12px; border-radius: 10px; margin-bottom: 8px; border: 1px solid #1e293b;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: 600; color: #f8fafc;">{pazar}</span>
                    <span style="color: {renk}; font-weight: bold;">{value:+.2f}</span>
                </div>
                <div style="font-size: 11px; color: #64748b; margin-bottom: 5px;">Model: %{model_p:.1f} | Oran: {oran}</div>
                <div style="width: 100%; background: #334155; height: 4px; border-radius: 2px;">
                    <div style="width: {yuzde}%; background: {renk}; height: 4px; border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        sinyal_satiri(f"MS 1 ({ev_sahibi})", ms1_olasilik, b_ms1, v_ms1)
        sinyal_satiri("Beraberlik (X)", x_olasilik, b_x, v_x)
        sinyal_satiri(f"MS 2 ({deplasman})", ms2_olasilik, b_ms2, v_ms2)

        if diger_pazar_aktif:
            fair_ust25, fair_alt25 = adil_olasiliklar([b_ust25, b_alt25])
            fair_kgvar, fair_kgyok = adil_olasiliklar([b_kgvar, b_kgyok])
            sinyal_satiri("2.5 Üst", metrics["ust25"], b_ust25, metrics["ust25"] - fair_ust25)
            sinyal_satiri("2.5 Alt", metrics["alt25"], b_alt25, metrics["alt25"] - fair_alt25)
            sinyal_satiri("KG Var", metrics["kg_var"], b_kgvar, metrics["kg_var"] - fair_kgvar)
            sinyal_satiri("KG Yok", metrics["kg_yok"], b_kgyok, metrics["kg_yok"] - fair_kgyok)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎯 Seçilen Bahsin Güven Göstergesi")

        if secim_olasiligi is not None:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=secim_olasiligi,
                number={'suffix': "%"},
                title={'text': en_iyi_bahis.split(" (")[0], 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                    'bar': {'color': '#8b5cf6'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'steps': [
                        {'range': [0, 35], 'color': '#1e293b'},
                        {'range': [35, 60], 'color': '#334155'},
                        {'range': [60, 100], 'color': '#475569'},
                    ],
                },
            ))
            fig_gauge.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300,
                margin=dict(t=50, b=10, l=30, r=30), font={'color': '#f8fafc'},
            )
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_guven")
            if value_puani is not None:
                st.caption(f"Bu seçim, bülten oranının ima ettiği adil olasılığa göre ~{value_puani:.1f} puanlık bir value taşıyor.")
            else:
                st.caption("Bu seçim ham model olasılığına dayanıyor; bu pazar için value hesaplanmadı.")
        else:
            st.info("Model şu an hiçbir pazarda yeterince güçlü bir sinyal bulamadı — temkinli olmakta fayda var (PAS).")

        st.markdown('</div>', unsafe_allow_html=True)

# =================================================================================
# TAB 4 — YORUM: kural tabanlı ya da gerçek Claude yorumu + indirme
# =================================================================================
with tab_yorum:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)

    yorum_kaynagi_ai = False
    yorum_metni_ai = None

    if ai_yorum_aktif and ANTHROPIC_PAKETI_MEVCUT and ai_api_anahtari:
        with st.spinner("Claude maçı yorumluyor..."):
            try:
                yorum_metni_ai = claude_ile_yorum_uret(
                    ai_api_anahtari, ev_sahibi, deplasman, ev_xg, dep_xg, metrics,
                    en_iyi_bahis, pazar_t, value_puani, b_ms1, b_x, b_ms2,
                )
                yorum_kaynagi_ai = True
            except Exception as e:
                st.warning(f"AI yorumu üretilemedi ({e}); kural tabanlı yoruma dönülüyor.")

    if yorum_kaynagi_ai:
        st.subheader("🧠 Claude Maç Yorumu")
        st.caption("Bu yorum, girdiğin API anahtarıyla gerçek bir Claude modelinden üretildi.")
        st.markdown(f'<div class="yorum-box"><p>{yorum_metni_ai}</p></div>', unsafe_allow_html=True)
    else:
        st.subheader("🧾 Kural Tabanlı Otomatik Yorum")
        st.caption("Bu bir yapay zeka/dil modeli çıktısı değildir — internet gerektirmeyen, sabit kurallara dayalı bir analiz metnidir.")
        paragraflar = kural_tabanli_yorum_uret(
            ev_sahibi, deplasman, ev_xg, dep_xg, ev_ppg, dep_ppg,
            ev_cs, dep_cs, ev_kritik_eksik, dep_kritik_eksik,
            ev_normal_eksik, dep_normal_eksik,
            metrics, en_iyi_bahis, pazar_t, value_puani,
        )
        html_govde = "".join(f"<p>{p}</p>" for p in paragraflar)
        st.markdown(f'<div class="yorum-box">{html_govde}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    ozet_metni = (
        f"SEZGİN GÖRMÜŞ VERİ ANALİZİ — {ev_sahibi} vs {deplasman}\n"
        f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Ev xG: {ev_xg:.2f} | Deplasman xG: {dep_xg:.2f}\n"
        f"MS1: %{ms1_olasilik:.1f} | X: %{x_olasilik:.1f} | MS2: %{ms2_olasilik:.1f}\n"
        f"2.5 Üst: %{ust_olasilik:.1f} | KG Var: %{kg_var_olasilik:.1f}\n"
        f"En olası skor: {tahmin_ev_skor}-{tahmin_dep_skor}\n"
        f"Model Tahmini: {en_iyi_bahis} ({pazar_t})\n"
    )
    st.download_button(
        "📥 Bu Analiz Özetini İndir (.txt)",
        data=ozet_metni,
        file_name=f"analiz_{ev_sahibi}_vs_{deplasman}.txt",
        mime="text/plain",
        use_container_width=True,
    )
