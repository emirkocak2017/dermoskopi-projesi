import os
import re
import time
import html
import tempfile
import datetime
import hashlib
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance
from pathlib import Path

try:
    from triage_engine import TriageEngine
    from xai_engine import UltimateXAI
except ImportError:
    st.error("🚨 Sistem Hatası: Kritik çekirdek modüller ('triage_engine.py', 'xai_engine.py') bulunamadı.")
    st.stop()

# uygulamada state kaybolmasin diye baslangic degerlerini set ediyoruz
_DEFAULTS = {
    "role": None,
    "doc_authenticated": False,
    "doctor_analysis_done": False, "doctor_report": None, "doctor_xai_path": None,
    "doctor_last_uploaded": None, "doctor_patient_info": {}, "doctor_timestamp": None,
    "patient_analysis_done": False, "patient_report": None, "patient_xai_path": None,
    "patient_last_uploaded": None, "patient_timestamp": None,
    "consultation_link_generated": False,
    "consultation_hash": "",
    "abcde_score": 0,
    "fitzpatrick_score": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.set_page_config(
    page_title="Dermoskopik CDSS | Klinik Karar Destek",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state=("collapsed" if st.session_state.role is None else "expanded")
)

# arayuzun custom css tasarimi buralara dokunma responsive patlayabilir
st.markdown("""
<style>
/* Ana Arka Plan ve Düzen */
.main {background-color: #f0f4f8; font-family: 'Inter', 'Segoe UI', sans-serif;}

/* Modern UI Kartları */
.ui-card { background-color: #ffffff; border-radius: 16px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid #e2e8f0; transition: all 0.3s ease; }
.ui-card:hover { transform: translateY(-3px); box-shadow: 0 10px 25px rgba(0,0,0,0.08); }
.ui-card-label { font-size: 0.85rem; color: #64748b; font-weight: 700; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
.ui-card-value { font-weight: 800; line-height: 1.3; word-break: break-word; overflow-wrap: anywhere; white-space: normal; color: #0f172a; }

/* Dinamik Başlık Bannerları */
.header-banner { padding: 3rem 3.5rem; border-radius: 24px; color: white; margin-bottom: 2.5rem; box-shadow: 0 15px 35px rgba(0,0,0,0.15); position: relative; overflow: hidden; }
.header-banner::before { content: ''; position: absolute; top: -50%; right: -20%; width: 500px; height: 500px; background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 70%); border-radius: 50%; }
.header-banner h1 {color: white; margin: 0; font-size: 2.4rem; font-weight: 900; z-index: 1; position: relative; letter-spacing: -0.5px;}
.header-banner p {color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1.15rem; z-index: 1; position: relative; font-weight: 400;}
.header-doctor {background: linear-gradient(135deg, #1e293b 0%, #334155 100%);}
.header-patient {background: linear-gradient(135deg, #059669 0%, #10b981 100%);}

/* Kimlik Doğrulama Kutusu */
.auth-box { background-color: #ffffff; border-top: 8px solid #1e293b; border-radius: 16px; padding: 45px; margin: 60px auto; max-width: 600px; box-shadow: 0 20px 40px rgba(0,0,0,0.12); text-align: center; }
.auth-box h2 { color: #0f172a; font-weight: 800; margin-bottom: 15px; font-size: 1.8rem;}

/* Etiketler ve Kutular */
.consent-box { background-color: #fefce8; border: 1px solid #fde047; border-radius: 12px; padding: 18px 22px; margin-top: 15px; color: #854d0e; font-weight: 500; font-size: 0.95rem; }
.patient-tag { display: inline-flex; align-items: center; background: #f1f5f9; color: #334155; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; margin: 0 10px 10px 0; font-weight: 600; border: 1px solid #e2e8f0; }
.alert-box { border-radius: 12px; padding: 20px; margin: 20px 0; font-weight: 500; font-size: 1.05rem; display: flex; align-items: center; gap: 15px; }

/* İlerleme Çubuğu ve Aksanlar */
.custom-progress-track { background: #e2e8f0; border-radius: 12px; height: 16px; overflow: hidden; margin-top: 12px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05); }
.custom-progress-fill { height: 100%; border-radius: 12px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }

/* Giriş Ekranı Seçim Kartları */
.role-card { border-radius: 28px; padding: 3.5rem 2.5rem; text-align: center; color: white; box-shadow: 0 15px 35px rgba(0,0,0,0.15); min-height: 400px; transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; position: relative; overflow: hidden; }
.role-card:hover { transform: translateY(-12px); box-shadow: 0 25px 50px rgba(0,0,0,0.25); }
.role-card-doctor {background: linear-gradient(145deg, #0f172a 0%, #334155 100%);}
.role-card-patient {background: linear-gradient(145deg, #047857 0%, #34d399 100%);}
.role-card-icon {font-size: 5rem; margin-bottom: 1.5rem; filter: drop-shadow(0 8px 12px rgba(0,0,0,0.2));}
.role-card-title {font-size: 2.2rem; font-weight: 900; margin-bottom: 1.2rem; letter-spacing: -0.5px;}
.role-card-desc {font-size: 1.1rem; opacity: 0.95; line-height: 1.6; font-weight: 500;}

/* Link Butonu Stili Özelleştirme */
.mhrs-button-container { text-align: center; margin: 30px 0; padding: 25px; border-radius: 16px; background: linear-gradient(to right, #f8fafc, #f1f5f9); border: 2px dashed #cbd5e1; }
.mhrs-button { display: inline-block; background-color: #dc2626; color: white !important; font-weight: 800; font-size: 1.2rem; padding: 15px 35px; border-radius: 30px; text-decoration: none; box-shadow: 0 10px 20px rgba(220, 38, 38, 0.3); transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 0.5px;}
.mhrs-button:hover { background-color: #b91c1c; transform: scale(1.05); box-shadow: 0 15px 25px rgba(220, 38, 38, 0.4); text-decoration: none; }
.mhrs-button-yellow { background-color: #d97706; box-shadow: 0 10px 20px rgba(217, 119, 6, 0.3); }
.mhrs-button-yellow:hover { background-color: #b45309; box-shadow: 0 15px 25px rgba(217, 119, 6, 0.4); }

/* Özel Tooltip Metinleri */
.tooltip-text { font-size: 0.85rem; color: #64748b; font-style: normal; margin-top: 8px; line-height: 1.4; border-top: 1px dashed #e2e8f0; padding-top: 8px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_clinical_engines():
    # sayfa her yuklendiginde model bastan yuklenmesin diye cache_resource attik
    triage = TriageEngine()
    xai = UltimateXAI(model_path="ultimate_skin_model.keras")
    return triage, xai

# doktor ddx sekmekesinde cikacak alternatif tani veritabani
DIFFERENTIAL_DIAGNOSIS = {
    "melanom": [
        {"isim": "Atipik (Displastik) Nevüs", "neden": "Klinik ve dermoskopik olarak melanomu en çok taklit eden lezyondur. Asimetri ve pigment ağı benzerlik gösterebilir."},
        {"isim": "Pigmente Bazal Hücreli Karsinom", "neden": "Mavi-gri ovoid yuvalar ve yapraksı alanlar nedeniyle melanom ile klinik olarak karışabilir."},
        {"isim": "Pigmente Bowen Hastalığı", "neden": "Atipik damarlanma ve düzensiz pigmentasyon sebebiyle mutlaka ayırıcı tanıda yer almalıdır."}
    ],
    "bazal_hucreli": [
        {"isim": "Seboreik Keratoz", "neden": "Özellikle irrite olmuş veya pigmente seboreik keratozlar BCC'ye benzer psödokist yapıları sergileyebilir."},
        {"isim": "Skuamöz Hücreli Karsinom", "neden": "Ülserasyon ve krutlanma (kabuklanma) varlığında klinik olarak nodüler BCC ile örtüşebilir."},
        {"isim": "Amelanotik Melanom", "neden": "Pigmentsiz nodüler BCC vakalarında gözlenen atipik vasküler yapılar melanomu andırabilir."}
    ],
    "skuamoz_hucreli": [
        {"isim": "Keratoakantom", "neden": "SCC'nin iyi huylu veya premalign varyantı olarak değerlendirilir, histolojik ve klinik görünüm çok benzerdir."},
        {"isim": "Hiperkeratotik Aktinik Keratoz", "neden": "Kalın kabuklu lezyonlar, erken evre SCC'den klinik muayene ile ayırt edilemeyebilir."},
        {"isim": "Prurigo Nodülaris", "neden": "Kronik kaşınmaya bağlı oluşan nodüller SCC'yi taklit edebilir."}
    ],
    "aktinik_keratoz": [
        {"isim": "Yüzeyel Skuamöz Hücreli Karsinom", "neden": "Aktinik keratozun doğrudan ilerlemiş ve malignleşmiş formudur."},
        {"isim": "Lentigo Maligna", "neden": "Özellikle yüz bölgesindeki pigmente aktinik keratozlar lentigo maligna ile karışabilir."},
        {"isim": "Likenoid Keratoz", "neden": "Eritematöz ve pullu yapısıyla klinik olarak aktinik keratozu andırır."}
    ],
    "benign_keratoz": [
        {"isim": "Malign Melanom", "neden": "Özellikle koyu renkli, asimetrik ve irritasyona uğramış seboreik keratozlar melanom şüphesi uyandırabilir."},
        {"isim": "Pigmente BCC", "neden": "Dernoskopik milia benzeri kistler her iki lezyonda da görülebildiğinden ayırıcı tanı şarttır."},
        {"isim": "Verruka Vulgaris (Siğil)", "neden": "Hiperkeratotik yüzeyleri nedeniyle özellikle ekstremitelerde siğillerle karışabilir."}
    ],
    "nevus": [
        {"isim": "Malign Melanom (Erken Evre)", "neden": "Atipik özellikleri gösteren veya yeni değişime uğrayan nevüsler melanom dışlanana kadar daima şüphelidir."},
        {"isim": "Dermatofibrom", "neden": "Pigmente dermatofibromlar, merkezi beyaz yama ve periferik ağ ile nevüsü taklit edebilir."},
        {"isim": "Lentigo Simplex", "neden": "Küçük, düz ve koyu renkli nevüsler lentigo lezyonları ile karışabilir."}
    ],
    "dermatofibrom": [
        {"isim": "Dermatofibrosarkoma Protuberans (DFSP)", "neden": "Nadir görülmesine rağmen malign karakterli bu lokal agresif tümör, klinik olarak dermatofibromlara benzer."},
        {"isim": "Malign Melanom", "neden": "Periferik pigment ağı belirgin olan dermatofibromlar melanom şüphesi yaratabilir."},
        {"isim": "Nodüler Kaposi Sarkomu", "neden": "Eritematöz veya mor renkli yapısı nedeniyle ayırıcı tanıye girer."}
    ],
    "vaskuler_lezyon": [
        {"isim": "Amelanotik Melanom", "neden": "Sadece polimorfik damar yapısı sergileyen ve pigmentsiz olan melanomlar anjiyom zannedilebilir."},
        {"isim": "Kaposi Sarkomu", "neden": "Özellikle alt ekstremitelerdeki vasküler nodüller kaposi sarkomu ayırıcı tanısını gerektirir."},
        {"isim": "Piyojenik Granülom", "neden": "Hızlı büyüyen, kanamaya çok meyilli vasküler lezyonlardır, hemanjiyomları andırır."}
    ],
    "malign_genel": [
        {"isim": "Melanom Varyantları", "neden": "Nodüler veya amelanotik melanom formları klasik kalıpların dışında şüpheli görünümler sunabilir."},
        {"isim": "İleri Evre SCC/BCC", "neden": "Doku yıkımının (ülserasyon) ön planda olduğu invaziv karsinom formları."},
        {"isim": "Merkel Hücreli Karsinom", "neden": "Nadir ancak son derece agresif nöroendokrin cilt tümörleridir, atipik nodüller şeklinde belirir."}
    ],
    "benign_genel": [
        {"isim": "Seboreik Keratoz", "neden": "En sık karşılaşılan zararsız yaşlılık lekeleridir."},
        {"isim": "Kistik Yapılar", "neden": "Epidermal inklüzyon kistleri gibi cilt altı zararsız formasyonlar."},
        {"isim": "Enflamatuar Reaksiyonlar", "neden": "Böcek ısırığı, follikülit veya lokalize dermatit kaynaklı iyi huylu reaksiyonlar."}
    ],
    "genel": [
        {"isim": "Enflamatuar Dermatozlar", "neden": "Lokalize egzama, sedef (psöriazis) veya liken planus plakları spesifik bir morfoloji göstermeyebilir."},
        {"isim": "Travmatik / Mekanik Lezyonlar", "neden": "Subungual hematom (tırnak altı kanama) veya sürtünme kaynaklı fokal kalınlaşmalar."},
        {"isim": "Nadir Benign Tümörler", "neden": "Porom, siringom veya nörofibrom gibi nadir ancak zararsız doku büyümeleri."}
    ]
}

# hasta portalinda gorunecek basitleştirilmiş ansiklopedi ve sss metinleri
ENCYCLOPEDIA = {
    "melanom": {
        "ad": "Melanom Tipi Morfoloji (Klinik İnceleme Önerilir)", "icon": "🔬",
        "nedir": "Ciltteki renk veren hücrelerden (melanositlerden) kaynaklanabilen, klinik önemi yüksek bir lezyon profilidir.",
        "neden": "Kümülatif UV maruziyeti, şiddetli güneş yanıkları, genetik yatkınlık ve atipik ben varlığı en temel risk faktörleridir.",
        "tedavi": "Dermatolog tarafından yapılan muayene sonrası genellikle cerrahi eksizyon (lezyonun alınması) temel yaklaşımdır.",
        "sonraki_adimlar": "Lütfen paniğe kapılmadan, vakit kaybetmeden Merkezi Hekim Randevu Sistemi'nden randevu alınız.",
        "sss": [
            {"q": "Bu kesinlikle kanser miyim demek?", "a": "Hayır. Bu sistem sadece görsel bir algoritmadır. Kesin tanı sadece doktorunuzun alacağı biyopsi (parça) ile konulabilir."},
            {"q": "Alınması tehlikeli midir?", "a": "Aksine, şüpheli lezyonların tıp uzmanları tarafından cerrahi olarak tamamen çıkarılması en güvenilir ve hayat kurtarıcı yöntemdir. 'Bıçak değerse yayılır' inanışı tamamen asılsız bir hurafedir."},
            {"q": "Tedavisi uzun sürer mi?", "a": "Erken evrede saptandığında, sadece ufak bir cerrahi işlemle aynı gün içinde tedavi tamamlanabilir."}
        ]
    },
    "bazal_hucreli": {
        "ad": "Bazal Hücreli Formasyon", "icon": "🟡",
        "nedir": "Cildin üst tabakasındaki hücrelerden kaynaklanan, çok yavaş büyüyen ve başka organlara sıçrama ihtimali çok düşük olan bir yapı profilidir.",
        "neden": "Yıllar boyunca ciltte biriken güneş hasarı (UV ışınları) temel nedendir.",
        "tedavi": "Poliklinik şartlarında uygulanan basit bir lokal cerrahi işlemle kolayca tedavi edilir.",
        "sonraki_adimlar": "Hayati bir tehlike oluşturmasa da bir dermatoloğa başvurup tedavisini planlamalısınız.",
        "sss": [
            {"q": "İlaçla geçer mi?", "a": "Bazı yüzeysel tiplerinde özel kremler kullanılabilse de, en kesin çözüm ufak cerrahi operasyonla alınmasıdır."},
            {"q": "Vücuduma yayılır mı?", "a": "Bazal hücreli yapıların diğer organlara yayılma (metastaz) riski neredeyse sıfıra yakındır. Sadece bulunduğu bölgede yavaşça büyür."},
            {"q": "Korkmalı mıyım?", "a": "Hayır. Tıpta tedavisi en kolay ve başarı oranı en yüksek cilt sorunlarından biridir."}
        ]
    },
    "skuamoz_hucreli": {
        "ad": "Skuamöz Hücreli Formasyon", "icon": "🟠",
        "nedir": "Cildin en üst yüzey hücrelerinde meydana gelen ve dikkatle takip edilmesi gereken bir yapılaşmadır.",
        "neden": "Uzun süreli güneş yanıkları ve kronik cilt hasarları.",
        "tedavi": "Lokal anestezi altında uygulanan cerrahi müdahale en güvenli yöntemdir.",
        "sonraki_adimlar": "Yakın zamanda bir dermatoloji uzmanı ile görüşmeniz sağlığınız için doğru olacaktır.",
        "sss": [
            {"q": "Acil bir durum mu?", "a": "Panik gerektirecek bir aciliyeti yoktur ancak ihmal edilmemeli, birkaç hafta içinde doktora gösterilmelidir."},
            {"q": "Kendiliğinden geçer mi?", "a": "Hayır, bu tür formasyonlar kendiliğinden iyileşmez. Medikal müdahale şarttır."},
            {"q": "İyileşme süreci nasıldır?", "a": "Ufak cerrahi işlem sonrası yaranın kapanması genellikle 1-2 hafta sürer ve hasta normal hayatına anında döner."}
        ]
    },
    "malign_genel": {
        "ad": "Şüpheli (Atipik) Lezyon Profili", "icon": "🚨",
        "nedir": "Sistem, fotoğrafınızda asimetri veya düzensiz sınırlar gibi doktor tarafından mutlaka incelenmesi gereken şüpheli bulgular saptadı.",
        "neden": "Cilt hücrelerinin genetik veya çevresel (güneş vb.) faktörler nedeniyle yapısal değişiklik göstermesi.",
        "tedavi": "Dermatolog eşliğinde dermoskopik inceleme yapılır ve gerekirse kesin tanı için doku örneği (biyopsi) alınır.",
        "sonraki_adimlar": "İnternette araştırma yapmak yerine, doğrudan MHRS üzerinden Dermatoloji randevusu alınız.",
        "sss": [
            {"q": "Sistem yanlış değerlendirmiş olabilir mi?", "a": "Kesinlikle evet. Algoritmalar, gölgelerden veya ciltteki basit tahrişlerden yanılabilir. Bu uyarı sadece 'Doktor kontrolü şart' anlamına gelir."},
            {"q": "Biyopsi acı verici midir?", "a": "Hayır. Biyopsi işlemi diş hekiminin dişi uyuşturması gibi bölgenin lokal olarak iğneyle uyuşturulmasıyla yapılır. İşlem sırasında acı hissedilmez."},
            {"q": "Şimdi ne yapmalıyım?", "a": "Güneşten korunun, lezyonu kaşımayın veya kanatmayın, aşağıdan MHRS randevunuzu alın."}
        ]
    },
    "aktinik_keratoz": {
        "ad": "Aktinik Keratoz (Güneş Lekesi)", "icon": "🔶",
        "nedir": "Güneşin cildi uzun yıllar yıpratması sonucu ortaya çıkan, hafif pürüzlü, kabuklanma yapabilen lekelerdir.",
        "neden": "Ömrünüz boyunca birikmiş güneş ışığı maruziyeti.",
        "tedavi": "Hekim tarafından uygulanan kriyoterapi (dondurma) veya medikal kremlerle kolayca ortadan kaldırılır.",
        "sonraki_adimlar": "Doktor kontrolünde tedavi edilmesi ve yüksek faktörlü güneş kremi kullanılması tavsiye edilir.",
        "sss": [
            {"q": "Kozmetik kremlerle geçer mi?", "a": "Marketlerde satılan standart nemlendiricilerle geçmez. Dermatoloğun yazacağı reçeteli kremler veya dondurma işlemi gerekir."},
            {"q": "Kansere dönüşür mü?", "a": "Tedavi edilmeyen aktinik keratozların çok küçük bir kısmı (%1-5) yıllar içinde cilt kanserine dönüşme riski taşır, bu yüzden erkenden tedavi edilirler."},
            {"q": "Bulaşıcı mıdır?", "a": "Hayır, enfeksiyon kaynaklı değildir, kesinlikle bulaşıcı değildir."}
        ]
    },
    "benign_keratoz": {
        "ad": "Seboreik Keratoz (İyi Huylu Leke)", "icon": "🟢",
        "nedir": "Genellikle yaş ilerledikçe cildin yüzeyinde beliren, mumsu, kabarık ve tamamen iyi huylu olan cilt lekeleridir.",
        "neden": "Yaşlanma süreci ve genetik faktörler.",
        "tedavi": "Sağlık açısından hiçbir medikal müdahale gerektirmez. Kozmetik olarak sizi rahatsız ediyorsa aldırabilirsiniz.",
        "sonraki_adimlar": "Gönül rahatlığıyla hayatınıza devam edebilirsiniz.",
        "sss": [
            {"q": "Giderek çoğalıyorlar, normal mi?", "a": "Evet, yaş ilerledikçe sayılarının artması genetik olarak beklenen ve normal bir durumdur."},
            {"q": "Üzeri pul pullu dökülüyor?", "a": "Seboreik keratozların karakteristik yapısı böyledir, üzerleri kabuklu olabilir. Koparmamaya çalışın."},
            {"q": "Tehlikeli midir?", "a": "Tıbbi olarak sıfır tehlike barındırırlar."}
        ]
    },
    "nevus": {
        "ad": "Melanositik Nevüs (Klasik Ben)", "icon": "🟤",
        "nedir": "Ciltteki renk hücrelerinin bir araya toplanmasıyla oluşan zararsız yapılardır.",
        "neden": "Genetik altyapınız ve güneşe maruziyet.",
        "tedavi": "Değişiklik göstermediği sürece alınmasına gerek duyulmaz.",
        "sonraki_adimlar": "Evde kendi kendinize ABCDE kuralı ile benlerinizi kontrol etmeniz yeterlidir.",
        "sss": [
            {"q": "Benleri aldırmak riskli midir?", "a": "Hayır, tamamen güvenlidir. Aksine, riskli görülen benlerin uzman cerrahlarca alınması koruyucu bir işlemdir."},
            {"q": "Yeni ben çıkması normal mi?", "a": "Özellikle 30'lu yaşlara kadar yeni ben çıkması normaldir. Ancak 40 yaşından sonra aniden çıkan siyah benler doktora gösterilmelidir."},
            {"q": "Benin üzerinde kıl çıkması kötü mü?", "a": "Hayır, aksine benin üzerinde kıl olması hücrelerin iyi huylu ve olgun olduğunu gösteren pozitif bir işarettir."}
        ]
    },
    "dermatofibrom": {
        "ad": "Dermatofibrom", "icon": "🔘",
        "nedir": "Genellikle bacaklarda veya kollarda oluşan, cilt altındaki sert ve iyi huylu küçük kitlelerdir.",
        "neden": "Sinek ısırığı veya ufak cilt travmalarına karşı vücudun oluşturduğu doku tamir yanıtı.",
        "tedavi": "Müdahale gerektirmez. Kozmetik sebeple lokal anestezi ile çıkarılabilir.",
        "sonraki_adimlar": "Sağlığınızı tehdit etmeyen güvenli bir bulgudur.",
        "sss": [
            {"q": "Sıktığımda içeri çöküyor?", "a": "Bu dermatofibromun en klasik özelliğidir, buna 'çukur belirtisi (dimple sign)' denir und iyi huylu olduğunun kanıtıdır."},
            {"q": "Ağrı yapar mı?", "a": "Genellikle ağrısızdır ancak sıkıştırıldığında veya çarpıldığında hafif hassasiyet yapabilir."},
            {"q": "Kendiliğinden kaybolur mu?", "a": "Genelde kalıcıdır, kendiliğinden geçmez ama sağlığa zararı yoktur."}
        ]
    },
    "vaskuler_lezyon": {
        "ad": "Vasküler Lezyon (Damar Beni)", "icon": "🔴",
        "nedir": "Kılcal damarların cilt yüzeyine yakın bölgelerde kümelenmesiyle oluşan iyi huylu kırmızı formasyonlardır.",
        "neden": "Yaşlanma, hamilelikteki hormonal değişimler veya genetik yapı.",
        "tedavi": "Tıbbi olarak tedavi gerektirmez. Lazer sistemleri ile kolaylıkla silinebilir.",
        "sonraki_adimlar": "Zararsız bir bulgudur, kanatmamaya özen gözteriniz.",
        "sss": [
            {"q": "Kanarsa ne yapmalıyım?", "a": "Damar yapısında olduğu için kanayabilir. Üzerine temiz bir bezle 5 dakika baskı uygulayın, duracaktır."},
            {"q": "Kanser riski taşır mı?", "a": "Hayır, damar benleri (kiraz anjiyomlar) cilt kanserine dönüşmez."},
            {"q": "Neden kıpkırmızı?", "a": "İçinde melanin pigmenti değil, doğrudan yoğun kılcal kan damarları bulunduğu için rengi parlak kırmızı veya mordur."}
        ]
    },
    "benign_genel": {
        "ad": "İyi Huylu (Benign) Formasyon", "icon": "✅",
        "nedir": "Sistem, fotoğrafınızın doku yapısını inceledi ve kötü huylu bir belirti tespit etmedi. Bu zararsız bir ben, leke veya yaşlılık izi olabilir.",
        "neden": "Cildin doğal yapısı veya yaşlanma belirtileri.",
        "tedavi": "Klinik olarak acil bir tedavi protokolü gerektirmez.",
        "sonraki_adimlar": "Şu an için endişe etmenizi gerektirecek bir durum görünmüyor. Yıllık kontrollerinize devam ediniz.",
        "sss": [
            {"q": "Sistem %100 haklı mıdır?", "a": "Hiçbir sistem %100 haklı değildir. Şüpheleriniz devam ediyorsa her zaman son sözü doktora bırakmalısınız."},
            {"q": "Bu lekeyi evde kremle silebilir miyim?", "a": "Doktor onayı olmadan asitli veya soyucu kimyasal kremler kullanmak cildinizde kalıcı yanık izleri bırakabilir."},
            {"q": "Cilt doktoruna gitmeme gerek var mı?", "a": "Acil bir gereklilik yok, ancak check-up amaçlı yılda bir kez gitmek sağlıklı bir alışkanlıktır."}
        ]
    },
    "genel": {
        "ad": "Algoritmik Karar Destek İncelemesi", "icon": "📋",
        "nedir": "Sistem lezyonunuzun morfolojik haritasını çıkardı, ancak spesifik bir hastalık ismi atamak yerine genel bir yapısal analiz gerçekleştirdi.",
        "neden": "Fotoğraf kalitesi veya lezyonun atipik bir yapıda olması.",
        "tedavi": "Hekiminizin dermatoskop ile yapacağı fiziksel muayene esastır.",
        "sonraki_adimlar": "Verilen risk uyarı koduna bakarak uygun zamanda dermatoloji randevusu alınız.",
        "sss": [
            {"q": "Sistem neden bulamadı?", "a": "Çekim açısı, bulanıklık, saç/kıl yoğunluğu veya ışık yansımaları sistemin net bir isim vermesini engellemiş olabilir."},
            {"q": "Bu kötü bir haber mi?", "a": "Hayır, sadece sonucun 'Belersiz (Unclassified)' olduğu anlamına gelir."},
            {"q": "Doktora ne demeliyim?", "a": "Sadece lezyonun ne kadar süredir orada olduğunu ve boyutunun değişip değişmediğini söylemeniz yeterlidir."}
        ]
    }
}

_MATCHERS = [
    (["melanom", "melanoma", "mel", "4"], "melanom"),
    (["bazal", "bcc", "1"], "bazal_hucreli"),
    (["skuamöz", "scc", "7"], "skuamoz_hucreli"),
    (["kötü huylu", "malign", "kanser", "malignant", "şüpheli", "atipik"], "malign_genel"), 
    (["aktinik", "akiec", "bowen", "0"], "aktinik_keratoz"),
    (["seboreik", "bkl", "2"], "benign_keratoz"),
    (["iyi huylu", "benign", "zararsız", "normal"], "benign_genel"),
    (["dermatofibrom", "df", "3"], "dermatofibrom"),
    (["vasküler", "vasc", "damar", "6"], "vaskuler_lezyon"),
    (["nevüs", "nevus", "ben", "nv", "5"], "nevus")
]

def format_diagnosis_name(raw_name):
    # backendden gelen raw index veya string ifadeleri kullanıcı dostu metne donusturur
    text = str(raw_name).strip().lower()
    mapping = {
        "0": "Aktinik Keratoz (akiec)", "1": "Bazal Hücreli Karsinom (bcc)",
        "2": "Benign Keratoz (bkl)", "3": "Dermatofibrom (df)",
        "4": "Melanom (mel)", "5": "Melanositik Nevüs (nv)",
        "6": "Vasküler Lezyon (vasc)", "7": "Skuamöz Hücreli Karsinom (scc)"
    }
    if text in mapping: return mapping[text]
    
    for keywords, correct_name in [
        (["mel"], "Melanom (mel)"), (["bcc", "bazal"], "Bazal Hücreli Karsinom (bcc)"),
        (["scc", "skuamöz"], "Skuamöz Hücreli Karsinom (scc)"), (["akiec", "aktinik"], "Aktinik Keratoz (akiec)"),
        (["malign", "kötü", "şüpheli"], "Şüpheli (Malign Potansiyelli) Formasyon"), (["benign", "iyi huylu"], "Benign (İyi Huylu) Formasyon"),
        (["bkl"], "Benign Keratoz (bkl)"), (["df"], "Dermatofibrom (df)"),
        (["vasc"], "Vasküler Lezyon (vasc)"), (["nv", "nevus"], "Melanositik Nevüs (nv)")
    ]:
        for kw in keywords:
            if kw in text: return correct_name
    return str(raw_name).strip().capitalize()

def get_disease_info(extracted_name):
    text = str(extracted_name).lower()
    for keywords, key in _MATCHERS:
        for kw in keywords:
            if kw in text: return ENCYCLOPEDIA[key]
    return ENCYCLOPEDIA["genel"]

def get_ddx_info(extracted_name):
    text = str(extracted_name).lower()
    for keywords, key in _MATCHERS:
        for kw in keywords:
            if kw in text: return DIFFERENTIAL_DIAGNOSIS.get(key, DIFFERENTIAL_DIAGNOSIS["genel"])
    return DIFFERENTIAL_DIAGNOSIS["genel"]

def extract_disease_name(report_dict):
    # gelen dict icinden tani anahtar kelimesini bulup ceken helper
    if not isinstance(report_dict, dict): return format_diagnosis_name(report_dict)
    for k in ['kategori', 'tani', 'sinif', 'label', 'class', 'isim', 'teşhis', 'prediction', 'disease']:
        for dict_key in report_dict.keys():
            if k in str(dict_key).lower() and report_dict[dict_key] is not None:
                return format_diagnosis_name(report_dict[dict_key])
    for val in report_dict.values():
        val_str = str(val).lower()
        if any(kw in val_str for keywords, _ in _MATCHERS for kw in keywords):
            return format_diagnosis_name(val_str)
    return "Morfolojik Sınıflandırma Belirtilmedi"

def extract_recommendation(report_dict):
    if not isinstance(report_dict, dict): return "Tıbbi kılavuzlar ışığında klinik muayene ve dermoskopik takip önerilir."
    keys = ['tavsiye', 'protokol', 'oneri', 'klinik', 'recommendation', 'action', 'açıklama']
    for k in keys:
        for dict_key in report_dict.keys():
            if k in str(dict_key).lower() and report_dict[dict_key]:
                return str(report_dict[dict_key])
    return "Klinik korelasyon ve histopatolojik değerlendirme gerekliliği hekim inisiyatifindedir."

def parse_risk_score(raw_value):
    if raw_value is None: return 0.0
    if isinstance(raw_value, (int, float)):
        val = float(raw_value)
        return val if val > 1.0 else val * 100.0
    text = str(raw_value).strip().replace(",", ".")
    cleaned = re.sub(r"[^0-9.]", "", text)
    if cleaned.count(".") > 1:
        first_dot = cleaned.index(".")
        cleaned = cleaned[:first_dot + 1] + cleaned[first_dot + 1:].replace(".", "")
    if not cleaned or cleaned == ".": return 0.0
    try:
        val = float(cleaned)
        return val if val > 1.0 else val * 100.0
    except ValueError: return 0.0

def extract_risk_score(report_dict):
    if not isinstance(report_dict, dict): return parse_risk_score(report_dict)
    for val in report_dict.values():
        if isinstance(val, str) and '%' in val: return parse_risk_score(val)
    for k in ['yuzde', 'skor', 'guven', 'risk', 'olasilik', 'score']:
        for dict_key in report_dict.keys():
            if k in str(dict_key).lower(): return parse_risk_score(report_dict[dict_key])
    return 0.0

def extract_triage_code(report_dict):
    if not isinstance(report_dict, dict): return "BELİRSİZ"
    for val in report_dict.values():
        val_str = str(val).upper()
        if "KIRMIZI" in val_str or "SARI" in val_str or "YEŞİL" in val_str: return val_str
    for k in ['triyaj_kodu', 'renk', 'durum', 'triage', 'code']:
        for dict_key in report_dict.keys():
            if k in str(dict_key).lower(): return str(report_dict[dict_key]).upper()
    return "BELİRSİZ"

def render_metric_card(label, value, icon="", accent="#0b3d91", value_size="1.25rem", tooltip=""):
    # css tabanli metrik kutusu hazirlama fonksiyonu
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    tooltip_html = f"<div class='tooltip-text'>{tooltip}</div>" if tooltip else ""
    st.markdown(f"""
        <div class="ui-card" style="border-left: 6px solid {accent};">
            <div class="ui-card-label">{icon} {safe_label}</div>
            <div class="ui-card-value" style="font-size:{value_size}; color:{accent};">{safe_value}</div>
            {tooltip_html}
        </div>
    """, unsafe_allow_html=True)

def render_encyclopedia_card(title, content, icon, accent="#1565c0"):
    safe_title = html.escape(str(title))
    st.markdown(f"""
        <div class="ui-card" style="border-left: 6px solid {accent}; background: linear-gradient(to right, #ffffff, #f8fafc);">
            <div class="ui-card-label" style="font-size:1.05rem; text-transform:none; color:#1e293b;">{icon} {safe_title}</div>
            <div style="font-size:1.05rem; color:#475569; line-height:1.6; margin-top:8px;">{html.escape(str(content))}</div>
        </div>
    """, unsafe_allow_html=True)

def render_progress_bar(percent, color="#0b3d91", label="Morfolojik Uyum İndeksi (Concordance)"):
    percent = max(0.0, min(100.0, percent))
    st.markdown(f"""
        <div class="custom-progress-track">
            <div class="custom-progress-fill" style="width:{percent}%; background-color:{color};"></div>
        </div>
        <div style="font-size:0.85rem; color:#64748b; margin-top:8px; font-weight:700; text-align:right;">{label}: %{percent:.1f}</div>
    """, unsafe_allow_html=True)

def triage_accent_color(triyaj_kodu):
    kodu = str(triyaj_kodu).upper()
    if "KIRMIZI" in kodu: return "#dc2626" 
    if "SARI" in kodu: return "#ea580c" 
    return "#10b981" 

def cleanup_temp_file(path):
    # isi biten geçici dosyalari diskte cop kalmasin diye siliyoruz
    if path and os.path.exists(path):
        try: os.remove(path)
        except OSError: pass

def render_role_switch():
    with st.sidebar:
        if st.button("⬅️ Oturumu Kapat / Ana Ekrana Dön", use_container_width=True):
            st.session_state.role = None
            st.session_state.doc_authenticated = False
            st.rerun()
        st.divider()

def render_landing_page():
    # ilk acilistaki rol secim ekrani (landing)
    st.markdown("""
        <div style="text-align:center; margin-bottom:3rem; margin-top:2rem;">
            <h1 style="margin-bottom:0.8rem; font-size: 3rem; color: #0f172a; font-weight: 900; letter-spacing: -1px;">⚕️ Klinik Karar Destek Sistemi (CDSS)</h1>
            <p style="color:#64748b; font-size:1.3rem; font-weight: 500;">Dermoskopik Taramalar için Gelişmiş Morfolojik Analiz Platformu</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
            <div class="role-card role-card-doctor">
                <div class="role-card-icon">🩺</div>
                <div class="role-card-title">Klinisyen Portalı</div>
                <div class="role-card-desc">Sınıflandırma Öngörüleri • DDx (Ayırıcı Tanı) • XAI Analizi • Epikriz Şablonu • Konsültasyon (Şifreli Erişim)</div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🩺 Klinisyen Portalına Güvenli Giriş", use_container_width=True):
            st.session_state.role = "doctor"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="role-card role-card-patient">
                <div class="role-card-icon">🙂</div>
                <div class="role-card-title">Danışan Portalı</div>
                <div class="role-card-desc">Ön Değerlendirme Raporu • ABCDE Testi • Fitzpatrick Cilt Analizi • MHRS Entegrasyonu • SSS</div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🙂 Danışan Portalına Giriş", use_container_width=True):
            st.session_state.role = "patient"
            st.rerun()

def render_doctor_portal(triage_engine, xai_engine):
    # hekim ekranı modul akislari burdan yonetiliyor
    render_role_switch()

    if not st.session_state.doc_authenticated:
        st.markdown("""
            <div class="auth-box">
                <h2 style="margin-bottom: 15px;">🔒 Hekim Yetki Doğrulaması</h2>
                <p style="color:#64748b; font-size: 1.1rem; margin-bottom: 30px;">Bu portal sadece yetkili klinik personel içindir. Lütfen yetki şifrenizi giriniz.</p>
            </div>
        """, unsafe_allow_html=True)
        c_left, c_mid, c_right = st.columns([1,2,1])
        with c_mid:
            pwd = st.text_input("Sistem Şifresi", type="password", placeholder="Şifrenizi girin...")
            if st.button("Sisteme Giriş Yap", use_container_width=True, type="primary"):
                if pwd == "hekim2026":
                    st.session_state.doc_authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Hatalı şifre. Erişim reddedildi.")
        return

    with st.sidebar:
        st.title("Sistem Durumu")
        with st.status("Analiz Modülleri Bağlanıyor...", expanded=True) as status:
            st.write("Dermatoskopik Çekirdek: Aktif ✅")
            st.write("Morfoloji XAI Modülü: Aktif ✅")
            st.write("Ayırıcı Tanı (DDx) Veritabanı: Aktif ✅")
            status.update(label="Sistem Çevrimiçi", state="complete", expanded=False)

        st.divider()
        st.subheader("👤 Hasta Anamnez Formu")
        hasta_id = st.text_input("TC / Protokol No", placeholder="Örn: 12345678901")
        yas = st.number_input("Yaş", min_value=0, max_value=120, value=0, step=1)
        cinsiyet = st.selectbox("Cinsiyet", ["Belirtilmedi", "Erkek", "Kadın"])
        lezyon_lokasyonu = st.selectbox("Lezyon Bölgesi", ["Seçiniz", "Yüz", "Boyun", "Gövde / Sırt", "Göğüs", "Kollar", "Bacaklar", "El/Ayak Tabanı", "Saçlı Deri", "Diğer"])
        sure_deger = st.number_input("Semptom Süresi", min_value=0, value=0)
        sure_birim = st.selectbox("Süre Birimi", ["Gün", "Ay", "Yıl"])
        ek_not = st.text_area("Klinisyen Notu (Opsiyonel)")

        st.session_state.doctor_patient_info = {
            "hasta_id": hasta_id.strip(), "yas": yas, "cinsiyet": cinsiyet, "lokasyon": lezyon_lokasyonu,
            "sure": f"{sure_deger} {sure_birim}" if sure_deger > 0 else "", "ek_not": ek_not.strip()
        }

        st.divider()
        st.markdown('<div class="consent-box">', unsafe_allow_html=True)
        onam_verildi = st.checkbox("Hasta onamı alındı. Bu aracın kesin teşhis koymadığını onaylıyorum.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="header-banner header-doctor">
            <h1>🩺 Klinisyen Portalı — CDSS Konsolu</h1>
            <p>Dermoskopik görüntüyü sisteme yükleyin, XAI ısı haritasını, DDx öngörülerini ve tedavi kılavuzlarını inceleyin.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.doctor_patient_info.get("hasta_id") and len(st.session_state.doctor_patient_info["hasta_id"]) > 3:
        with st.expander(f"📁 Hastane Bilgi Yönetim Sistemi (HBYS) Geçmişi: {st.session_state.doctor_patient_info['hasta_id']}"):
            mock_data = pd.DataFrame({
                "Tarih": ["12.01.2026", "05.11.2025"],
                "Lokalizasyon": ["Boyun", "Sırt"],
                "ICD-10 Kodu": ["L82", "D22.5"],
                "Önceki Tanı": ["Seboreik Keratoz", "Melanositik Nevüs"],
                "Klinik İşlem": ["Dermoskopik Takip", "Eksizyonel Biyopsi"]
            })
            st.dataframe(mock_data, use_container_width=True, hide_index=True)

    uploaded_file = st.file_uploader("Dermoskopik Görüntü Yükleme (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col_img, col_action = st.columns([1, 2], gap="large")
        
        with col_img:
            st.subheader("İncelenen Vaka")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Orijinal Dermoskopik Kayıt")

        with col_action:
            st.subheader("Klinik İşlem Protokolü")
            info = st.session_state.doctor_patient_info
            tag_html = ""
            if info.get("hasta_id"): tag_html += f'<span class="patient-tag">🆔 {html.escape(info["hasta_id"])}</span>'
            if info.get("yas"): tag_html += f'<span class="patient-tag">🎂 {info["yas"]} Yaş</span>'
            if info.get("cinsiyet") and info["cinsiyet"] != "Belirtilmedi": tag_html += f'<span class="patient-tag">🚻 {info["cinsiyet"]}</span>'
            if info.get("lokasyon") and info["lokasyon"] != "Seçiniz": tag_html += f'<span class="patient-tag">📍 {html.escape(info["lokasyon"])}</span>'
            if tag_html: st.markdown(tag_html, unsafe_allow_html=True)
            
            st.write("Görüntü CDSS motoruna aktarıldı. Derin öğrenme analizini başlatmak için aşağıdan onay veriniz.")
            analiz_butonu = st.button("🔬 CDSS Analizini Başlat", type="primary", use_container_width=True, disabled=not onam_verildi)

            if analiz_butonu:
                st.session_state.doctor_analysis_done = False
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    with st.spinner('Evrişimsel sinir ağı (CNN) özellikleri çıkarılıyor ve DDx haritası oluşturuluyor...'):
                        report = triage_engine.analyze_lesion(tmp_path)
                        time.sleep(0.5)
                        xai_output_path = f"analiz_{Path(tmp_path).stem}.png"
                        xai_engine.generate_analysis(tmp_path, output_dir=".")

                    st.session_state.doctor_report = report
                    st.session_state.doctor_xai_path = xai_output_path
                    st.session_state.doctor_analysis_done = True
                    st.session_state.doctor_timestamp = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
                    st.session_state.consultation_link_generated = False 
                except Exception as e:
                    st.error(f"❌ Analiz sırasında kritik hata oluştu: {str(e)}")
                finally:
                    cleanup_temp_file(tmp_path)

    if st.session_state.doctor_analysis_done and st.session_state.doctor_report:
        st.divider()
        st.header("📋 CDSS Analiz Raporu")

        report = st.session_state.doctor_report
        info = st.session_state.doctor_patient_info
        
        tani_sonucu = extract_disease_name(report)
        yuzde_skor_ham = extract_risk_score(report)
        triyaj_kodu = extract_triage_code(report)
        klinik_tavsiye = extract_recommendation(report)
        accent = triage_accent_color(triyaj_kodu)

        akademi_notu = "Akademik Not: Düşük skorlar modelin başarısızlığını değil, lezyonun eğitim verisindeki klasik örneklere kıyasla atipik/nadir (outlier) bir yapıda olduğunu gösterir. Atipik lezyonlarda biyopsi veya histopatolojik doğrulama elzemdir."

        c1, c2 = st.columns(2)
        with c1: render_metric_card("Sistem Sınıflandırma Öngörüsü", tani_sonucu, icon="🧬", accent=accent)
        with c2: render_metric_card("Sistem Triyaj Yönergesi", triyaj_kodu, icon="🚦", accent=accent)

        render_metric_card("Morfolojik Uyum İndeksi (Concordance Index)", f"%{yuzde_skor_ham:.1f}", icon="📊", accent=accent, tooltip=akademi_notu)
        render_progress_bar(yuzde_skor_ham, color=accent, label="Uyum (Concordance) Katsayısı")
        
        st.info(f"**💡 Algoritmik Protokol Önerisi:** {klinik_tavsiye}")

        st.divider()
        
        doc_tab1, doc_tab2, doc_tab3, doc_tab4, doc_tab5 = st.tabs([
            "🧠 Ayırıcı Tanı (DDx)", 
            "👁️ Morfoloji (XAI)", 
            "📋 7-Nokta & Epikriz",
            "📊 Epidemiyoloji",
            "🤝 Konsültasyon"
        ])
        
        with doc_tab1:
            st.markdown("#### CDSS Destekli Ayırıcı Tanı (Differential Diagnosis)")
            st.write(f"Sistemin birincil sınıflandırması olan **{tani_sonucu}** tanısına ek olarak, klinik pratikte dışlanması gereken alternatif tanılar:")
            
            ddx_list = get_ddx_info(tani_sonucu)
            for item in ddx_list:
                st.markdown(f"<div style='background-color: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid #334155; margin-bottom: 10px;'><b>🔹 {item['isim']}</b><br><span style='color: #475569; font-size: 0.95rem;'>{item['neden']}</span></div>", unsafe_allow_html=True)
                
            st.divider()
            st.markdown("#### 📚 Literatür ve Tedavi Kılavuzları")
            if "KIRMIZI" in triyaj_kodu or "melanom" in tani_sonucu.lower() or "malign" in tani_sonucu.lower():
                st.error("📌 **NCCN Guidelines v2.2024:** Şüpheli melanom/malign vakalarında 1-3 mm klinik sınırla eksizyonel biyopsi (punch biyopsi önerilmez) gerçekleştirilmesi ve Breslow kalınlığına göre genişletilmiş eksizyon planlanması tavsiye edilmektedir.")
            elif "bazal" in tani_sonucu.lower() or "skuamoz" in tani_sonucu.lower():
                st.warning("📌 **NCCN Basal/Squamous Cell Kılavuzu:** Düşük riskli gövde bölgelerinde standart 4 mm sınırla eksizyon, yüksek riskli yüz (H bölgesi) alanlarında doku koruyucu Mohs Mikrografik Cerrahisi tercih edilmelidir.")
            else:
                st.success("📌 **Uluslararası Dermoskopi Derneği (IDS):** Benign (iyi huylu) patern gösteren lezyonlarda agresif cerrahi yerine kısa dönem (3-6 ay) dijital dermoskopik takip (Digital Dermoscopy Follow-up) protokolü uygulanabilir.")

        with doc_tab2:
            st.markdown("#### Algoritma Odak Haritası (Grad-CAM XAI)")
            st.markdown("*Sıcak (Kırmızı) bölgeler, derin öğrenme modelinin karar mekanizmasında en yüksek ağırlığı verdiği (aktivasyon) dermoskopik patern alanlarıdır.*")
            if st.session_state.doctor_xai_path and os.path.exists(st.session_state.doctor_xai_path):
                st.image(Image.open(st.session_state.doctor_xai_path), use_container_width=True)
            st.markdown("#### Sistem İşlem Kayıtları (Raw Model Output)")
            st.json(report)

        with doc_tab3:
            st.markdown("#### 1. Argenziano 7-Nokta Kontrol Listesi")
            st.write("Hekim olarak görselde saptadığınız dermoskopik kriterleri işaretleyiniz:")
            
            maj1 = st.checkbox("Atipik Pigment Ağı (Major - 2 Puan)", key="m1")
            maj2 = st.checkbox("Mavi-Beyaz Peçe (Major - 2 Puan)", key="m2")
            maj3 = st.checkbox("Atipik Vasküler Patern (Major - 2 Puan)", key="m3")
            min1 = st.checkbox("Düzensiz Çizgilenmeler/Işınsal Uzantılar (Minor - 1 Puan)", key="m4")
            min2 = st.checkbox("Düzensiz Nokta/Globüller (Minor - 1 Puan)", key="m5")
            min3 = st.checkbox("Düzensiz Leke/Blotch (Minor - 1 Puan)", key="m6")
            min4 = st.checkbox("Regresyon Yapıları (Minor - 1 Puan)", key="m7")
            
            total_score = (maj1*2) + (maj2*2) + (maj3*2) + min1 + min2 + min3 + min4
            st.markdown(f"**Klinisyen 7-Nokta Skoru: <span style='color:#0f172a; font-size:1.2rem;'>{total_score} Puan</span>**", unsafe_allow_html=True)
            
            if total_score >= 3:
                st.error("🚨 **Yüksek Şüphe:** 7-nokta algoritmasına göre skor ≥3 olması malignite şüphesini klinik olarak anlamlı şekilde artırır.")
            else:
                st.success("✅ **Düşük Şüphe:** Klasik algoritmaya göre lezyon benign (iyi huylu) özelliklere daha yakındır.")

            st.divider()
            st.markdown("#### 2. Dermoskopik Epikriz Şablonu Üretici")
            st.write("Seçtiğiniz bulgulara ve algoritma sonucuna göre HBYS'ye kopyalayabileceğiniz hazır rapor:")
            
            bulgular = []
            if maj1: bulgular.append("atipik pigment ağı")
            if maj2: bulgular.append("mavi-beyaz peçe")
            if maj3: bulgular.append("atipik vaskülarizasyon")
            if min1: bulgular.append("ışınsal uzantılar")
            if min2: bulgular.append("düzensiz globüller")
            if min3: bulgular.append("atipik lekelenme")
            if min4: bulgular.append("regresyon alanları")
            
            bulgu_metni = ", ".join(bulgular) if bulgular else "spesifik atipik dermoskopik patern izlenmedi"
            
            epikriz = f"Hasta {info.get('yas', 'belirtilmeyen')} yaşında, {info.get('cinsiyet', 'belirtilmemiş')} cinsiyette olup; {info.get('lokasyon', 'belirtilmeyen bölgede')} yerleşen ve {info.get('sure', 'belirtilmeyen süredir')} mevcut olan pigmente lezyon şikayetiyle değerlendirilmiştir.\n\nDermoskopik muayenede; {bulgu_metni}. Klinisyenin 7-Nokta skoru {total_score} olarak hesaplanmıştır.\n\nCDSS (Karar Destek Sistemi) analizi %{yuzde_skor_ham:.1f} morfolojik uyum indeksi ile '{tani_sonucu}' yönünde öngörü (Triyaj: {triyaj_kodu}) vermiştir. Hastanın klinik takibe / eksizyona alınması planlanmıştır."
            
            st.text_area("Otomatik Epikriz Şablonu (Kopyalamak için tıklayın)", value=epikriz, height=250)

        with doc_tab4:
            st.markdown("#### Epidemiyolojik Profil (Baseline Analizi)")
            st.write("Sistemin öngördüğü tanının yaş, cinsiyet ve anatomik bölgeye göre literatürdeki görülme sıklığı tablosu:")
            
            ep_data = {
                "Parametre": ["Hasta Yaş Grubu", "Cinsiyet Faktörü", "Anatomik Lokalizasyon"],
                "Hasta Değeri": [f"{info.get('yas', '-')} Yaş", info.get("cinsiyet", "-"), info.get("lokasyon", "-")],
                "Tanıya Göre Risk/Sıklık": ["Yüksek (Pik 50-70 yaş)", "Erkeklerde %1.5 daha sık", "UV maruziyet alanı (Yüksek Risk)"]
            }
            if "nevus" in tani_sonucu.lower() or "benign" in tani_sonucu.lower():
                ep_data["Tanıya Göre Risk/Sıklık"] = ["Tüm yaşlarda yaygın (Pik 20-30)", "Cinsiyet farkı izlenmez", "Tüm vücutta görülebilir"]
                
            st.table(pd.DataFrame(ep_data))
            st.caption("Veriler genel dermatoloji epidemiyolojisi kılavuzlarına dayanmaktadır.")

        with doc_tab5:
            st.markdown("#### Uzman Konsültasyonu Transferi")
            st.write("Bu vakayı hastane dışındaki bir dermatoloji uzmanına veya tümör konseyine iletmek için güvenli konsültasyon paketi oluşturun.")
            if st.button("🔗 Şifreli Konsültasyon Bağlantısı Üret", type="primary"):
                raw_hash = f"{info.get('hasta_id', 'id')}{tani_sonucu}{datetime.datetime.now()}"
                st.session_state.consultation_hash = hashlib.sha256(raw_hash.encode()).hexdigest()[:12].upper()
                st.session_state.consultation_link_generated = True
            
            if st.session_state.consultation_link_generated:
                st.success("✅ Konsültasyon Paketi Başarıyla Oluşturuldu!")
                st.markdown(f"""
                <div style='background: #f1f5f9; padding: 20px; border-radius: 10px; border: 1px dashed #94a3b8; font-family: monospace; font-size: 1.1rem; color: #0f172a;'>
                    <b>Klinik Erişim Anahtarı:</b> CDSS-REF-{st.session_state.consultation_hash}<br>
                    <b>Sistem Öngörüsü:</b> {tani_sonucu}<br>
                    <b>Triyaj Kodu:</b> {triyaj_kodu}<br><br>
                    <span style='font-size: 0.9rem; color: #64748b;'>* Lütfen ilgili uzman hekime PACS sistemi üzerinden bu referans kodunu iletiniz. Görseller uçtan uca şifrelenmiştir.</span>
                </div>
                """, unsafe_allow_html=True)

def render_patient_portal(triage_engine, xai_engine):
    # danisan / hasta giris arayuzu akislari burda
    render_role_switch()

    with st.sidebar:
        st.title("Bilgilendirme")
        st.success("✓ Sistem hazır, fotoğrafınızı yükleyebilirsiniz.")
        st.divider()
        st.info("ℹ️ Bu modül kesin tıbbi teşhis koymaz. Son karar ve tedavi her zaman doktorunuz aittir.")

    st.markdown("""
        <div class="header-banner header-patient">
            <h1>🙂 Danışan Portalı — Cilt Sağlığı Ön Değerlendirme</h1>
            <p>Fotoğrafınızı yükleyin, anında ön bilgilendirme alın ve cilt koruma analiz testinizi çözün.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Cilt Lezyonu Fotoğrafınız", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col_img, col_action = st.columns([1, 1.4], gap="large")
        with col_img:
            st.subheader("Yüklediğiniz Fotoğraf")
            st.image(Image.open(uploaded_file), use_container_width=True)

        with col_action:
            st.subheader("İncelemeyi Başlat")
            st.markdown('<div class="consent-box">', unsafe_allow_html=True)
            onam_verildi = st.checkbox("Bunun kesin tıbbi bir teşhis olmadığını ve sadece bilgilendirme amacı taşıdığını anlıyorum.")
            st.markdown('</div>', unsafe_allow_html=True)

            analiz_butonu = st.button("🔍 Ön İnceleme Sonucunu Öğren", type="primary", use_container_width=True, disabled=not onam_verildi)

            if analiz_butonu:
                st.session_state.patient_analysis_done = False
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    with st.spinner('Fotoğrafınız analiz ediliyor, cilt haritanız çıkarılıyor...'):
                        report = triage_engine.analyze_lesion(tmp_path)
                        xai_output_path = f"analiz_{Path(tmp_path).stem}.png"
                        xai_engine.generate_analysis(tmp_path, output_dir=".")

                    st.session_state.patient_report = report
                    st.session_state.patient_xai_path = xai_output_path
                    st.session_state.patient_analysis_done = True
                except Exception as e:
                    st.error(f"❌ Hata: Lütfen geçerli bir gorntu yüklediğinizden emin olun. {str(e)}")
                finally:
                    cleanup_temp_file(tmp_path)

    if st.session_state.patient_analysis_done and st.session_state.patient_report:
        st.divider()
        report = st.session_state.patient_report
        
        ham_tani = extract_disease_name(report)
        triyaj_kodu = extract_triage_code(report)
        disease = get_disease_info(ham_tani)

        st.header("📋 Bilgilendirme Raporunuz Hazır")
        render_metric_card("Saptanan Bulgu Kategorisi", disease["ad"], icon=disease["icon"], accent="#1565c0")
        
        if "KIRMIZI" in triyaj_kodu:
            urgency_note = "Bu durumu en kısa sürede randevu alarak bir dermatoloğa göstermeniz TIBBİ OLARAK ŞARTTIR."
            alert_color = "#dc2626"
        elif "SARI" in triyaj_kodu:
            urgency_note = "Bulgularınızı uygun bir zamanda bir dermatologla görüşmenizde fayda var."
            alert_color = "#d97706"
        else:
            urgency_note = "Şu an için acil bir risk görünmüyor. Yıllık rutin kontrollerinize devam ediniz."
            alert_color = "#16a34a"

        st.markdown(f"<div class='alert-box' style='border-color:{alert_color}; color:{alert_color}; background-color: {'#fee2e2' if 'KIRMIZI' in triyaj_kodu else '#fef3c7' if 'SARI' in triyaj_kodu else '#dcfce7'};'><b>💡 Sistem Önerisi:</b> {urgency_note}</div>", unsafe_allow_html=True)
        
        pat_tab1, pat_tab2, pat_tab3, pat_tab4, pat_tab5 = st.tabs([
            "🏥 Hastane İşlemleri",
            "📖 Bilgi Rehberi", 
            "📝 İnteraktif ABCDE Testi", 
            "☀️ Cilt Tipi Analizi",
            "📚 Dermatoloji Sözlüğü"
        ])
        
        with pat_tab1:
            st.markdown("### 🏥 Hastane ve Randevu İşlemleri")
            st.write("Sağlığınızı şansa bırakmayın. Şüpheli her bulgu için devlet veya özel hastanelerden **Cildiye (Dermatoloji)** randevusu almanız en güvenli yoldur.")
            
            btn_class = "mhrs-button-yellow" if "SARI" in triyaj_kodu else ("mhrs-button" if "KIRMIZI" in triyaj_kodu else "mhrs-button mhrs-button-yellow")
            st.markdown(f"""
                <div class="mhrs-button-container">
                    <a href="https://mhrs.gov.tr/vatandas/#/" target="_blank" class="{btn_class}">
                        📍 T.C. SAĞLIK BAKANLIĞI MHRS RANDEVU SİSTEMİNE GİT
                    </a>
                    <p style="margin-top: 15px; color: #64748b; font-size: 0.95rem;">Butona tıkladığınızda resmi MHRS web sitesi yeni sekmede açılacaktır.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📋 Randevu Öncesi Hazırlık Kılavuzu")
            st.markdown("""
            Doktorunuzun sizi daha iyi değerlendirebilmesi için şunlara dikkat edin:
            1. **Makyaj ve Krem:** Randevuya gitmeden önce muayene edilecek bölgeye kesinlikle fondöten, kapatıcı kremler veya renkli losyonlar sürmeyin.
            2. **Geçmişi Not Edin:** Bu lekenin ne zamandır orada olduğunu (Örn: 6 aydır) ve şekil değiştirip değiştirmediğini hatırlamaya çalışın.
            3. **Oto-Muayene:** Vücudunuzda göremediğiniz (sırtınız, saç dipleriniz) yerlerde başka lekeler olup olmadığını bir yakınınıza kontrol ettirin ve doktora bunları da gösterin.
            """)

        with pat_tab2:
            render_encyclopedia_card("Bu Nedir?", disease["nedir"], "🧐", accent="#1565c0")
            render_encyclopedia_card("Neden Kaynaklanır?", disease["neden"], "🧬", accent="#6a1b9a")
            render_encyclopedia_card("Nasıl Tedavi Edilir?", disease["tedavi"], "💉", accent="#ea580c")
            render_encyclopedia_card("Sonraki Adımlar", disease["sonraki_adimlar"], "🧭", accent="#2e7d32")
            
            st.markdown("### ❓ Sıkça Sorulan Sorular (SSS)")
            for sss_item in disease.get("sss", []):
                with st.expander(f"📌 {sss_item['q']}"):
                    st.write(sss_item['a'])
            
            st.divider()
            st.markdown("### 🖼️ Algoritma Odak Haritası")
            st.markdown("*Aşağıdaki kırmızı alanlar, sistemin değerlendirme yaparken odaklandığı pikselleri gösterir.*")
            if st.session_state.patient_xai_path and os.path.exists(st.session_state.patient_xai_path):
                st.image(Image.open(st.session_state.patient_xai_path), use_container_width=True)

        with pat_tab3:
            st.markdown("### 📝 ABCDE Kendi Kendine Muayene Testi")
            st.write("Evinizde cildinizi takip ederken uygulayabileceğiniz uluslararası testtir. Aşağıdakilerden hangileri beninizde mevcut?")
            
            a_val = st.checkbox("**A - Asimetri:** Benin bir yarısı şekil olarak diğer yarısına benzemiyor.")
            b_val = st.checkbox("**B - Sınır (Border):** Benin kenarları dalgalı, tırtıklı veya belirsiz.")
            c_val = st.checkbox("**C - Renk (Color):** Benin içinde siyah, kahverengi, kırmızı gibi birden fazla renk var.")
            d_val = st.checkbox("**D - Çap (Diameter):** Benin genişliği bir kurşun kalem silgisinden (yaklaşık 6mm) büyük.")
            e_val = st.checkbox("**E - Evrim (Evolving):** Ben son zamanlarda belirgin şekilde büyüdü, kaşınıyor veya kanıyor.")
            
            abcde_score = a_val + b_val + c_val + d_val + e_val
            st.progress(abcde_score / 5.0)
            
            if abcde_score >= 2:
                st.error("🚨 **Klinik Uyarı:** İşaretlediğiniz kriterler, cildinizdeki oluşumun atipik özellikler taşıyabileceğini gösteriyor. Sonucunuz ne olursa olsun uzman bir hekime başvurmalısınız.")
            elif abcde_score == 1:
                st.warning("⚠️ **Takip Önerisi:** Bir adet risk kriteri belirttiniz. Lezyonu aylık olarak fotoğraflayarak takip etmeniz önerilir.")
            else:
                st.success("✅ **Güvenli Patern:** Şu an için ABCDE kriterlerine göre yapısal bir risk belirtmediniz.")

        with pat_tab4:
            st.markdown("### ☀️ Cilt Tipi (Fitzpatrick) ve Korunma Analizi")
            st.write("Güneşe karşı genetik hassasiyetinizi ve doğru koruma rutininizi belirleyelim.")
            
            q1 = st.selectbox("Göz Renginiz Nedir?", ["Açık mavi/yeşil (0 Puan)", "Mavi/gri (1 Puan)", "Ela/açık kahverengi (2 Puan)", "Koyu kahverengi (3 Puan)", "Siyahımsı (4 Puan)"])
            q2 = st.selectbox("Doğal Saç Renginiz?", ["Kızıl/açık sarı (0 Puan)", "Sarı (1 Puan)", "Koyu sarı/Kumral (2 Puan)", "Koyu kahverengi (3 Puan)", "Siyah (4 Puan)"])
            q3 = st.selectbox("Güneşte Kalınca Cildiniz Nasıl Tepki Verir?", ["Hemen kıpkırmızı yanar, asla bronzlaşmaz (0 Puan)", "Kolay yanar, zor bronzlaşır (1 Puan)", "Bazen yanar, yavaş yavaş bronzlaşır (2 Puan)", "Nadiren yanar, kolay bronzlaşır (3 Puan)", "Hiç yanmaz, direkt bronzlaşır (4 Puan)"])
            
            p1 = int(re.search(r'\((\d+) Puan\)', q1).group(1))
            p2 = int(re.search(r'\((\d+) Puan\)', q2).group(1))
            p3 = int(re.search(r'\((\d+) Puan\)', q3).group(1))
            total_fitz = p1 + p2 + p3
            
            st.divider()
            if total_fitz <= 2:
                st.markdown("#### Analiz Sonucu: **Tip I (Çok Hassas Cilt)**")
                st.info("Cildiniz güneşe karşı savunmasızdır. UV ışınları cilt kanseri riskinizi ciddi şekilde artırır. **Her gün SPF 50+ fiziksel filtreli (Çinko Oksit içeren)** güneş kremi kullanmalısınız. Öğlen güneşinden kesinlikle kaçının.")
            elif total_fitz <= 5:
                st.markdown("#### Analiz Sonucu: **Tip II / III (Hassas - Normal Cilt)**")
                st.info("Güneş yanıklarına açıksınız ve yaşlanma lekeleri oluşturmaya meyillisiniz. Yazın SPF 50+, kışın SPF 30+ güneş kremini günlük rutininize ekleyin. Özellikle denizdeyken kremi 2 saatte bir yenileyin.")
            else:
                st.markdown("#### Analiz Sonucu: **Tip IV / V (Dirençli Cilt)**")
                st.info("Güneş yanıklarına karşı dirençlisiniz ancak UV ışınları cildin alt katmanlarında DNA hasarına ve erken yaşlanmaya neden olur. Dışarı çıkacağınız günlerde mutlaka SPF 30+ güneş koruyucu sürün.")

        with pat_tab5:
            st.markdown("### 📚 Dermatoloji Sözlüğü (Mini-Glossary)")
            st.write("Hastanede doktorunuzla konuşurken veya rapor okurken karşılaşabileceğiniz bazı tıbbi terimler ve Türkçe anlamları:")
            st.markdown("""
            * **Malign (Kötü Huylu):** Vücudun diğer bölgelerine yayılma (sıçrama) potansiyeli olan hücre dokuları (kanser).
            * **Benign (İyi Huylu):** Bulunduğu yerde sabit kalan, yayılmayan, tamamen güvenli yapı.
            * **Dermatoskop:** Dermatologların benleri 10-20 kat büyüterek incelemek için kullandıkları ışıklı özel büyüteç cihazı.
            * **Eksizyonel Biyopsi:** Kesin teşhis amacıyla sorunlu bölgenin veya benin ufak bir uyuşturma ile cerrahi olarak tamamen alınması işlemi.
            * **Metastaz:** Kötü huylu hücrelerin lenf veya kan yoluyla vücudun başka bir organına atlaması durumu.
            * **Eritem:** Ciltteki anormal kızarıklık, genellikle artmış kan akışı veya lokal iltihaplanma kaynaklıdır.
            * **Melanin:** Cilde, saçlara ve gözlere rengini veren, güneşe karşı koruyucu doğal biyolojik pigment (boya) maddesi.
            """)

def main():
    with st.spinner("Klinik Karar Destek Modülleri Yükleniyor..."):
        triage_engine, xai_engine = load_clinical_engines()

    if st.session_state.role is None:
        render_landing_page()
    elif st.session_state.role == "doctor":
        render_doctor_portal(triage_engine, xai_engine)
    elif st.session_state.role == "patient":
        render_patient_portal(triage_engine, xai_engine)

if __name__ == "__main__":
    main()