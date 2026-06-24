import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [🔵 %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

class Config:
    MODEL_PATH = 'ultimate_skin_model.keras'
    IMG_SIZE = (224, 224)
    # 14. epoch sonrasi cikan auc degeri (0.9018) icin en optimum esikler bunlar
    SAFE_THRESHOLD = 0.35  
    RISK_THRESHOLD = 0.65  

@tf.keras.utils.register_keras_serializable()
def squeeze_excite_block(inputs, ratio=8):
    from tensorflow.keras import layers, regularizers
    b, _, _, c = inputs.shape
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(c // ratio, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dense(c, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Reshape((1, 1, c))(x)
    return layers.Multiply()([inputs, x])

class TriageEngine:
    def __init__(self, config=Config):
        self.cfg = config
        logging.info("Triage Engine başlatılıyor...")
        self.model = self._load_ai_brain()
        logging.info(f"Klinik Eşikler Aktif: [Güvenli < {self.cfg.SAFE_THRESHOLD} | Gri Alan | Riskli > {self.cfg.RISK_THRESHOLD}]")

    def _load_ai_brain(self):
        if not os.path.exists(self.cfg.MODEL_PATH):
            logging.error(f"Kritik Hata: '{self.cfg.MODEL_PATH}' bulunamadı. Lütfen eğitim dosyanızı kontrol edin.")
            sys.exit(1)
        try:
            return tf.keras.models.load_model(
                self.cfg.MODEL_PATH, 
                custom_objects={'squeeze_excite_block': squeeze_excite_block}
            )
        except Exception as e:
            logging.error(f"Model yüklenirken donanımsal/yazılımsal bir hata oluştu: {str(e)}")
            sys.exit(1)

    def _preprocess_image(self, img_path: str) -> np.ndarray:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Görsel bulunamadı: {img_path}")
        
        # modele girmeden once resmi 224x224 yapiyoruz
        img = image.load_img(img_path, target_size=self.cfg.IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def analyze_lesion(self, img_path: str) -> dict:
        try:
            logging.info(f"Analiz ediliyor: {os.path.basename(img_path)}")
            img_tensor = self._preprocess_image(img_path)
            
            # buraya grad-cam isi haritasi eklencek unutma!!
            prediction_score = float(self.model.predict(img_tensor, verbose=0)[0][0])
            
            return self._build_clinical_report(prediction_score)
            
        except Exception as e:
            logging.error(f"Analiz başarısız: {str(e)}")
            return {"error": str(e)}

    def _build_clinical_report(self, score: float) -> dict:
        # doktorlarin raproda rahat gormesi icin formati ayarla
        report = {
            "risk_skoru": score,
            "yuzde_format": f"%{score * 100:.1f}",
            "triyaj_kodu": "",
            "tani": "",
            "protokol": ""
        }

        if score < self.cfg.SAFE_THRESHOLD:
            report["triyaj_kodu"] = "🟢 YEŞİL (Rutin)"
            report["tani"] = "Benign (İyi Huylu) Karakter"
            report["protokol"] = "Rutin yıllık takip yeterlidir."
            
        elif score > self.cfg.RISK_THRESHOLD:
            report["triyaj_kodu"] = "🔴 KIRMIZI (Acil)"
            report["tani"] = "Malignant (Kötü Huylu) Şüphesi"
            report["protokol"] = "Acil Dermato-Onkoloji konsültasyonu ve biyopsi gereklidir."
            
        else:
            report["triyaj_kodu"] = "🟡 SARI (Gri Alan)"
            report["tani"] = "Atipik Sınır Lezyon"
            report["protokol"] = "Sistemik yapay zeka kararsızlığı. Uzman hekim tarafından dermoskopik inceleme önerilir."

        return report

if __name__ == "__main__":
    print("\n" + "═"*60)
    print(" 🏥 MEDİKAL TRİYAJ SİSTEMİ V2.0 ".center(60))
    print("═"*60 + "\n")
    
    engine = TriageEngine()
    
    while True:
        img_input = input("\n📁 Analiz edilecek fotoğrafın yolu ('çıkış' için q): ").strip()
        
        if img_input.lower() in ['q', 'çıkış', 'quit']:
            logging.info("Sistem kapatılıyor. İyi günler.")
            break
            
        if not img_input:
            continue
            
        rapor = engine.analyze_lesion(img_input)
        
        if "error" not in rapor:
            print("\n" + "█"*60)
            print(f" 📌 TRİYAJ KODU : {rapor['triyaj_kodu']}")
            print(f" 🔬 RİSK SKORU  : {rapor['yuzde_format']}")
            print(f" 📋 TANI        : {rapor['tani']}")
            print(f" 💊 PROTOKOL    : {rapor['protokol']}")
            print("█"*60 + "\n")