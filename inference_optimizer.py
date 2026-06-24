# --- MAC KİLİTLENME KORUMASI VE TEMİZLİK ---
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

plt.style.use('seaborn-v0_8-whitegrid')

def load_data_and_predict(model_path, data_dir, img_size=(224, 224), batch_size=32):
    print("\n⏳ 1. Model ve doğrulama verisi yükleniyor...")
    model = tf.keras.models.load_model(model_path)

    # DİKKAT: shuffle=False KALDIRILDI! Orijinal eğitimdeki gibi adil dağılım için seed kullanıyoruz.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    print("🧠 2. Tahminler yapılıyor ve gerçek etiketlerle eşleştiriliyor...")
    y_true = []
    y_pred_probs = []

    # Etiketlerin ve tahminlerin kaymaması için aynı döngü içinde çekiyoruz
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    return y_true, y_pred_probs

def find_optimal_medical_threshold(y_true, y_pred_probs, target_precision=0.75):
    print("\n⚙️ 3. Klinik Karar Eşiği (Threshold) F-Beta Skoru ile Optimize Ediliyor...")
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
    
    beta = 0.5 
    f_beta_scores = ((1 + beta**2) * (precisions[:-1] * recalls[:-1])) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-10)
    
    valid_indices = np.where(precisions[:-1] >= target_precision)[0]
    
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(f_beta_scores[valid_indices])]
        strategy = f"Hedeflenen Kesinlik Kriteri Başarılı (>{target_precision})"
    else:
        best_idx = np.argmax(f_beta_scores)
        strategy = "Maksimum F0.5 Skoru (En Dengeli Tıbbi Karar Noktası)"
        
    optimal_threshold = thresholds[best_idx]
    
    return optimal_threshold, precisions, recalls, thresholds, best_idx, strategy

def generate_academic_report(y_true, y_pred_probs, optimal_threshold, precisions, recalls, thresholds, best_idx):
    print(f"\n" + "="*60)
    print(" 🏥 KLİNİK KARAR DESTEK SİSTEMİ OPTİMİZASYON RAPORU")
    print("="*60)
    print(f"🔹 Eski Standart Karar Eşiği : 0.5000")
    print(f"🔹 YENİ KLİNİK KARAR EŞİĞİ  : {optimal_threshold:.4f}")
    print(f"🔹 Beklenen Precision       : {precisions[best_idx]:.2f}")
    print(f"🔹 Beklenen Recall          : {recalls[best_idx]:.2f}")
    print("-" * 60)

    y_pred_optimal = (y_pred_probs >= optimal_threshold).astype(int)
    
    print("\n📋 YENİ SINIFLANDIRMA RAPORU (Jüri Tablosu)")
    print(classification_report(y_true, y_pred_optimal, target_names=['Benign (İyi Huylu)', 'Malignant (Kötü Huylu)']))

    print("📊 4. Akademik Grafikler Çiziliyor...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(thresholds, precisions[:-1], color='#2ecc71', lw=2.5, label='Precision (Kesinlik)')
    axes[0].plot(thresholds, recalls[:-1], color='#e74c3c', lw=2.5, label='Recall (Duyarlılık)')
    axes[0].axvline(x=optimal_threshold, color='#34495e', linestyle='--', lw=2, 
                    label=f'Optimum Eşik ({optimal_threshold:.2f})')
    
    axes[0].scatter(optimal_threshold, precisions[best_idx], color='#2ecc71', s=100, zorder=5)
    axes[0].scatter(optimal_threshold, recalls[best_idx], color='#e74c3c', s=100, zorder=5)

    axes[0].set_title('Klinik Karar Eşiği Optimizasyon Eğrisi', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Karar Eşiği Olasılığı (Threshold)', fontsize=12)
    axes[0].set_ylabel('Performans Skoru', fontsize=12)
    axes[0].legend(fontsize=11)
    
    cm = confusion_matrix(y_true, y_pred_optimal)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=['Benign (İyi)', 'Malignant (Kötü)'], 
                yticklabels=['Benign', 'Malignant'])
    axes[1].set_title(f'Optimize Edilmiş Karmaşıklık Matrisi', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Gerçek Tanı (Ground Truth)', fontsize=12)
    axes[1].set_xlabel('Modelin Yeni Kararı', fontsize=12)

    plt.tight_layout()
    plt.savefig('klinik_karar_optimizasyonu.png', dpi=300, bbox_inches='tight')
    print("✅ Başarılı! 'klinik_karar_optimizasyonu.png' dosyası kaydedildi.")

if __name__ == "__main__":
    MODEL_PATH = 'flawless_skin_model.keras'
    DATA_DIR = 'data' 
    TARGET_PRECISION = 0.75 
    
    try:
        y_true, y_pred_probs = load_data_and_predict(MODEL_PATH, DATA_DIR)
        
        opt_thresh, precisions, recalls, thresholds, best_idx, strategy = find_optimal_medical_threshold(
            y_true, y_pred_probs, target_precision=TARGET_PRECISION)
        
        print(f"\n💡 Optimizasyon Stratejisi: {strategy}")
        generate_academic_report(y_true, y_pred_probs, opt_thresh, precisions, recalls, thresholds, best_idx)
        
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")