# --- MAC KİLİTLENME KORUMASI ---
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

print("🚀 FLAWLESS (KUSURSUZ) MODEL EĞİTİMİ BAŞLIYOR...")

# 1. MODERN VERİ YÜKLEME (tf.data API)
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data', validation_split=0.2, subset="training", seed=42,
    image_size=(224, 224), batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'data', validation_split=0.2, subset="validation", seed=42,
    image_size=(224, 224), batch_size=32)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. GELİŞMİŞ VERİ ARTIRMA (Işık ve Ten Rengi Toleransı)
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.35),
    RandomZoom(0.25),
    RandomBrightness(factor=0.2), 
    RandomContrast(factor=0.2)    
])

# 3. YÜKSELTİLMİŞ MİMARİ VE DERİN KARAR KATMANI
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x) 

outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# 4. FOCAL LOSS & SINIF AĞIRLIKLARI (Dengeleme)
class_weight = {0: 0.62, 1: 2.56} 

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')])

callbacks_phase1 = [
    EarlyStopping(monitor='val_auc', patience=4, restore_best_weights=True, mode='max'),
    ModelCheckpoint('best_model_phase1.keras', monitor='val_auc', mode='max', save_best_only=True)
]

# ---------------------------------------------------------
# FAZ 1: ISINMA 
# ---------------------------------------------------------
print("\n" + "="*50)
print(" 🟢 FAZ 1: ISINMA BAŞLIYOR (Üst Katmanlar Eğitimde)")
print("="*50)
history_1 = model.fit(train_ds, validation_data=val_ds, epochs=8, class_weight=class_weight, callbacks=callbacks_phase1)

# ---------------------------------------------------------
# FAZ 2: FINE-TUNING (Derin Öğrenme / Zincirler Kırılıyor)
# ---------------------------------------------------------
print("\n" + "="*50)
print(" 🔥 FAZ 2: MİKRO-AYAR BAŞLIYOR (Tüm Model Yeniden Şekilleniyor)")
print("="*50)

base_model.trainable = True
for layer in base_model.layers[:-30]: 
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')])

callbacks_phase2 = [
    EarlyStopping(monitor='val_auc', patience=6, restore_best_weights=True, mode='max'),
    ModelCheckpoint('flawless_skin_model.keras', monitor='val_auc', mode='max', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7, verbose=1)
]

history_2 = model.fit(train_ds, validation_data=val_ds, epochs=20, class_weight=class_weight, callbacks=callbacks_phase2)

# ---------------------------------------------------------
# FAZ 3: AKADEMİK RAPORLAMA VE GÖRSELLEŞTİRME
# ---------------------------------------------------------
print("\n" + "="*50)
print(" 📊 FAZ 3: AKADEMİK RAPOR VE GRAFİKLER HAZIRLANIYOR")
print("="*50)

# Gerçek etiketleri (y_true) tf.data içerisinden çıkarma
y_true = np.concatenate([y for x, y in val_ds], axis=0)

# Modelin tahminleri (y_pred)
y_pred_probs = model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# Sınıflandırma Raporu (Jürinin aradığı tablo)
print("\n             SINIFLANDIRMA RAPORU")
print("-" * 45)
print(classification_report(y_true, y_pred, target_names=['Benign (İyi)', 'Malignant (Kötü)']))

# Kesintisiz Grafik Çizimi (Faz 1 ve Faz 2 Birleştirilir)
acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
loss = history_1.history['loss'] + history_2.history['loss']
val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

plt.figure(figsize=(12, 5))

# Doğruluk Grafiği
plt.subplot(1, 2, 1)
plt.plot(acc, label='Eğitim Başarısı', color='blue')
plt.plot(val_acc, label='Test Başarısı', color='cyan', linestyle='--')
plt.axvline(x=len(history_1.history['accuracy'])-1, color='red', linestyle=':', label='Faz 2 Başlangıcı')
plt.title('Kesintisiz Model Başarısı (Accuracy)')
plt.legend()

# Hata (Loss) Grafiği
plt.subplot(1, 2, 2)
plt.plot(loss, label='Eğitim Hatası', color='orange')
plt.plot(val_loss, label='Test Hatası', color='red', linestyle='--')
plt.axvline(x=len(history_1.history['loss'])-1, color='red', linestyle=':', label='Faz 2 Başlangıcı')
plt.title('Kesintisiz Model Hatası (Focal Loss)')
plt.legend()

plt.tight_layout()
plt.savefig('akademik_egitim_grafigi.png', dpi=300) # dpi=300 teze koymak için yüksek çözünürlük

print("\n✅ FLAWLESS EĞİTİM TAMAMLANDI!")
print("✅ Şaheserin 'flawless_skin_model.keras' olarak projene kaydedildi.")
print("✅ Tezine ekleyeceğin yüksek çözünürlüklü grafik 'akademik_egitim_grafigi.png' olarak oluşturuldu.")