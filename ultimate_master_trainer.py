import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
from sklearn.utils import class_weight
import math

print("\n" + "█"*65)
print(" 🧬 ULTIMATE DERMOSKOPİ SİSTEMİ EĞİTİMİ (SOTA MİMARİSİ)")
print("█"*65)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = 'data'
EPOCHS = 35  # cosine decay kullanacagimiz icin epoch sayisini biraz yuksek tuttum
INITIAL_LR = 1e-5
MAX_LR = 1e-3

print("⏳ [1/5] Veri setleri yükleniyor ve ağırlıklar hesaplanıyor...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", 
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", 
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

# kasner verisi az oldugu icin class weight hesaplayip agirligi dengelemek lazim
train_labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {0: weights[0], 1: weights[1]}
print(f"⚖️ Dinamik Sınıf Ağırlıkları Hesaplandı: İyi Huylu={weights[0]:.2f}, Kötü Huylu={weights[1]:.2f}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("🔬 [2/5] Tensör Bazlı Medikal Veri Artırma Modülü Devrede...")
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.3, 0.3),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.25),
    # model ezbere gitmesin diye dokuya azicik noise ekliyoruz
    layers.GaussianNoise(0.01) 
], name="medikal_augmentation")

def squeeze_excite_block(inputs, ratio=8):
    b, _, _, c = inputs.shape
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(c // ratio, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dense(c, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Reshape((1, 1, c))(x)
    return layers.Multiply()([inputs, x])

print("🧠 [3/5] SOTA (State-of-the-Art) Ağ Mimarisi İnşa Ediliyor...")

base_model = tf.keras.applications.EfficientNetB4(
    input_shape=IMG_SIZE + (3,), 
    include_top=False, 
    weights='imagenet'
)

# son 50 katmani fine tune icin acik birakiyoruz
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False) 

# araya se_block ekleyip attention yapıyoruz
x = squeeze_excite_block(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

# ezberlemeyi kesmek icin buralarda dropout yuksek tutulmalı
x = layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

print("⚙️ [4/5] AdamW Optimizer ve Dinamik Kayıp Fonksiyonları Ayarlanıyor...")

# tibbi datada adamw agirlik curumesini en iyi yoneten optmizer duruma gore degistirilebilir
optimizer = tf.keras.optimizers.AdamW(learning_rate=MAX_LR, weight_decay=1e-4)

# label smoothing bce ile birlesince focal loss gibi calisiyor, daha stabil
loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

metrics = [
    'accuracy',
    tf.keras.metrics.AUC(name='auc_roc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

print("📉 [5/5] Öğrenme Hızı: Kosinüs Tavlaması (Cosine Annealing) Zamanlayıcısı Aktif.")

def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        # ilk 3 epoch isinma asamasi parametreleri soke sokmamak icin
        return INITIAL_LR + (MAX_LR - INITIAL_LR) * (epoch / warmup_epochs)
    else:
        # kalan kısımlarda kosinus grafigine gore ogrenme hizini yavaslatiyoruz
        progress = epoch - warmup_epochs
        decay_steps = EPOCHS - warmup_epochs
        return INITIAL_LR + 0.5 * (MAX_LR - INITIAL_LR) * (1 + math.cos(math.pi * progress / decay_steps))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup, verbose=1)

callbacks = [
    lr_scheduler,
    tf.keras.callbacks.ModelCheckpoint(
        'ultimate_skin_model.keras', 
        save_best_only=True, 
        monitor='val_auc_roc', # loss yerine direkt auc takibi yapiyoruz en mantiklisi bu
        mode='max', 
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor='val_auc_roc', patience=10, restore_best_weights=True, mode='max', verbose=1)
]

print("\n" + "═"*65)
print(" 🔥 ULTIMATE EĞİTİM BAŞLIYOR (Model zayıf noktalarını parçalıyor...)")
print(" Lütfen bilgisayarı prize takılı tutun.")
print("═"*65 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights_dict, # yukarda hesapladigimiz dynamic agirliklari buraya verdik
    callbacks=callbacks
)

print("\n" + "█"*65)
print(" ✅ SOTA (EN İYİ) EĞİTİM KUSURSUZ ŞEKİLDE TAMAMLANDI!")
print(" 🎯 Tıbbi literatüre uygun modelin 'ultimate_skin_model.keras' olarak kaydedildi.")
print("█"*65)