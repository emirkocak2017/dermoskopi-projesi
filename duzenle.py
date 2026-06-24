import os
import shutil
import pandas as pd

# CSV dosyanı oku
csv_path = 'HAM10000_metadata.csv'
source_folder = 'all_images'
target_base = 'data'

if not os.path.exists(csv_path):
    print(f"HATA: {csv_path} bulunamadı!")
    exit()

df = pd.read_csv(csv_path)

# Klasörleri oluştur
os.makedirs(os.path.join(target_base, 'benign'), exist_ok=True)
os.makedirs(os.path.join(target_base, 'malignant'), exist_ok=True)

# İstatistikler için sayaçlar
counts = {'benign': 0, 'malignant': 0}
malignant_labels = ['mel', 'bcc', 'akiec']

print(f"İşlem başlıyor... Toplam {len(df)} resim kontrol edilecek.")

for index, row in df.iterrows():
    img_name = row['image_id'] + '.jpg'
    source_path = os.path.join(source_folder, img_name)
    
    if os.path.exists(source_path):
        category = 'malignant' if row['dx'] in malignant_labels else 'benign'
        target_path = os.path.join(target_base, category, img_name)
        
        # Kopyala
        shutil.copy(source_path, target_path)
        counts[category] += 1
        
        # 500 resimde bir güncelleme ver (ekranı doldurmamak için)
        if (index + 1) % 500 == 0:
            print(f"İşleniyor: {index + 1}/{len(df)}...")
    else:
        # Eğer resim klasörde yoksa uyarı ver
        continue

print("-" * 30)
print("İŞLEM TAMAMLANDI!")
print(f"Toplam Taşınan Resimler:")
print(f"İyi Huylu (Benign): {counts['benign']}")
print(f"Kötü Huylu (Malignant): {counts['malignant']}")
print(f"Model eğitimine hazırsın!")