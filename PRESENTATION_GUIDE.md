# Sunum Kılavuzu - Conway Game of Life + Izhikevich Neuron Simülasyonu

## Oluşturulan Görseller

### 1. Temel Simülasyon Çıktıları
**Konum:** `outputs/presentation_demo/` ve `outputs/presentation_feedback/`

- **final_gol.png**: Son Conway Game of Life durumu (siyah-beyaz)
- **final_v.png**: Son membran potansiyeli heatmap (renkli)
- **spike_raster.png**: Spike raster plot (zaman × hücre indeksi)
- **anim.gif**: Animasyon (eğer GIF oluşturulduysa)
- **metrics.csv**: Tüm zaman serisi verileri

### 2. Sunum Görselleri
**Konum:** `outputs/presentation_plots/`

#### a) Zaman Serisi Grafikleri
- **time_series_demo.png**: Feedback olmadan simülasyon metrikleri
  - Alive cell sayısı
  - Spike sayısı
  - Ortalama membran potansiyeli
  - Firing rate
  
- **time_series_feedback.png**: Feedback ile simülasyon metrikleri
  - Aynı metrikler, feedback etkisiyle

#### b) Karşılaştırma Görselleri
- **comparison_feedback.png**: Feedback etkisinin karşılaştırması
  - İki simülasyonun yan yana karşılaştırması
  - Feedback'in Conway ve neuron davranışına etkisi

#### c) Mekanizma Açıklaması
- **coupling_mechanism.png**: Coupling mekanizmasının görsel açıklaması
  - GoL state → Neighbor count → Input current akışı
  - 3 panel: GoL state, neighbor count, input current

## Sunum Önerileri

### Slide 1: Giriş ve Motivasyon
- **Kullanılacak görsel:** `coupling_mechanism.png`
- **Açıklama:** 
  - Conway Game of Life ve Izhikevich neuron modellerinin birleşimi
  - Bidirectional coupling kavramı
  - Neden bu kombinasyon ilginç?

### Slide 2: Metodoloji
- **Kullanılacak görseller:** 
  - `coupling_mechanism.png` (mekanizma detayı)
  - Kod yapısı diyagramı (README'deki mimari)
- **Açıklama:**
  - GoL → Neuron: I = k_neighbors × neighbors + k_alive × alive
  - Neuron → GoL: Spike'lar alive hücreleri etkiler (feedback)

### Slide 3: Simülasyon Sonuçları - Temel
- **Kullanılacak görseller:**
  - `time_series_demo.png` (4 panel)
  - `final_gol.png` ve `final_v.png` yan yana
- **Açıklama:**
  - Conway dinamikleri (alive cell sayısı azalıyor)
  - Neuron aktivitesi (spike sayıları)
  - Membran potansiyeli değişimi

### Slide 4: Feedback Etkisi
- **Kullanılacak görseller:**
  - `comparison_feedback.png` (4 panel karşılaştırma)
- **Açıklama:**
  - Feedback açık/kapalı karşılaştırması
  - Feedback'in Conway'e etkisi
  - Feedback'in neuron aktivitesine etkisi
  - Sonuçlar: Feedback ile daha dinamik sistem

### Slide 5: Animasyon ve Dinamikler
- **Kullanılacak görseller:**
  - `anim.gif` (animasyon)
  - `spike_raster.png`
- **Açıklama:**
  - Zaman içinde Conway pattern'lerinin evrimi
  - Spike aktivitesinin zaman içinde dağılımı
  - Sistemin emergent davranışları

### Slide 6: Sonuçlar ve Gelecek Çalışmalar
- **Kullanılacak görseller:**
  - Özet metrikler tablosu (metrics.csv'den)
- **Açıklama:**
  - Hibrit sistemin avantajları
  - Olası uygulamalar
  - Gelecek çalışmalar için öneriler

## Sunum İpuçları

### Görsel Kullanımı
1. **Yüksek çözünürlük:** Tüm görseller 200 DPI'da kaydedildi, sunumda net görünecek
2. **Renk paleti:** 
   - Conway: Siyah-beyaz (gray colormap)
   - Membran potansiyeli: Viridis (yeşil-mavi)
   - Spike aktivitesi: Kırmızı
   - Input current: Hot (sarı-kırmızı)

### Animasyon Kullanımı
- GIF'i sunumda gösterirken:
  - İlk birkaç saniye durdurup açıklama yapın
  - Sonra animasyonu oynatın
  - Önemli anları işaret edin (pattern oluşumu, spike patlamaları)

### Metriklerin Sunumu
- `metrics.csv` dosyasından önemli istatistikleri çıkarın:
  - Ortalama alive cell sayısı
  - Toplam spike sayısı
  - Firing rate ortalaması
  - Membran potansiyeli varyansı

### Teknik Detaylar (İsteğe Bağlı)
- Grid boyutu: 80×80
- Simülasyon adımları: 400
- Time step (dt): 0.1
- Coupling parametreleri: k_neighbors=0.5, k_alive=2.0 (veya 3.0 feedback'te)

## Ek Görsel Önerileri

### 1. Side-by-Side Karşılaştırma
İki simülasyonun final state'lerini yan yana gösterin:
- Sol: Feedback kapalı
- Sağ: Feedback açık

### 2. Zaman Serisi Özeti
Tek bir grafikte tüm metrikleri overlay edin (normalize edilmiş)

### 3. Pattern Analizi
Conway pattern'lerinin evrimini gösteren frame'ler:
- Başlangıç (step 0)
- Orta (step 200)
- Son (step 400)

### 4. Spike Aktivite Haritası
Hangi hücrelerin daha aktif olduğunu gösteren heatmap

## Komutlar

### Simülasyon Çalıştırma
```bash
# PYTHONPATH ayarla
$env:PYTHONPATH="C:\Users\ufukf\OneDrive\Desktop\tez\src"

# Temel simülasyon
python -m scripts.run_grid --height 80 --width 80 --steps 400 --seed 42 --gif --frame-stride 5

# Feedback ile
python -m scripts.run_grid --height 80 --width 80 --steps 400 --seed 42 --feedback --k-alive 3.0

# Sunum görselleri oluştur
python -m scripts.create_presentation_plots
```

## Dosya Yapısı

```
outputs/
├── presentation_demo/          # Temel simülasyon
│   ├── final_gol.png
│   ├── final_v.png
│   ├── spike_raster.png
│   ├── metrics.csv
│   └── anim.gif
├── presentation_feedback/     # Feedback ile simülasyon
│   └── ...
└── presentation_plots/         # Sunum görselleri
    ├── time_series_demo.png
    ├── time_series_feedback.png
    ├── comparison_feedback.png
    └── coupling_mechanism.png
```

## Sunum Süresi Önerisi

- **Toplam:** 10-15 dakika
- **Giriş:** 2 dakika
- **Metodoloji:** 3 dakika
- **Sonuçlar:** 5-7 dakika
- **Sonuç ve Tartışma:** 2-3 dakika

## Sorulara Hazırlık

### Olası Sorular:
1. **Neden Conway + Izhikevich?**
   - İki farklı dinamik sistemin birleşimi
   - Emergent davranışlar
   - Biyolojik sistemlerde benzer coupling mekanizmaları

2. **Feedback'in etkisi nedir?**
   - Daha dinamik sistem
   - Spike aktivitesinin Conway'e geri beslemesi
   - Karşılaştırma grafiklerinde görülebilir

3. **Parametreler nasıl seçildi?**
   - Izhikevich: Standart parametreler (a=0.02, b=0.2, c=-65, d=8)
   - Coupling: Deneme-yanılma ile optimize edildi
   - Grid boyutu: Performans ve görsel kalite dengesi

4. **Gerçek uygulamalar?**
   - Biyolojik sistemlerin modellenmesi
   - Emergent davranış çalışmaları
   - Hybrid dinamik sistemler

