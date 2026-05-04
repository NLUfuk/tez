# Proje Özeti: Conway + Izhikevich Hibrit Simülasyon

## 1) Projenin Amacı

Bu projenin ana amacı, **Conway Game of Life tabanlı compute-gating yaklaşımının** Izhikevich nöron dinamikleri ile birleştirildiğinde:

- hesaplama/verimlilik tarafında kazanım sağlayıp sağlamadığını,
- dinamik kararlılığı koruyup korumadığını,
- ölçülebilir metriklerle ne kadar sürdürülebilir bir hibrit model sunduğunu

sistematik olarak test etmektir.

Kısa tanım: Bu repo bir simülasyon kodundan fazlasıdır; aynı zamanda deney, metrik üretimi, görselleştirme ve verimlilik tahmini için araştırma altyapısıdır.

## 2) Kapsam ve Üst Düzey Mimari

Temel katmanlar:

- `src/conway_izh/`: Çekirdek simülasyon paketleri
- `scripts/`: Çalıştırma, deney, canlı görselleştirme ve veri üretim scriptleri
- `tests/`: Birim ve smoke testler
- `web/neuron_network/`: İnteraktif web demosu (p5.js tabanlı)

Çekirdek sorumluluk dağılımı:

- `config.py`: Simülasyon parametreleri
- `conway.py`: Conway state yönetimi, komşu sayımı, B3/S23 güncelleme
- `izhikevich.py`: Vektörize nöron dinamiği (membran potansiyeli + recovery)
- `coupling.py`: Conway -> nöron ve nöron -> Conway sinyal bağlama
- `game_theory_coupling.py`: Oyun teorisi destekli spike üretim/feedback yaklaşımı
- `grid.py`: Tüm adımları orkestre eden ana döngü (`NeuralGrid`)
- `metrics.py`, `efficiency.py`, `viz.py`: Ölçüm, skor ve çıktı üretimi

## 3) Conway Tarafı (Detaylı Teknik Akış)

### 3.1 Veri Modeli

- Conway grid: `H x W` boyutlu `numpy` dizi
- Hücre durumu: `0` (ölü) / `1` (canlı)
- Komşuluk: 8-komşu (Moore neighborhood), opsiyonel wrap-around

### 3.2 Kural Seti

Klasik Conway **B3/S23** uygulanır:

- Birth (B3): Ölü hücre, tam 3 komşuda canlı olur
- Survival (S23): Canlı hücre, 2 veya 3 komşuda canlı kalır
- Diğer durumlar: Hücre ölür veya ölü kalır

### 3.3 Hibrit Döngü İçindeki Konumu

Her adımda genel olarak:

1. Conway komşu sayımları alınır.
2. Conway özellikleri kullanılarak nöronlara input current hesaplanır.
3. Izhikevich adımı çalışır ve spike maskesi oluşur.
4. Conway grid bir sonraki jenerasyona güncellenir.
5. Seçenek açıksa nöron spike'ları Conway tarafını geri etkiler.
6. Metrikler hesaplanır, görselleştirme/çıktılar yazılır.

Bu tasarım, Conway'i yalnızca görsel bir katman olarak değil, nöral aktiviteyi kapılayan ve şekillendiren bir sinyal kaynağı olarak konumlandırır.

## 4) Game-Theory Coupling (Conway Odaklı Genişleme)

`game_theory_coupling.py` içinde Conway kuralları, spike olasılığına ve yayılımına daha güçlü şekilde bağlanır.

Öne çıkan fikirler:

- Conway durumuna göre spike olasılığı modülasyonu
- Spike propagation (komşulara etkileyici sinyal taşınması)
- İşbirliği/rekabet benzeri etki terimleri ile karar mekanizması

Pratik etkisi:

- Normal moda kıyasla daha yüksek aktivite ve daha zengin emergent davranış
- Parametre ayarına daha duyarlı, deneysel karakteri daha yüksek bir sistem

Not: Bu mod daha güçlü davranış üretse de performans/deterministiklik açısından ek mühendislik ihtiyacı doğurur (bkz. Bölüm 8).

## 5) Çalıştırma Akışı

Temel çalıştırmalar:

- Ana grid simülasyonu: `python -m scripts.run_grid ...`
- Canlı görsel çalışma: `python -m scripts.run_live ...`
- Game-theory canlı varyant: `python -m scripts.run_live_gametheory ...`
- Tek nöron referans senaryosu: `python -m scripts.run_single ...`

Tipik çıktı konumu:

- `outputs/<run_id>/`
- Final görseller, spike raster, zaman serisi metrikleri, opsiyonel GIF/frame çıktıları

## 6) Web Demo (Conway ile İlişki)

`web/neuron_network/` altındaki demo, çekirdek Python modelinin bire bir kopyası değil; Conway benzeri mantığı gerçek-zamanlı görsel/etkileşimli bir ağ modeli içinde kullanır.

Demo üzerinden:

- topoloji ve akım gibi parametreler değiştirilebilir,
- jenerasyon geçişleri izlenebilir,
- aktivite dinamikleri hızlıca kıyaslanabilir.

Bu katman, araştırma çıktısını anlatma ve davranışı sezgisel doğrulama için önemlidir.

## 7) Başarı Kriterleri (Net Hedef Tanımı)

Projenin hedefi aşağıdaki üç eksende ölçülmelidir:

1. **Doğruluk ve kararlılık**
   - Conway kuralları beklenen şekilde çalışıyor mu?
   - Nöron dinamiği numerik olarak stabil mi?

2. **Verimlilik**
   - Compute-gating ile işlem maliyeti/aktivite yükü azalıyor mu?
   - Aynı davranış seviyesi daha düşük maliyetle üretilebiliyor mu?

3. **Açıklanabilir ölçüm**
   - Metrikler tekrarlanabilir ve karşılaştırılabilir mi?
   - Parametre değişimlerinin etkisi izlenebilir mi?

Önerilen resmi amaç cümlesi:

> Conway tabanlı compute-gating yaklaşımını Izhikevich nöron dinamikleriyle birleştirerek, hibrit ağlarda verimlilik-karlılık dengesini ölçülebilir metriklerle optimize eden deneysel bir simülasyon altyapısı geliştirmek.

## 8) Teknik Borç ve İyileştirme Öncelikleri

Kısa vadede yüksek etkili iyileştirmeler:

- `game_theory_coupling.py` içindeki yoğun Python looplarının vektörizasyonu
- Rastgelelik akışının seed kontrollü tekil RNG ile deterministik hale getirilmesi
- Game-theory akışı için ayrı test kapsamı (özellikle karar fonksiyonları)
- Parametre tutarlılığı: kullanılmayan veya etkisiz parametrelerin temizlenmesi

Orta vadede:

- performans profil çıkarımı ve darboğaz bazlı optimizasyon,
- deney sonuçlarının standart rapor formatında toplanması,
- web demo ile çekirdek model davranışı arasında açık eşleme dokümantasyonu.

## 9) Kısa Sonuç

Bu proje, Conway'i klasik hücresel otomat sınırından çıkarıp nöral hesaplama için bir gating/sinyal mekanizmasına dönüştüren, araştırma odaklı bir hibrit platformdur. Başarının anahtarı, davranış zenginliğini korurken ölçülebilir verimlilik kazancı üretmek ve bunu testlerle güvence altına almaktır.

