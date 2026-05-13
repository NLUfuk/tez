# Tez Posteri — İçerik Taslağı ve Çıktı Rehberi

Bu dosya poster panelleri için metin iskeleti, tez amacına göre hangi çıktının nerede olduğu ve hangi script ile hangi grafiğin üretileceği bilgisini bir arada toplar. Teknik gerçek kaynağı kod ve `README.md` / `PROJE_OZETI.md` ile uyumludur.

---

## 1. Tez amacı (özet ve odak)

**Bizim için en önemli odak:** Conway tabanlı *compute-gating* (canlı hücre ve komşu sayısı üzerinden nöronlara akım taşıma) ile **Izhikevich** dinamiklerini birleştiren hibrit sistemin;

1. **Verimlilik / hesaplama yükü** açısından anlamlı bir kazanım veya düzenlenebilir bir maliyet–davranış profili sunup sunmadığı,
2. **Dinamik kararlılık** ve tekrarlanabilir ölçümle izlenebilir davranış,
3. Bu ikisinin **ölçülebilir metrikler** (ör. `metrics.csv`, verim skoru bileşenleri, ablation koşulları) ile karşılaştırılabilir şekilde raporlanması

üzerinden **sistematik olarak değerlendirilmesi**.

Önerilen resmi amaç cümlesi (`PROJE_OZETI.md` ile uyumlu):

> Conway tabanlı compute-gating yaklaşımını Izhikevich nöron dinamikleriyle birleştirerek, hibrit ağlarda verimlilik–kararlılık dengesini ölçülebilir metriklerle inceleyen ve gerektiğinde optimize eden deneysel bir simülasyon altyapısı geliştirmek ve sonuçları ablation, karşılaştırma ve (isteğe bağlı) verim tahmini modelleriyle desteklemek.

**Posterde vurgulanacak “neden önemli”:** Tek bir görsel CA değil; gating sinyali olarak Conway, sürekli zamanda Izhikevich; isteğe bağlı geri besleme ve genişletilmiş eşleme (oyun teorisi / çoklu topoloji) ile **hibrit dinamik sistem** iddiası.

---

## 2. Materyal ve yöntem

| Bileşen | Ne | Nerede (kod) |
|--------|-----|----------------|
| Conway B3/S23 | Izgara durumu, komşu sayımı | `src/conway_izh/conway.py` |
| Izhikevich | Vektörize nöron adımı, spike | `src/conway_izh/izhikevich.py` |
| Eşleme (coupling) | GoL → akım; opsiyonel spike → GoL | `src/conway_izh/coupling.py` |
| Ana döngü (ızgara) | Orkestrasyon | `src/conway_izh/grid.py` |
| Graf tabanlı çalışma | Seyrek graf, SWC birleştirme | `src/conway_izh/graph_grid.py`, `topology_manager.py`, `swc_loader.py` |
| Metrik ve görselleştirme | PNG/GIF/CSV | `src/conway_izh/metrics.py`, `viz.py` |
| Verim / veri şeması | Tez verim skoru ve veri üretimi | `src/conway_izh/efficiency.py`, `dataset_schema.py` |
| Genişletme | Oyun teorisi eşlemesi | `src/conway_izh/game_theory_coupling.py`, `scripts/run_live_gametheory.py` |

**Yöntem özeti (poster dili):** Aynı tohum ve parametre setiyle simülasyon koşuları; geri besleme açık/kapalı (ve koşullu ablation) karşılaştırması; zaman serisi metrikleri; gerektiğinde ablation özeti ve CNN tabanlı verim tahmini hattı.

---

## 3. Tez çıktıları — amaca göre nerede?

Tüm koşular için genel kural: `python -m scripts.run_grid ...` çıktıları **`outputs/<run_id>/`** altına yazılır (`README.md`).

### 3.1 Amaç: “Hibrit davranış ve metrikler”

| Çıktı | Dosya / klasör | Nasıl üretilir |
|--------|------------------|----------------|
| Son Conway durumu | `final_gol.png` | `run_grid` |
| Membran potansiyeli haritası | `final_v.png` | `run_grid` |
| Spike raster | `spike_raster.png` | `run_grid` |
| Zaman serisi | `metrics.csv` | `run_grid` |
| Animasyon | `anim.gif`, `frames/` | `run_grid --gif` |

### 3.2 Amaç: “Sunum / poster görselleri (zaman serisi, coupling şeması, feedback karşılaştırması)”

| Çıktı | Varsayılan konum | Script |
|--------|------------------|--------|
| Zaman serisi (demo / feedback) | `outputs/presentation_plots/time_series_*.png` | `python -m scripts.create_presentation_plots` |
| Feedback karşılaştırması | `outputs/presentation_plots/comparison_feedback.png` | aynı |
| Coupling mekanizması figürü | `outputs/presentation_plots/coupling_mechanism.png` | aynı |

**Not:** `create_presentation_plots` girdi olarak varsayılan olarak `outputs/presentation_demo/` ve `outputs/presentation_feedback/` altındaki `metrics.csv` dosyalarını bekler; bu klasörleri doldurmak için önce ilgili `run_grid` koşularını bu dizinlere (`--out`) yazdırmanız gerekir. Ayrıntılı akış: `PRESENTATION_GUIDE.md`.

### 3.3 Amaç: “Bileşenleri tek tek kapatma (ablation) ve verim karşılaştırması”

| Çıktı | Konum | Ön koşul script |
|--------|--------|------------------|
| Ablation koşu dosyaları ve özet | `outputs/ablation/`, özellikle `ablation_summary.csv` | `python -m scripts.run_ablation_suite` |
| İsteğe bağlı kalibrasyon | `outputs/ablation/calibration.json` | `python -m scripts.calibrate_baselines` |

### 3.4 Amaç: “Verim tahmini (CNN) ve tez figürleri”

| Çıktı | Konum | Ön koşul |
|--------|--------|----------|
| Eğitim veri seti | `outputs/datasets/efficiency/` | `python -m scripts.generate_efficiency_dataset` |
| Model ve metrikler | `outputs/models/efficiency_cnn/` (`train_history.json`, `metrics.json`) | `python -m scripts.train_efficiency_cnn` |
| Değerlendirme | `eval` çıktıları (script parametrelerine bağlı) | `python -m scripts.eval_efficiency_cnn` |

### 3.5 Amaç: “Izhikevich tek başına vs entegre karşılaştırma”

| Çıktı | Konum | Script |
|--------|--------|--------|
| Analiz grafikleri | `outputs/comparison_analysis/` (script içi varsayılan) | `compare_izhikevich` + `python -m scripts.analyze_comparison` |

`analyze_comparison.py` girdi olarak `outputs/comparison_analysis` altında `comparison_metrics.csv` ve `comparison_summary.json` bekler.

### 3.6 Amaç: “Canlı 3B topoloji ve çoklu SWC”

- Sunucu: `python -m scripts.run_live --mode stream ...`
- Ön yüz: `web/three_visualizer/index.html` (statik sunucu ile)
- Tam yığın (Windows): `scripts/start_full_stack.ps1`
- Log: `outputs/live_logs/` (`README.md`)

### 3.7 Yan yana final durum (eski Türkçe klasör adları)

`scripts/create_side_by_side.py` sabit olarak `outputs/sunum_demo/` ve `outputs/sunum_feedback/` okur; `presentation_*` ile klasör adları farklıysa ya simülasyonları bu dizinlere yazdırın ya da scriptteki yolları hizalayın.

---

## 4. Grafik / figür — hangi script?

| Posterde kullanım | Script | Komut örneği | Üretilen dosya (varsayılan) |
|-------------------|--------|----------------|-----------------------------|
| Tez (Türkçe) ablation + CNN özeti | `scripts/plot_thesis_figures.py` | `python -m scripts.plot_thesis_figures` | `outputs/thesis_figures/ablation_efficiency.png`, `ablation_components.png`, `cnn_training_curves.png`, `cnn_vs_naive.png` |
| Sunum zaman serisi + coupling + feedback | `scripts/create_presentation_plots.py` | `python -m scripts.create_presentation_plots` | `outputs/presentation_plots/*.png` |
| Yan yana final GoL / V | `scripts/create_side_by_side.py` | `python -m scripts.create_side_by_side` | `outputs/sunum_plots/` (girdi: `outputs/sunum_demo`, `outputs/sunum_feedback`) |
| Karşılaştırma detay grafikleri | `scripts/analyze_comparison.py` | `python -m scripts.analyze_comparison` | `outputs/comparison_analysis/` içi çıktılar |
| Hiperparametre taraması | `scripts/hyperparameter_optimization.py` | `--out` ile (varsayılan `outputs/optimization`) | Optimizasyon çıktıları |
| Parametre süpürgesi | `scripts/sweep_k_syn_gamma.py` | `outputs/` alt alt klasörler | Süpürme grafikleri / özetler |
| Tek nöron referans | `scripts/run_single.py` | İstenen parametrelerle | `outputs/<run_id>/` veya script çıktısı |

**`plot_thesis_figures` için sıra önerisi:** `run_ablation_suite` → (isteğe bağlı) `calibrate_baselines` → `generate_efficiency_dataset` → `train_efficiency_cnn` → `plot_thesis_figures`.

---

## 5. Sonuçlar ve genel değerlendirme (poster metni taslağı)

- **Hibrit sistem:** Conway özellikleri, Izhikevich ağına uzamsal–topolojik bir gating sinyali sağlar; geri besleme açıldığında GoL ile nöron aktivitesi **karşılıklı** etkilenir.
- **Ölçüm:** `metrics.csv` üzerinden canlı hücre sayısı, spike sayısı, ortalama potansiyel ve firing rate zaman içinde raporlanabilir; ablation ile **Conway katkısının** kapatılıp açılmasıyla davranış eksenleri ayrıştırılır.
- **Verim ekseni:** `efficiency` veri hattı ve isteğe bağlı CNN, tez kapsamında “ölçülebilir verim / tahmin” iddiasını destekler; sonuçlar `outputs/thesis_figures/` altında tek sayfada toplanabilir.
- **Çoklu topoloji:** Stream modu ve SWC birleştirme, çalışmayı klasik ızgaranın ötesine taşıyan **mimari çıktı** olarak posterde “genişletilmiş yöntem” olarak konumlandırılabilir.

*(Rakamları posterde doldurmak için kendi koşularınızdan ortalama / std ekleyin.)*

---

## 6. Güçlü ve zayıf yönler

**Güçlü yönler**

- Açık modüler yapı (Conway, Izhikevich, coupling, grid, graf motoru ayrı).
- Tekrarlanabilir koşular (`--seed`), CSV/PNG ile izlenebilir deney çıktısı.
- Ablation + verim veri hattı + tez figür scripti ile **tez hikayesine uygun** paketlenmiş analiz.
- Canlı görselleştirme ve çoklu topoloji ile **demonstrasyon ve iletişim** gücü.

**Zayıf yönler / sınırlamalar** (`PROJE_OZETI.md` ile uyumlu)

- Oyun teorisi / yoğun Python döngüleri: performans ve deterministiklik için ek mühendislik gerekebilir.
- Web demosu ile çekirdek Python modeli bire bir aynı değil; iddialarda “hangi ortam” net ayrılmalı.
- Parametre uzayı geniş; tek koşu yerine **çok tohum + özet tablo** posterde zayıflığı dengelemek için önemli.

---

## 7. İleri çalışmalar

- Oyun teorisi eşlemesinin vektörizasyonu ve seed kontrollü RNG ile deterministik testler.
- Graf modunda daha fazla biyolojik plausibility metriği ve istatistiksel güç analizi (daha fazla tohum).
- Çekirdek simülasyon ile web görselleştirici arasında davranış eşlemesi dokümantasyonu.
- Gerçek donanım / enerji ölçümü ile “verim” kavramının dış doğrulaması (kapsam genişlemesi).

---

## 8. Poster panel önerisi (kısa)

1. **Başlık + amaç** (Bölüm 1’deki odak cümlesi).
2. **Şema:** `coupling_mechanism.png` veya mimari kutu diyagramı (`README.md` yapısı).
3. **Sonuç:** `time_series_*.png` ve/veya `comparison_feedback.png`.
4. **Tez derinliği:** `outputs/thesis_figures/` ablation + CNN (varsa).
5. **3B / topoloji:** ekran görüntüsü veya `run_live` stream.
6. **Sonuç + ileri çalışma** (bir madde maddesi).

---

## 9. Hızlı komut hatırlatıcısı

```powershell
$env:PYTHONPATH="C:\Users\ufukf\OneDrive\Desktop\tez\src"

# Temel poster görselleri (önce demo/feedback klasörlerini doldurun)
python -m scripts.create_presentation_plots

# Ablation + model sonrası tez figürleri
python -m scripts.run_ablation_suite
python -m scripts.plot_thesis_figures
```

---

*Bu dosya poster metninin taslak kaynağıdır; jüri formatına göre madde başlıklarını kısaltıp rakamları kendi deney tablolarınızla güncelleyin.*
