# Tez Projesi Kullanim Rehberi

Bu rehber, projeyi bastan sona calistirip tez bulgularini tekrar uretebilmen icin hazirlandi.

## 1) Ortam Kurulumu

- Python 3.12+ onerilir.
- Proje kok dizininde:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 2) Hizli Dogrulama

```powershell
$env:PYTHONPATH="C:\Users\ufukf\OneDrive\Desktop\tez\src"
python -m pytest tests/test_grid_smoke.py tests/test_izhikevich.py
```

Beklenen: tum testlerin gecmesi.

## 3) Ana Simulasyon (Game Theory + Memory + Strateji)

```powershell
$env:PYTHONPATH="C:\Users\ufukf\OneDrive\Desktop\tez\src"
python -m scripts.run_grid --height 60 --width 60 --steps 300 --seed 42 --game-theory
```

Feedback acik:

```powershell
python -m scripts.run_grid --height 60 --width 60 --steps 300 --seed 42 --game-theory --feedback
```

Ciktilar `outputs/<run_id>` altina yazilir.

## 4) Gorsel Bulgulari Uretme

Demo ve feedback kosularini birlikte al:

```powershell
python -m scripts.run_grid --height 60 --width 60 --steps 300 --seed 42 --game-theory --out outputs --run-id presentation_demo
python -m scripts.run_grid --height 60 --width 60 --steps 300 --seed 42 --game-theory --feedback --out outputs --run-id presentation_feedback
python -m scripts.create_presentation_plots --demo-dir outputs/presentation_demo --feedback-dir outputs/presentation_feedback --out outputs/presentation_plots_latest
```

Olusan kritik gorseller:
- `outputs/presentation_plots_latest/time_series_demo.png`
- `outputs/presentation_plots_latest/time_series_feedback.png`
- `outputs/presentation_plots_latest/comparison_feedback.png`
- `outputs/presentation_plots_latest/coupling_mechanism.png`

## 5) Ablation Deneyleri (Bilimsel Dogrulama)

### 5.1 Baseline kalibrasyon

```powershell
python -m scripts.calibrate_baselines --height 32 --width 32 --steps 120 --seeds 2 --out outputs/ablation/calibration.json
```

### 5.2 Full ablation

```powershell
python -m scripts.run_ablation_suite --height 48 --width 48 --steps 250 --seeds 5 --out outputs/ablation_full --calibration-json outputs/ablation/calibration.json
```

### 5.3 Tez figurlari

```powershell
python -m scripts.plot_thesis_figures --ablation-dir outputs/ablation_full --model-dir outputs/models/efficiency_cnn_full --out outputs/thesis_figures_full_final
```

## 6) Veri Uretimi ve CNN Egitimi

### 6.1 Full dataset

```powershell
python -m scripts.generate_efficiency_dataset --rollouts 30 --steps 220 --window 10 --height 48 --width 48 --out outputs/datasets/efficiency_full
```

### 6.2 Egitim

```powershell
python -m scripts.train_efficiency_cnn --data-dir outputs/datasets/efficiency_full --out outputs/models/efficiency_cnn_full --epochs 35 --batch-size 64 --patience 8 --lr 0.001
```

### 6.3 Degerlendirme

```powershell
python -m scripts.eval_efficiency_cnn --data-dir outputs/datasets/efficiency_full --model-dir outputs/models/efficiency_cnn_full --split test
```

## 7) Beklenen Basari Kriterleri

- Ablation'da `full_model`, klasik baseline'lara gore daha yuksek `efficiency_score` verir.
- CNN test performansi naive baseline'dan belirgin iyi olmalidir (MAE/RMSE).
- Gorsellerde feedback acik durumda asiri aktivite ve maliyet artisinin verim skoruna etkisi gorulmelidir.

## 8) Sik Karsilasilan Sorunlar

- `ModuleNotFoundError: conway_izh`
  - `PYTHONPATH` degiskenini `src` klasorunu gosterecek sekilde ayarla.
- `gh` komutu bulunamiyor
  - `C:\Program Files\GitHub CLI\gh.exe` tam yolunu kullan.
- CNN sonuclari zayif
  - Daha buyuk dataset ve daha uzun egitim dene; split'in rollout bazli kaldigindan emin ol.

## 9) Guvenli Yedekleme

- Repo: `https://github.com/NLUfuk/tez.git`
- Push icin:

```powershell
git add .
git commit -m "Update thesis experiments"
git push
```
