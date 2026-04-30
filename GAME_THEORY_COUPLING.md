# Game Theory Based Spike Generation

## Genel Bakış

Yeni spike mekanizması Conway Game of Life prensiplerine ve oyun teorisine dayalıdır. Spike'lar artık rastgele değil, Conway kurallarına göre oluşur ve hücreler arasında bağlantı kurar.

## Temel Prensipler

### 1. Conway Kurallarına Dayalı Spike Tetikleme

**B3/S23 Kuralları:**
- **Birth (B3)**: Ölü hücreler tam olarak 3 komşuya sahipse → Yüksek spike olasılığı (0.8)
  - Hücreler "doğmak" için işbirliği yapar
  
- **Survival (S23)**: Canlı hücreler 2-3 komşuya sahipse → Orta spike olasılığı (0.6)
  - Topluluğu korumak için işbirliği
  
- **Overcrowding**: Canlı hücreler >3 komşuya sahipse → Düşük spike olasılığı (0.2)
  - Rekabet nedeniyle işbirliği azalır
  
- **Isolation**: Canlı hücreler <2 komşuya sahipse → Çok düşük spike olasılığı (0.1)
  - Yalnızlık nedeniyle aktivite düşer

### 2. Spike Propagation (Yayılma)

Spike'lar komşu hücrelere yayılır:
- Spike olan hücre, 8 komşusuna sinyal gönderir
- Canlı komşular daha güçlü sinyal alır (işbirliği)
- Ölü komşular daha zayıf sinyal alır
- Cascade effect: Bir spike diğer spike'ları tetikleyebilir

### 3. Oyun Teorisi Karar Verme

Spike kararı üç faktöre bağlıdır:
1. **Conway-based probability**: Conway durumuna göre spike olasılığı
2. **Membrane potential**: Nöronun membran potansiyeli
3. **Propagation influence**: Komşulardan gelen spike sinyalleri

## Kullanım

### Temel Kullanım

```bash
python -m scripts.run_grid --height 60 --width 60 --steps 300 --game-theory
```

### Feedback ile

```bash
python -m scripts.run_grid --height 60 --width 60 --steps 300 --game-theory --feedback
```

### Parametreleri Ayarlama

```bash
python -m scripts.run_grid \
  --height 60 --width 60 --steps 300 \
  --game-theory \
  --propagation-strength 0.7 \
  --cooperation-factor 0.4 \
  --cooperation-strength 0.8
```

## Parametreler

- `--game-theory`: Game theory modunu aktifleştirir
- `--propagation-strength`: Spike yayılma gücü (0-1, varsayılan: 0.5)
  - Yüksek değer = daha güçlü yayılma
- `--cooperation-factor`: İşbirliği faktörü (varsayılan: 0.3)
  - Yayılma sinyalinin spike kararına etkisi
- `--cooperation-strength`: Feedback işbirliği gücü (varsayılan: 0.7)
  - Spike'ların Conway'e geri besleme gücü

## Karşılaştırma: Normal vs Game Theory

### Normal Mode
- Spike'lar sadece membran potansiyeli threshold'una göre oluşur
- Conway durumu sadece input current'ı etkiler
- Spike sayısı genellikle düşük (0-10)

### Game Theory Mode
- Spike'lar Conway kurallarına göre oluşur
- Spike propagation hücreler arası bağlantı kurar
- Spike sayısı çok daha yüksek (yüzlerce-binlerce)
- Daha dinamik ve işbirlikçi sistem

## Örnek Sonuçlar

**Normal Mode (50x50 grid, 100 steps):**
- Step 0: alive=818, spikes=0
- Step 50: alive=261, spikes=0

**Game Theory Mode (50x50 grid, 100 steps):**
- Step 0: alive=818, spikes=792
- Step 50: alive=261, spikes=1547

## Teknik Detaylar

### Spike Probability Hesaplama

```python
# Birth potential (B3)
if dead_cell and neighbors == 3:
    spike_prob = 0.8
    I = k_alive * 3.0 + bias

# Survival activity (S23)
if alive_cell and neighbors in [2, 3]:
    spike_prob = 0.6
    I = k_alive * 2.5 + k_neighbors * neighbors + bias

# Overcrowding
if alive_cell and neighbors > 3:
    spike_prob = 0.2
    I = k_alive * 1.0 + k_neighbors * neighbors * 0.5 + bias

# Isolation
if alive_cell and neighbors < 2:
    spike_prob = 0.1
    I = k_alive * 0.5 + bias
```

### Spike Propagation

```python
# Her spike için komşulara sinyal gönder
for each spiking_cell:
    for each neighbor (8-neighborhood):
        if neighbor is alive:
            propagation[neighbor] += strength * 1.5  # Güçlü sinyal
        else:
            propagation[neighbor] += strength * 0.8  # Zayıf sinyal
```

### Final Spike Decision

```python
# Kombine olasılık
combined_prob = conway_prob + cooperation_factor * propagation

# Stokastik karar
if random() < combined_prob:
    spike = True

# Veya membran potansiyeli yeterince yüksekse
if v_normalized > 0.7:
    spike = True
```

## Avantajlar

1. **Deterministik Davranış**: Conway kurallarına dayalı, rastgele değil
2. **Hücreler Arası Bağlantı**: Spike propagation ile network oluşur
3. **Oyun Teorisi**: İşbirliği ve rekabet dinamikleri
4. **Daha Fazla Aktivite**: Spike sayısı önemli ölçüde artar
5. **Emergent Davranış**: Sistem daha karmaşık ve ilginç davranışlar sergiler

## Gelecek Geliştirmeler

- [ ] Farklı oyun teorisi stratejileri (Tit-for-Tat, Pavlov, vb.)
- [ ] Uzun mesafe bağlantılar
- [ ] Spike timing dependent plasticity (STDP)
- [ ] Adaptive coupling strength
- [ ] Multi-layer networks

