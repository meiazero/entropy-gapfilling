# Análise quantitativa e espacial da relação entre entropia local e desempenho de métodos clássicos de interpolação de lacunas em imagens de satélite.

## Objetivo:

Avaliar e comparar procedimentos de processamento de imagem para preenchimento de lacunas em imagens de satélite e quantificar, por meio de métricas objetivas e análise especial com a correlação entre entropia calculada em janelas locais e a perda de qualidade dos métodos, identificando regiões com maior deficiência dos algoritmos convencionais.

## Hipóteses

- H1 — Regiões com maior entropia local (mais textura/complexidade) apresentarão pior desempenho médio de interpoladores clássicos.
- H2 — Métodos baseados em modelos espectro-temporais e geostáticos terão desempenho superior em áreas de baixa entropia.
- H3 — Existe relação estatisticamente significante entre métricas de qualidade locais e entropia local.

## Técnicas

1. Interpoladores espaciais clássicos (nearest neighbor, bilinear, bicubic, Lanczos).
2. Métodos baseados em kernel (IDW, RBF, splines).
3. Métodos geostáticos (kriging).
4. Métodos de transformadas/regularização (DCT, wavelets, TV inpainting).
5. Métodos patch-based (exemplar-based, non-local).
6. Métodos de compressive sensing (L1 em wavelet/DCT).
7. Baselines de aprendizado profundo (AE, VAE, GAN, ViT) - avaliados separadamente.

## Base de Dados

Uso de Sentinel-2, Landsat-8/9 e MODIS. Simulação de lacunas (máscaras de nuvem realistas, patches de diferentes tamanhos, níveis de ruído). Validação com cenas limpas temporais.

## Pipeline

1. Coleta e pré-processamento.
2. Segmentação em janelas.
3. Cálculo de entropia local.
4. Aplicação dos métodos.
5. Avaliação (PSNR, SSIM, RMSE, SAM, ERGAS).
6. Análise estatística e espacial.
7. Visualização (mapas, gráficos, tabelas).
8. Relatório e recomendações.

## Métricas e validação estatística

**Métricas:** PSNR, SSIM, RMSE, IoU, SAM, ERGAS.

**Análises:** correlação (Pearson, Spearman), regressão robusta, ANOVA/Kruskal-Wallis, testes pareados, bootstrap IC95%, Moran’s I, LISA.

## Parametrizações

- Janelas de entropia: 7x7, 15x15, 31x31.
- Patches 64x64 com sobreposição.
- N mínimo de patches: 500–2000.
- Níveis de ruído: ∞, 40 dB, 30 dB, 20 dB.
- Repetições: 10 seeds aleatórias.
- α=0.05 com correção múltipla (FDR).

## Analisar possíveis limitações

- Disponibilidade de ground truth.
- Sensibilidade do Kriging.
- Custo computacional de métodos patch-based.

## Resultados

### Figuras

- **Fig 1:** Mapas de entropia local em 3 escalas (7x7, 15x15, 31x31), TODOS os sensores.
- **Fig 2:** Scatterplots entropia vs. PSNR por metodo (15 metodos classicos) incluindo coeficientes (beta), p-values, R-squared, R-squared ajustado, além da reta de regressão.
- **Fig 3:** Boxplots PSNR por metodo, agrupados por bin de entropia (low/medium/high).
- **Fig 4:** Boxplots PSNR por metodo, agrupados por nivel de ruido (inf, 40, 30, 20 dB).
- **Fig 5:** Mapas de clusters LISA sobrepostos em mapas de erro de reconstrucao.
- **Fig 6:** Exemplos visuais: clean / degraded / todos os metodos para patches organizados de baixa e alta entropia.
- **Fig 7:** Heatmap de correlacao (Spearman rho) entre entropia multi-escala e metricas.
- **Fig 8:** Mapas de PSNR local e SSIM local para patch representativo (todos os metodos).

### Tabelas

- **Tabela 1:** Visao geral dos 15 metodos classicos (categoria, parametros, complexidade).
- **Tabela 2:** Media PSNR +/- IC95% por metodo x nivel de ruido (bold-best, underline-second).
- **Tabela 3:** Media PSNR +/- IC95% estratificada por tercil de entropia em cada escala (7x7, 15x15, 31x31).
- **Tabela 4:** Correlacao Spearman entre entropia (multi-escala) e metricas (PSNR, SSIM, RMSE, SAM) com p-values FDR-corrigidos.
- **Tabela 5:** Teste Kruskal-Wallis (H, p, epsilon-squared) e pares significativos (Dunn post-hoc com Cliff's delta).
- **Tabela 6:** Regressao robusta (RLM/HuberT) por metrica: coeficientes (beta), p-values, R-squared, R-squared ajustado, VIF.
- **Tabela 7:** Media PSNR +/- IC95% por metodo x sensor satelite (Sentinel-2, Landsat-8, Landsat-9, MODIS).
- **Tabela 8:** Comparacao classicos vs. DL (PSNR, SSIM, RMSE side-by-side).

### Analises estatisticas

- Correlacao Pearson e Spearman com correcao FDR (alpha=0.05) para ambos.
- Kruskal-Wallis + Dunn post-hoc com Bonferroni, incluindo epsilon-squared e Cliff's delta.
- Regressao robusta: metrica ~ entropia (multi-escala) + metodo + ruido. Coeficientes, p-values, IC95%, R-squared ajustado, VIF.
- Moran's I global e LISA local para autocorrelacao espacial de mapas de erro.

## Deep Learning — Treinamento e Avaliação

### Pré-processamento

Converte patches GeoTIFF para NPY. Gera `preprocessed/manifest.csv` com splits `train`, `val` e `test`. Resume-safe — se o manifest já estiver completo, pula automaticamente.

```bash
uv run python scripts/preprocess_dataset.py --resume
```

### Quick-start (via Make)

Os targets do `make` usam defaults rápidos (2 epochs) para validação:

```bash
make dl-train-vae        # treina VAE com 2 epochs
make dl-train-all        # treina todos os 5 modelos sequencialmente
make dl-eval-all         # avalia todos no test split
```

Para alterar epochs via make: `make dl-train-ae AE_EPOCHS=50`.

### Treinamento (produção)

Comandos completos com hiperparâmetros de produção:

```bash
# AE — Autoencoder
uv run python -m dl_models.ae.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output dl_models/checkpoints/ae_best.pth \
    --epochs 50 --batch-size 32 --lr 1e-3 --patience 10

# VAE — Variational Autoencoder
uv run python -m dl_models.vae.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output dl_models/checkpoints/vae_best.pth \
    --epochs 60 --batch-size 32 --lr 1e-3 --beta 0.001 --patience 10

# GAN — UNet Generator + PatchGAN Discriminator
uv run python -m dl_models.gan.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output dl_models/checkpoints/gan_best.pth \
    --epochs 100 --batch-size 16 --lr 2e-4 \
    --lambda-l1 10.0 --lambda-adv 0.1 --patience 15

# U-Net (PyTorch)
uv run python -m dl_models.unet.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output dl_models/checkpoints/unet_best.pth \
    --epochs 60 --batch-size 32 --lr 1e-3 \
    --weight-decay 1e-4 --patience 12

# ViT (MAE-style)
uv run python -m dl_models.vit.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output dl_models/checkpoints/vit_best.pth \
    --epochs 100 --batch-size 32 --lr 1e-4 \
    --weight-decay 0.05 --patience 15

```

### Avaliação

```bash
# Exemplo: avaliar U-Net no test split
uv run python -m dl_models.evaluate \
    --model unet \
    --checkpoint dl_models/checkpoints/unet_best.pth \
    --manifest preprocessed/manifest.csv \
    --satellite sentinel2 \
    --output results/dl_eval

# Modelos disponíveis: ae, vae, gan, unet, vit
```

## Cluster (Slurm) Quickstart

This project requires Python >= 3.12. The cluster configuration below has
been validated on the target cluster (Rocky Linux 9.5, NVIDIA A100 80GB PCIe,
NVIDIA driver 550.90.07 / CUDA 12.4, CUDA toolkit 12.6.2).

Confirmed values:

| Resource | Value |
|----------|-------|
| Partition (GPU) | `gpuq` |
| Partition (CPU) | `cpuq` |
| GRES | `gpu:a100:1` |
| Conda module | `miniconda3/py312_25.1.1` |
| CUDA module | `cuda/12.6.2` |

If submitting to a different cluster, use `hello_cuda.sbatch` to rediscover
the correct values (see "Diagnostics" section below).

### 1) Bootstrap the full environment (one time)

`setup_env.sh` creates the `pdi312` conda environment, installs all
dependencies, and installs a CUDA-enabled PyTorch wheel (torch 2.3.1+cu121,
compatible with the driver's CUDA 12.4 runtime). It also sets
`PYTHONNOUSERSITE=1` to prevent `~/.local` packages from interfering.

```bash
bash scripts/slurm/setup_env.sh /path/to/pdi_models_v5
```

Verify that the final output shows `CUDA available: True` and lists the
expected GPU (`GPU 0: NVIDIA A100 80GB PCIe  (capability 8.0)`).

> **Note on user-local pip packages**: if a different torch version exists in
> `~/.local/lib/python3.12/site-packages` (e.g. from a previous install), it
> will be shadowed by the conda env as long as `PYTHONNOUSERSITE=1` is set.
> All job scripts in `scripts/slurm/` already export this variable.

### 2) Set dataset location

```bash
export PDI_DATA_ROOT=/path/to/dataset
```

### 3) Preprocess once (shared for all workflows)

Use the Slurm job (recommended) or run the script directly:

```bash
sbatch scripts/slurm/preprocess.sbatch
```

```bash
python scripts/preprocess_dataset.py --resume
```

### 4) Train all DL models on GPU (Slurm)

```bash
sbatch scripts/slurm/train_all.sbatch
```

The job runs with `python` directly and does not require `make`.

### 5) Monitor jobs and inspect output

```bash
squeue -u $USER
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS

cat slurm_<job-name>_<JOB_ID>.out
cat slurm_<job-name>_<JOB_ID>.err
```

### Diagnostics (new cluster or broken environment)

`hello_cuda.sbatch` is a diagnostic-only job that collects environment info
without aborting on errors. Run it to rediscover module names, partition
labels, GPU availability, and CUDA state:

```bash
# Verify partition name first:
sinfo -o "%P %a %G"

# Submit with confirmed values, or override at submission time:
sbatch scripts/slurm/hello_cuda.sbatch
sbatch -p <partition> --gres=gpu:1 scripts/slurm/hello_cuda.sbatch
```

The output file will contain sections for available modules, loaded modules,
nvidia-smi GPU info, and PyTorch CUDA status.
