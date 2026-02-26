# Analise quantitativa e espacial da entropia local e metodos classicos de preenchimento de lacunas em imagens de satelite

Pipeline de pesquisa reproduzivel que avalia 15 metodos classicos de interpolacao e 5 baselines de aprendizado profundo para preenchimento de lacunas em imagens de satelite, com analise espacial e estatistica da relacao entre entropia local e qualidade de reconstrucao.

## Hipoteses de Pesquisa

- **H1** - Regioes com maior entropia local (mais textura/complexidade) apresentarao pior desempenho medio dos interpoladores classicos.
- **H2** - Metodos espectro-temporais e geostatisticos terao desempenho superior em areas de baixa entropia.
- **H3** - Existe relacao estatisticamente significante entre metricas de qualidade locais e entropia local.

## Metodos

**Classicos (15):**

| Categoria                    | Metodos                              |
| ---------------------------- | ------------------------------------ |
| Espacial                     | nearest, bilinear, bicubic, Lanczos  |
| Kernel                       | IDW, RBF (thin-plate spline), spline |
| Geostatistico                | kriging                              |
| Transformada / regularizacao | DCT, wavelet, TV inpainting          |
| Compressive sensing          | L1-DCT, L1-wavelet                   |
| Patch-based                  | non-local means, exemplar-based      |

**Baselines de aprendizado profundo (5):** AE, VAE, GAN (gerador UNet + discriminador PatchGAN), U-Net, ViT (estilo MAE).

## Base de Dados

77.916 patches (64x64 px, 4 bandas) extraidos de Sentinel-2, Landsat-8, Landsat-9 e MODIS. Lacunas sinteticas geradas a partir de mascaras de nuvem realistas em quatro niveis de SNR (inf, 40, 30, 20 dB). Veja [DATASET.md](DATASET.md) para detalhes completos sobre colunas, splits e layout de saida do pre-processamento.

| Satelite   | Patches |
| ---------- | ------- |
| Sentinel-2 | 73.984  |
| Landsat-9  | 1.936   |
| Landsat-8  | 1.936   |
| MODIS      | 60      |

Divisao: 52.842 treino / 10.225 validacao / 14.849 teste.

## Estrutura do Projeto

```
.
├── config/
│   ├── paper_results.yaml      # Execucao completa de producao (10 seeds, 75 patches/config)
│   └── quick_validation.yaml   # Execucao de smoke-test (1 seed, 1 patch, 1 epoch)
├── dl_models/
│   ├── ae/                     # Autoencoder (model + train)
│   ├── vae/                    # Variational Autoencoder
│   ├── gan/                    # GAN (gerador UNet + discriminador PatchGAN)
│   ├── unet/                   # U-Net (skip connections + blocos residuais)
│   ├── vit/                    # Vision Transformer estilo MAE
│   ├── shared/                 # Utilitarios compartilhados: base, dataset, metricas, trainer, visualizacao
│   ├── evaluate.py             # Script de avaliacao unificado para todos os modelos DL
│   └── plot_training.py        # Plota curvas de treinamento a partir de arquivos JSON de historico
├── docs/
│   ├── main.tex                # Fonte LaTeX do artigo
│   ├── figures/                # Figuras copiadas por make paper-assets
│   ├── tables/                 # Tabelas copiadas por make paper-assets
│   └── dist/                   # Saida PDF compilada
├── paper_assets/
│   ├── classic/                # Figuras, tabelas, resultados brutos e logs do experimento classico
│   └── dl/                     # Resultados de avaliacao, plots de treinamento e historico do experimento DL
├── results/
│   ├── paper_results/          # Saida do experimento classico completo do artigo
│   └── quick_validation/       # Saida da execucao de validacao rapida
├── scripts/
│   ├── preprocess_dataset.py   # Converte patches GeoTIFF para NPY + gera manifest
│   ├── run_experiment.py       # Executador principal do experimento de metodos classicos
│   ├── aggregate_results.py    # Agrega CSVs brutos (classicos + DL) em CSVs prontos para analise
│   ├── generate_figures.py     # Gera todas as figuras do artigo a partir dos resultados
│   ├── generate_tables.py      # Gera todas as tabelas LaTeX a partir dos resultados
│   ├── precompute_entropy.py   # Pre-computa mapas de entropia para todos os patches
│   ├── pack_paper_assets.sh    # Empacota assets do artigo em paper_assets.zip
│   ├── grid_search_classical.py
│   ├── grid_search_dl.py
│   └── slurm/                  # Scripts de submissao SLURM (ver secao Cluster)
├── src/pdi_pipeline/
│   ├── methods/                # Implementacoes dos 15 metodos classicos de preenchimento
│   ├── metrics.py              # PSNR, SSIM, RMSE, SAM, ERGAS, pixel_acc, F1, RMSE por banda
│   ├── statistics.py           # Pearson/Spearman, Kruskal-Wallis, regressao robusta, Moran's I
│   ├── entropy.py              # Calculo de entropia local (janelas multi-escala)
│   ├── aggregation.py          # Utilitarios de agregacao de resultados
│   ├── dataset.py              # Carregamento de dataset e manifest
│   ├── config.py               # Parsing de configuracao YAML
│   ├── logging_utils.py        # Logger StreamProgress (substitui tqdm)
│   └── experiment_artifacts.py # Salvamento de artefatos e serializacao de resultados
├── tests/
│   ├── unit/                   # Testes unitarios para modelos DL e pipeline
│   └── integration/            # Testes de integracao
├── graphs-examples/            # Exemplos de estilo matplotlib e scripts de plot
├── images/                     # Estilo matplotlib compartilhado
├── Makefile                    # Todos os targets de workflow (ver abaixo)
├── pyproject.toml
├── DATASET.md
└── pdi.md (README.md)
```

## Configuracao do Ambiente

Requer Python 3.12. Usa `uv` como gerenciador de pacotes.

```bash
uv sync
uv run pre-commit install
```

Verificacoes de qualidade de codigo:

```bash
make check    # verificacao do lock file + pre-commit + deptry
make test     # pytest com cobertura
```

## Configuracao

Todos os parametros do experimento sao definidos em arquivos YAML em `config/`. Selecione uma configuracao com a variavel de ambiente `PDI_CONFIG` (padrao: `config/paper_results.yaml`):

```bash
# Usar a configuracao completa do artigo (padrao)
uv run python scripts/run_experiment.py

# Usar a configuracao de validacao rapida
PDI_CONFIG=config/quick_validation.yaml uv run python scripts/run_experiment.py --quick
```

Parametros principais de um arquivo de configuracao:

```yaml
experiment:
  seeds: [42, 123, ...] # Seeds aleatorias para reproducibilidade
  noise_levels: ["inf", "40", "30", "20"]
  satellites: ["sentinel2", "landsat8", "landsat9", "modis"]
  entropy_windows: [7, 15, 31] # Tamanhos de janela para entropia local (pixels)
  max_patches: 75 # Maximo de patches por configuracao

dl_training:
  satellite: sentinel2
  device: cuda
  eval_after_train: true # roda evaluate.py automaticamente apos treinamento
  noise_levels: ["inf", "40", "30", "20"]
  models:
    ae: { epochs: 60, batch_size: 96, lr: 1.5e-3, patience: 12 }
    vae: { epochs: 80, batch_size: 96, lr: 1.5e-3, beta: 0.001, patience: 15 }
    gan:
      {
        epochs: 120,
        batch_size: 32,
        lr: 2e-4,
        lambda_l1: 10.0,
        lambda_adv: 0.1,
        patience: 20
      }
    unet:
      {
        epochs: 80,
        batch_size: 96,
        lr: 1.5e-3,
        weight_decay: 1e-4,
        patience: 15
      }
    vit:
      {
        epochs: 120,
        batch_size: 64,
        lr: 2e-4,
        weight_decay: 0.05,
        patience: 20
      }

entropy_filter:
  window: 7
  train_buckets: [high] # treina apenas em patches de alta complexidade
  quantiles: [0.33, 0.67]
  eval_scenarios: # avalia em 3 cenarios de cobertura de entropia
    - { name: entropy_high, buckets: [high] }
    - { name: entropy_medium_high, buckets: [medium, high] }
    - { name: entropy_all, buckets: [low, medium, high] }
```

## Quick Start (Make)

```bash
# Pre-processar dataset (necessario uma vez; seguro re-executar)
make preprocess-all

# Smoke test rapido: apenas metodos classicos (1 seed, 1 patch)
make experiment-quick

# Preview completo: classico + todos os 5 modelos DL + figuras unificadas
make preview

# Experimento completo do artigo (producao)
make experiment
```

## Pipeline do Experimento (Metodos Classicos)

### 1. Pre-processar dataset

Converte patches GeoTIFF para NPY e produz `preprocessed/manifest.csv`:

```bash
uv run python scripts/preprocess_dataset.py --resume
# ou: make preprocess-all
```

### 2. Pre-computar entropia (opcional, melhora a velocidade do experimento)

```bash
uv run python scripts/precompute_entropy.py
```

### 3. Executar experimento

```bash
uv run python scripts/run_experiment.py --config config/paper_results.yaml
```

Saida: `results/paper_results/raw_results.csv`

Colunas do CSV classico: `seed, noise_level, method, method_category, patch_id, satellite, gap_fraction, status, error_msg, elapsed_s, entropy_7, entropy_15, entropy_31, psnr, ssim, rmse, sam, ergas, pixel_acc_002, pixel_acc_005, pixel_acc_01, f1_002, f1_005, f1_01, rmse_b0, rmse_b1, rmse_b2, rmse_b3`

### 4. Agregar resultados

```bash
uv run python scripts/aggregate_results.py \
    --results results/paper_results \
    --output results/paper_results/aggregated
```

Para incluir resultados DL apos os treinamentos completarem (ver nota sobre nomenclatura abaixo):

```bash
uv run python scripts/aggregate_results.py \
    --results results/paper_results \
    --dl-eval results/dl_models/eval/entropy_all \
    --dl-history results/dl_models/checkpoints \
    --output results/paper_results/aggregated
```

> **Nota de nomenclatura DL:** `evaluate.py` grava arquivos como `{model}_{noise_label}.csv`
> (ex: `ae_gap_only.csv`, `ae_snr40dB.csv`), enquanto `aggregate_results.py --dl-eval`
> espera `eval_{noise}.csv` (ex: `eval_inf.csv`). Se o aggregator nao encontrar os arquivos,
> as figuras e tabelas DL ainda sao geradas corretamente via `--dl-results results/dl_models`
> nos passos abaixo - isso cobre a tabela comparativa classicos vs DL.

### 5. Gerar figuras e tabelas

```bash
uv run python scripts/generate_figures.py \
    --results results/paper_results \
    --aggregated-dir results/paper_results/aggregated

uv run python scripts/generate_tables.py \
    --results results/paper_results \
    --aggregated-dir results/paper_results/aggregated \
    --dl-results results/dl_models
```

Adicione `--png-only` para pular a saida PDF (mais rapido, util para previews).

### 6. Empacotar assets do artigo

```bash
bash scripts/pack_paper_assets.sh
# Produz o diretorio paper_assets/ e paper_assets.zip
```

## Pipeline de Aprendizado Profundo

### Treinamento + Avaliacao automatica

Com `eval_after_train: true` no YAML, `train_model.py` executa `evaluate.py` automaticamente
ao fim de cada treinamento, para todos os `noise_levels` e todos os `eval_scenarios` definidos
na config. Nao e necessario rodar `evaluate.py` manualmente.

```bash
make dl-train-ae     # Autoencoder
make dl-train-vae    # Variational Autoencoder
make dl-train-gan    # GAN
make dl-train-unet   # U-Net
make dl-train-vit    # ViT estilo MAE
make dl-train-all    # Todos os 5 modelos sequencialmente
```

Sobrescrever hiperparametros ou satelite pela linha de comando:

```bash
make dl-train-ae SATELLITE=landsat8 AE_EPOCHS=100 DEVICE=cuda:1
```

### Saida do evaluate.py (schema completo)

Os CSVs de avaliacao DL produzidos pelo `evaluate.py` atual contem o schema completo:

```
results/dl_models/eval/{scenario_name}/{model}/{model}_{noise_label}.csv
```

Exemplo: `results/dl_models/eval/entropy_all/unet/unet_gap_only.csv`

Colunas: `model, architecture, satellite, noise_level, patch_id, gap_fraction, entropy_7,
entropy_15, entropy_31, status, error_msg, elapsed_s, psnr, ssim, rmse, sam, ergas,
rmse_b0, rmse_b1, rmse_b2, rmse_b3, pixel_acc_002, pixel_acc_005, pixel_acc_01,
f1_002, f1_005, f1_01`

> As metricas do test set (eval CSV) sao a fonte autoritativa para reportar performance
> dos modelos DL no artigo. Os valores de `*_history.json` sao do val set durante
> treinamento e servem apenas para as curvas de convergencia.

### Avaliacao manual (opcional, para noise levels ou satelites adicionais)

```bash
uv run python -m dl_models.evaluate \
    --model unet \
    --checkpoint results/dl_models/checkpoints/unet_best.pth \
    --manifest preprocessed/manifest.csv \
    --satellite sentinel2 \
    --noise-level inf \
    --output results/dl_models/eval/entropy_all
```

Modelos disponiveis: `ae`, `vae`, `gan`, `unet`, `vit`.

### Plots de curva de treinamento

```bash
make dl-plot
```

Le arquivos `*_history.json` de `results/dl_models/checkpoints/` e escreve plots em `results/dl_plots/`.

### Pipeline DL completo

```bash
make dl-all    # treinar todos -> avaliar todos -> plotar
```

## Compilacao do Artigo

Copiar assets gerados e compilar o artigo LaTeX:

```bash
make paper-assets    # Copia tabelas e figuras de results/ para docs/
make paper           # paper-assets + compilacao latexmk
make paper-only      # Compilar usando os arquivos existentes em docs/ (sem copia de assets)
```

Saida: `docs/dist/draft.pdf`. Requer XeLaTeX + latexmk.

Instalar toolchain LaTeX (Ubuntu/Debian):

```bash
make install-latex
```

## Cluster (SLURM)

Validado em Rocky Linux 9.5 com NVIDIA A100 80GB PCIe, driver 550.90.07 / CUDA 12.4, toolkit CUDA 12.6.2.

| Recurso        | Valor                     |
| -------------- | ------------------------- |
| Particao (GPU) | `gpuq`                    |
| Particao (CPU) | `cpuq`                    |
| GRES           | `gpu:a100:1`              |
| Modulo Conda   | `miniconda3/py312_25.1.1` |
| Modulo CUDA    | `cuda/12.6.2`             |

### Sequencia de execucao no cluster

```
[Fase 0] setup_env.sh            <- uma vez por cluster, sessao interativa
[Fase 1] preprocess.sbatch       <- uma vez, aguardar COMPLETED antes da Fase 2
[Fase 2] experiment_classical.sbatch  |
         train_ae.sbatch              |  paralelo, todos dependem de Fase 1
         train_vae.sbatch             |  (cada train_*.sbatch roda eval automaticamente)
         train_gan.sbatch             |
         train_unet.sbatch            |
         train_vit.sbatch             |
[Fase 3] aggregate_results.py    <- interativo, apos todos os jobs da Fase 2 completarem
         generate_figures.py     <- interativo
         generate_tables.py      <- interativo
```

### Fase 0 - Preparar ambiente (uma vez por cluster)

```bash
srun -p gpuq --gres=gpu:a100:1 --mem=16G --cpus-per-task=4 --time=00:30:00 \
    bash scripts/slurm/setup_env.sh "$(pwd)"
```

A saida esperada confirma `CUDA available: True` e `Python 3.12.x`.

### Fase 1 - Pre-processar (aguardar completar antes de continuar)

```bash
mkdir -p logs
PREP_JID=$(sbatch --parsable scripts/slurm/preprocess.sbatch)
echo "preprocess job: $PREP_JID"
# Monitorar: squeue -u $USER
# Aguardar estado COMPLETED antes de submeter a Fase 2
```

### Fase 2 - Experimento completo (submeter em paralelo, com dependencia do preprocessamento)

```bash
# Experimento classico (cpuq, 200 CPUs, 128G, 144h)
CLASS_JID=$(sbatch --parsable \
    --dependency=afterok:$PREP_JID \
    scripts/slurm/experiment_classical.sbatch)

# Modelos DL em paralelo (gpuq, cada um com seu proprio job)
AE_JID=$(sbatch   --parsable --dependency=afterok:$PREP_JID scripts/slurm/train_ae.sbatch)
VAE_JID=$(sbatch  --parsable --dependency=afterok:$PREP_JID scripts/slurm/train_vae.sbatch)
GAN_JID=$(sbatch  --parsable --dependency=afterok:$PREP_JID scripts/slurm/train_gan.sbatch)
UNET_JID=$(sbatch --parsable --dependency=afterok:$PREP_JID scripts/slurm/train_unet.sbatch)
VIT_JID=$(sbatch  --parsable --dependency=afterok:$PREP_JID scripts/slurm/train_vit.sbatch)

echo "classical=$CLASS_JID | ae=$AE_JID | vae=$VAE_JID | gan=$GAN_JID | unet=$UNET_JID | vit=$VIT_JID"
```

> Cada `train_*.sbatch` ja roda `train_model.py --config $CONFIG --model <model>`, que ao
> final do treinamento executa `evaluate.py` automaticamente para todos os noise_levels e
> eval_scenarios definidos na config. Nao e necessario submeter jobs de avaliacao separados.

### Monitorar jobs

```bash
squeue -u $USER

# Verificar estado de um job especifico
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS

# Acompanhar log em tempo real
tail -f logs/slurm_pdi-train-unet_<JOB_ID>.out
```

### Fase 3 - Agregar e gerar assets (interativo, apos todos COMPLETED)

Verificar que todos os jobs terminaram com sucesso:

```bash
sacct -j $CLASS_JID,$AE_JID,$VAE_JID,$GAN_JID,$UNET_JID,$VIT_JID \
    --format=JobID,JobName,State,Elapsed
```

Rodar em sessao interativa (nao precisa de GPU):

```bash
# Agregar resultados classicos + historico DL
uv run python scripts/aggregate_results.py \
    --results results/paper_results \
    --dl-history results/dl_models/checkpoints \
    --output results/paper_results/aggregated

# Gerar figuras (usa aggregated/ para fast-path + raw data para fig1-6)
uv run python scripts/generate_figures.py \
    --results results/paper_results \
    --aggregated-dir results/paper_results/aggregated

# Gerar tabelas (inclui tabela comparativa classicos vs DL via --dl-results)
uv run python scripts/generate_tables.py \
    --results results/paper_results \
    --aggregated-dir results/paper_results/aggregated \
    --dl-results results/dl_models

# Empacotar
bash scripts/pack_paper_assets.sh
```

### Smoke tests (antes da producao)

```bash
sbatch scripts/slurm/experiment_classical_quick.sbatch   # ~1h, CPU
sbatch scripts/slurm/train_dl_quick.sbatch               # ~1h, GPU
```

### Diagnostico (cluster novo ou ambiente desconhecido)

```bash
sinfo -o "%P %a %G"
sbatch scripts/slurm/hello_cuda.sbatch
```

O arquivo de saida `logs/slurm_pdi-hello-cuda_<JOB_ID>.out` contem listagem de modulos, informacoes de GPU e status do PyTorch/CUDA.

## Metricas e Analise Estatistica

Todas as metricas sao calculadas **exclusivamente sobre pixels de lacuna** (mask = 1), sem dilucao com pixels nao-afetados.

### Metricas de qualidade (por patch)

| Metrica              | Descricao                                                                   | Melhor |
| -------------------- | --------------------------------------------------------------------------- | ------ |
| PSNR                 | Peak Signal-to-Noise Ratio (dB) = 10\*log10(1/MSE)                          | maior  |
| SSIM                 | Structural Similarity Index - media do mapa espacial sobre pixels de lacuna | maior  |
| RMSE                 | Root Mean Squared Error sobre pixels de lacuna                              | menor  |
| SAM                  | Spectral Angle Mapper - angulo medio (graus) entre vetores espectrais       | menor  |
| ERGAS                | Relative Global Adimensional Synthesis Error (h/l=1 para gap-filling)       | menor  |
| RMSE_b{0-3}          | RMSE por banda espectral                                                    | menor  |
| pixel_acc_002/005/01 | Fracao de pixels de lacuna com \|erro\| <= 0.02 / 0.05 / 0.10               | maior  |
| f1_002/005/01        | F1 derivado de pixel_acc: 2\*acc/(1+acc)                                    | maior  |

> **Nota SSIM:** computado sobre a imagem inteira com `structural_similarity(full=True)` e
> media restrita aos pixels de lacuna. A janela deslizante que inclui vizinhos nao-lacuna e
> intencional - captura continuidade de fronteira.

> **Nota F1:** F1 = 2\*acc/(1+acc) assume Precision=1 (nao ha falsos positivos, pois somente
> pixels de lacuna sao avaliados). F1 e transformacao monotona de pixel_acc; ambos sao
> reportados por convencao.

### Analises estatisticas

- Correlacao de Pearson e Spearman com correcao FDR Benjamini-Hochberg (alpha = 0,05).
- Kruskal-Wallis + Mann-Whitney U post-hoc (correcao Bonferroni), epsilon-squared, Cliff's delta.
- Regressao robusta (RLM/HuberT): metrica ~ entropia (multi-escala) + metodo + ruido. Reporta beta, p-values, IC95%, R-squared ajustado, VIF.
- Moran's I (global) e LISA (local) para autocorrelacao espacial de mapas de erro de reconstrucao.
- Bootstrap 95% CI (n=10.000) para todas as medias de grupo.

## Estrutura de Saida

```
results/
  paper_results/
    raw_results.csv              # todos os patches: metodos classicos x seeds x noise x satelite
    entropy/                     # mapas de entropia para top-k patches
    figures/                     # figuras PDF e PNG (8 figuras)
    tables/                      # arquivos de tabelas LaTeX (8 tabelas)
    aggregated/                  # CSVs prontos para analise (gerados por aggregate_results.py)
      by_method_{metric}.csv
      by_noise_{metric}.csv
      by_satellite_{metric}.csv
      by_gap_bin_{metric}.csv
      by_entropy_bin_{ws}_{metric}.csv
      spearman_correlation.csv
      method_comparison_global.csv
      method_comparison_pairwise.csv
      robust_regression_coefs.csv
      robust_regression_vif.csv
      robust_regression_summary.csv
      dl_training_history.csv
      dl_model_metadata.csv
      dl_eval_summary.csv
      combined_comparison.csv

  dl_models/
    checkpoints/
      {model}_best.pth           # checkpoint do melhor epoch (menor val_loss)
      {model}_history.json       # historico por epoch: train/val loss, val_psnr/ssim/rmse/pixel_acc/f1
    eval/
      {scenario_name}/           # entropy_high | entropy_medium_high | entropy_all
        {model}/
          {model}_gap_only.csv   # noise_level=inf  (schema completo)
          {model}_snr40dB.csv    # noise_level=40
          {model}_snr30dB.csv    # noise_level=30
          {model}_snr20dB.csv    # noise_level=20

paper_assets/
  classic/
    figures/
    tables/
    raw_results/
    logs/
  dl/
    eval_results/
    figures/
    history/
    training_logs/
    training_plots/
```

> **Historico vs Avaliacao DL:** `*_history.json` contem metricas do **val set durante
> treinamento** (psnr, ssim, rmse, pixel_acc, f1 - sem SAM/ERGAS). Os CSVs em `eval/`
> contem metricas do **test set** com schema completo (inclui SAM, ERGAS, entropy,
> gap_fraction). Tabelas e figuras do artigo devem usar os CSVs de `eval/`, nao o historico.

## Parametrizacoes

| Parametro                 | Valores                                                    |
| ------------------------- | ---------------------------------------------------------- |
| Janelas de entropia       | 7x7, 15x15, 31x31                                          |
| Tamanho do patch          | 64x64                                                      |
| Niveis de ruido (SNR)     | inf, 40, 30, 20 dB                                         |
| Seeds aleatorias (artigo) | 10 (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021) |
| Max patches por config    | 75 (artigo) / 1 (rapido)                                   |
| Alpha estatistico         | 0,05 com correcao FDR                                      |

## Figuras e Tabelas

**Figuras (8):**

1. Mapas de entropia local em 3 escalas (7x7, 15x15, 31x31) para todos os sensores.
2. Scatterplots entropia vs. PSNR por metodo (15 classicos) com reta de regressao, beta, p-value, R-squared.
3. Boxplots PSNR por metodo, agrupados por tercil de entropia (baixo / medio / alto).
4. Boxplots PSNR por metodo, agrupados por nivel de ruido.
5. Mapas de clusters LISA sobrepostos em mapas de erro de reconstrucao.
6. Exemplos visuais: clean / degraded / todos os metodos para patches de baixa e alta entropia.
7. Heatmap de correlacao Spearman rho (entropia multi-escala vs. metricas).
8. Mapas de PSNR local e SSIM local para patch representativo (todos os metodos).

**Tabelas (8):**

1. Visao geral dos 15 metodos classicos (categoria, parametros, complexidade).
2. Media PSNR +/- IC95% por metodo x nivel de ruido.
3. Media PSNR +/- IC95% estratificada por tercil de entropia em cada escala.
4. Correlacao Spearman entre entropia multi-escala e metricas (p-values FDR-corrigidos).
5. Kruskal-Wallis (H, p, epsilon-squared) e pares significativos (Dunn post-hoc, Cliff's delta).
6. Regressao robusta (RLM/HuberT) por metrica: beta, p-values, R-squared, R-squared ajustado, VIF.
7. Media PSNR +/- IC95% por metodo x satelite.
8. Comparacao classicos vs. DL (PSNR, SSIM, RMSE, SAM, ERGAS, pixel_acc side-by-side).

## Licenca

MIT - ver [LICENSE](LICENSE).
