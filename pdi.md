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
│   ├── classico/               # Figuras, tabelas, resultados brutos e logs do experimento classico
│   └── dl/                     # Resultados de avaliacao, plots de treinamento e historico do experimento DL
├── results/
│   ├── paper_results/          # Saida do experimento completo do artigo
│   └── quick_validation/       # Saida da execucao de validacao rapida
├── scripts/
│   ├── preprocess_dataset.py   # Converte patches GeoTIFF para NPY + gera manifest
│   ├── run_experiment.py       # Executador principal do experimento de metodos classicos
│   ├── generate_figures.py     # Gera todas as figuras do artigo a partir dos resultados
│   ├── generate_tables.py      # Gera todas as tabelas LaTeX a partir dos resultados
│   ├── precompute_entropy.py   # Pre-computa mapas de entropia para todos os patches
│   ├── pack_paper_assets.sh    # Empacota assets do artigo em paper_assets.zip
│   ├── grid_search_classical.py
│   ├── grid_search_dl.py
│   └── slurm/                  # Scripts de submissao SLURM (ver secao Cluster)
├── src/pdi_pipeline/
│   ├── methods/                # Implementacoes dos 15 metodos classicos de preenchimento
│   ├── metrics.py              # PSNR, SSIM, RMSE, SAM, ERGAS
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
  models:
    ae: { epochs: 30, batch_size: 32, lr: 1e-3, patience: 7 }
    vae: { epochs: 40, batch_size: 32, lr: 1e-3, beta: 0.001, patience: 8 }
    gan:
      {
        epochs: 60,
        batch_size: 16,
        lr: 2e-4,
        lambda_l1: 10.0,
        lambda_adv: 0.1,
        patience: 10
      }
    unet:
      { epochs: 40, batch_size: 32, lr: 1e-3, weight_decay: 1e-4, patience: 8 }
    vit:
      { epochs: 60, batch_size: 32, lr: 1e-4, weight_decay: 0.05, patience: 10 }
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
uv run python scripts/run_experiment.py --config config/paper_results.yaml --save-entropy-top-k 5
```

Use a flag `--quick` para rodar com `config/quick_validation.yaml` automaticamente.

### 4. Gerar figuras e tabelas

```bash
uv run python scripts/generate_figures.py --results results/paper_results
uv run python scripts/generate_tables.py  --results results/paper_results
```

Adicione `--png-only` para pular a saida PDF (mais rapido, util para previews).

### 5. Empacotar assets do artigo

```bash
bash scripts/pack_paper_assets.sh
# Produz o diretorio paper_assets/ e paper_assets.zip
```

## Pipeline de Aprendizado Profundo

### Treinamento (local)

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

### Avaliacao (test split)

```bash
make dl-eval-all          # Avaliar todos os 5 modelos
make dl-eval-unet         # Avaliar um modelo individual
```

Checkpoints esperados em `results/dl_models/checkpoints/` (configuravel via `CKPT_DIR`).

### Plots de curva de treinamento

```bash
make dl-plot
```

Le arquivos `*_history.json` de `dl_models/` e escreve plots em `results/dl_plots/`.

### Pipeline DL completo

```bash
make dl-all    # treinar todos -> avaliar todos -> plotar
```

### Invocacao direta

```bash
# Treinamento
uv run python -m dl_models.ae.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output results/dl_models/checkpoints/ae_best.pth \
    --epochs 30 --batch-size 32 --lr 1e-3 --patience 7

# Avaliacao
uv run python -m dl_models.evaluate \
    --model unet \
    --checkpoint results/dl_models/checkpoints/unet_best.pth \
    --manifest preprocessed/manifest.csv \
    --satellite sentinel2 \
    --output results/dl_eval
```

Modelos disponiveis: `ae`, `vae`, `gan`, `unet`, `vit`.

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

### Sequencia de execucao

```
[1] setup_env.sh              <- uma vez, aguardar completar
[2] export PDI_DATA_ROOT      <- obrigatorio antes de qualquer job
[3] preprocess.sbatch         <- uma vez, aguardar completar (squeue -u $USER)
        |
        +-- [4a] submit_all.sh               (todos os 5 modelos DL, paralelo)
        +-- [4b] experiment_classical.sbatch (metodos classicos, independente)
```

Os passos 1-3 sao pre-requisitos sequenciais. Nao submeta jobs de treinamento antes de o pre-processamento terminar.

### 1. Preparar ambiente (uma vez)

```bash
# Alocar 1 no GPU de forma interativa e executar o setup
srun -p gpuq --gres=gpu:a100:1 --mem=16G --cpus-per-task=4 --time=00:30:00 \
    bash scripts/slurm/setup_env.sh "$(pwd)"
```

A saida esperada confirma `CUDA available: True` e `Python 3.12.x`.

### 2. Definir caminho do dataset

```bash
export PDI_DATA_ROOT=/caminho/para/imagens-satelite
```

### 3. Pre-processar

```bash
sbatch scripts/slurm/preprocess.sbatch
```

### 4a. Treinar modelos DL

Submeter todos os 5 modelos em paralelo (cada um em seu proprio job):

```bash
bash scripts/slurm/submit_all.sh
```

Ou submeter modelos individuais:

```bash
sbatch scripts/slurm/train_ae.sbatch    # 8h, 48G
sbatch scripts/slurm/train_vae.sbatch   # 12h, 48G
sbatch scripts/slurm/train_gan.sbatch   # 36h, 64G
sbatch scripts/slurm/train_unet.sbatch  # 16h, 48G
sbatch scripts/slurm/train_vit.sbatch   # 36h, 64G
```

Usar uma configuracao diferente:

```bash
PDI_CONFIG=config/quick_validation.yaml bash scripts/slurm/submit_all.sh
PDI_CONFIG=config/quick_validation.yaml sbatch scripts/slurm/train_ae.sbatch
```

### 4b. Experimento classico

Roda em `cpuq` (32 CPUs, 128G, 72h):

```bash
sbatch scripts/slurm/experiment_classical.sbatch
```

### 5. Smoke tests (antes da producao)

```bash
sbatch scripts/slurm/experiment_classical_quick.sbatch   # ~1h, CPU
sbatch scripts/slurm/train_dl_quick.sbatch               # ~1h, GPU
```

### 6. Monitorar jobs

```bash
squeue -u $USER
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS
tail -f slurm_pdi-train-all_<JOB_ID>.out
```

### Diagnostico (cluster novo ou ambiente desconhecido)

```bash
sinfo -o "%P %a %G"
sbatch scripts/slurm/hello_cuda.sbatch
# ou sobrescrever particao/GRES:
sbatch -p <particao> --gres=gpu:1 scripts/slurm/hello_cuda.sbatch
```

O arquivo de saida `slurm_pdi-hello-cuda_<JOB_ID>.out` contem listagem de modulos, informacoes de GPU e status do PyTorch/CUDA.

## Metricas e Analise Estatistica

**Metricas de qualidade de imagem:** PSNR, SSIM, RMSE, SAM, ERGAS.

**Analises estatisticas:**

- Correlacao de Pearson e Spearman com correcao FDR (alpha = 0,05).
- Kruskal-Wallis + Dunn post-hoc (Bonferroni), epsilon-squared, Cliff's delta.
- Regressao robusta (RLM / HuberT): metrica ~ entropia (multi-escala) + metodo + ruido. Inclui coeficientes beta, p-values, IC95%, R-squared ajustado, VIF.
- Moran's I (global) e LISA (local) para autocorrelacao espacial de mapas de erro de reconstrucao.

## Estrutura de Saida

```
results/
  paper_results/
    raw_results.parquet         # Resultados por patch para todos os metodos, seeds, niveis de ruido
    entropy/                    # Mapas de entropia salvos para os top-k patches
    figures/                    # Figuras PDF e PNG (8 figuras)
    tables/                     # Arquivos de tabelas LaTeX (8 tabelas)
  dl_eval/
    {model}_inpainting/
      results.csv
  dl_plots/
    {model}_training_curves.{pdf,png}

paper_assets/
  classico/
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
8. Comparacao classicos vs. DL (PSNR, SSIM, RMSE side-by-side).

## Licenca

MIT - ver [LICENSE](LICENSE).
