# Analise quantitativa e espacial da relacao entre entropia local e desempenho de metodos classicos de interpolacao de lacunas em imagens de satelite

## Objetivo

Avaliar e comparar procedimentos de processamento de imagem para preenchimento de lacunas em imagens de satelite e quantificar, por meio de metricas objetivas e analise espacial, a correlacao entre entropia calculada em janelas locais e a perda de qualidade dos metodos, identificando regioes com maior deficiencia dos algoritmos convencionais.

## Hipoteses

- H1 - Regioes com maior entropia local (mais textura/complexidade) apresentarao pior desempenho medio de interpoladores classicos.
- H2 - Metodos baseados em modelos espectro-temporais e geostaticos terao desempenho superior em areas de baixa entropia.
- H3 - Existe relacao estatisticamente significante entre metricas de qualidade locais e entropia local.

## Tecnicas

1. Interpoladores espaciais classicos (nearest neighbor, bilinear, bicubic, Lanczos).
2. Metodos baseados em kernel (IDW, RBF, splines).
3. Metodos geostaticos (kriging).
4. Metodos de transformadas/regularizacao (DCT, wavelets, TV inpainting).
5. Metodos patch-based (exemplar-based, non-local).
6. Metodos de compressive sensing (L1 em wavelet/DCT).
7. Baselines de aprendizado profundo (AE, VAE, GAN, ViT) - avaliados separadamente.

## Base de Dados

Uso de Sentinel-2, Landsat-8/9 e MODIS. Simulacao de lacunas (mascaras de nuvem realistas, patches de diferentes tamanhos, niveis de ruido). Validacao com cenas limpas temporais.

## Pipeline

1. Coleta e pre-processamento.
2. Segmentacao em janelas.
3. Calculo de entropia local.
4. Aplicacao dos metodos.
5. Avaliacao (PSNR, SSIM, RMSE, SAM, ERGAS).
6. Analise estatistica e espacial.
7. Visualizacao (mapas, graficos, tabelas).
8. Relatorio e recomendacoes.

## Metricas e validacao estatistica

**Metricas:** PSNR, SSIM, RMSE, IoU, SAM, ERGAS.

**Analises:** correlacao (Pearson, Spearman), regressao robusta, ANOVA/Kruskal-Wallis, testes pareados, bootstrap IC95%, Moran's I, LISA.

## Parametrizacoes

- Janelas de entropia: 7x7, 15x15, 31x31.
- Patches 64x64 com sobreposicao.
- N minimo de patches: 500-2000.
- Niveis de ruido: infinito, 40 dB, 30 dB, 20 dB.
- Repeticoes: 10 seeds aleatorias.
- alpha=0.05 com correcao multipla (FDR).

## Posssiveis limitacoes

- Disponibilidade de ground truth.
- Sensibilidade do Kriging.
- Custo computacional de metodos patch-based.

## Resultados

### Figuras

- **Fig 1:** Mapas de entropia local em 3 escalas (7x7, 15x15, 31x31), TODOS os sensores.
- **Fig 2:** Scatterplots entropia vs. PSNR por metodo (15 metodos classicos) incluindo coeficientes (beta), p-values, R-squared, R-squared ajustado, alem da reta de regressao.
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

## Deep Learning - Treinamento e Avaliacao

### Pre-processamento

Converte patches GeoTIFF para NPY. Gera `preprocessed/manifest.csv` com splits `train`, `val` e `test`. Resume-safe - se o manifest ja estiver completo, pula automaticamente.

```bash
uv run python scripts/preprocess_dataset.py --resume
```

### Quick-start (via Make)

Os targets do `make` usam defaults rapidos (2 epochs) para validacao rapida:

```bash
make dl-train-vae        # treina VAE com 2 epochs
make dl-train-all        # treina todos os 5 modelos sequencialmente
make dl-eval-all         # avalia todos no test split
```

Para alterar epochs via make: `make dl-train-ae AE_EPOCHS=50`.

### Treinamento (producao)

Comandos completos com hiperparametros de producao:

```bash
# AE - Autoencoder
uv run python -m dl_models.ae.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output dl_models/checkpoints/ae_best.pth \
    --epochs 50 --batch-size 32 --lr 1e-3 --patience 10

# VAE - Variational Autoencoder
uv run python -m dl_models.vae.train \
    --manifest preprocessed/manifest.csv --satellite sentinel2 \
    --output dl_models/checkpoints/vae_best.pth \
    --epochs 60 --batch-size 32 --lr 1e-3 --beta 0.001 --patience 10

# GAN - UNet Generator + PatchGAN Discriminator
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

### Avaliacao

```bash
# Exemplo: avaliar U-Net no test split
uv run python -m dl_models.evaluate \
    --model unet \
    --checkpoint dl_models/checkpoints/unet_best.pth \
    --manifest preprocessed/manifest.csv \
    --satellite sentinel2 \
    --output results/dl_eval

# Modelos disponiveis: ae, vae, gan, unet, vit
```

## Cluster (SLURM)

Este projeto requer Python >= 3.12. A configuracao abaixo foi validada no
cluster alvo (Rocky Linux 9.5, NVIDIA A100 80GB PCIe, driver NVIDIA
550.90.07 / CUDA 12.4, toolkit CUDA 12.6.2).

Valores confirmados:

| Recurso | Valor |
|---------|-------|
| Particao (GPU) | `gpuq` |
| Particao (CPU) | `cpuq` |
| GRES | `gpu:a100:1` |
| Modulo Conda | `miniconda3/py312_25.1.1` |
| Modulo CUDA | `cuda/12.6.2` |

Se for submeter em um cluster diferente, use `hello_cuda.sbatch` para
redescobrir os valores corretos (veja a secao "Diagnostico" abaixo).

### 1) Preparar o ambiente (executar apenas uma vez)

O script `setup_env.sh` cria o ambiente conda `pdi312`, instala todas as
dependencias do projeto e instala o PyTorch com suporte a CUDA
(torch 2.3.1+cu121, compativel com o runtime CUDA 12.4 do driver). Ele
tambem exporta `PYTHONNOUSERSITE=1` para impedir que pacotes instalados
em `~/.local` interfiram no ambiente.

**Importante:** o setup deve ser executado em um no com GPU, pois a
verificacao final testa se o PyTorch consegue acessar a placa. Nos de login
nao possuem GPU, entao `torch.cuda.is_available()` retornara `False` se o
script for executado diretamente com `bash`.

Use `srun` para alocar um no da particao `gpuq` de forma interativa:

```bash
# Aloca 1 GPU A100, 16 GB de RAM, 4 CPUs e 30 minutos de tempo limite.
# O script instala o ambiente conda e verifica se CUDA esta acessivel.
srun -p gpuq --gres=gpu:a100:1 --mem=16G --cpus-per-task=4 --time=00:30:00 \
    bash scripts/slurm/setup_env.sh "$(pwd)"
```

Ao final da execucao, a saida deve mostrar:

```
torch: 2.3.1+cu121
torch.version.cuda: 12.1
CUDA available: True
  GPU 0: NVIDIA A100 80GB PCIe  (capability 8.0)
Environment ready: Python 3.12.12
```

Se `CUDA available: False` aparecer, verifique se o comando foi executado
via `srun` (e nao diretamente no no de login) e se a particao `gpuq` esta
correta (`sinfo -o "%P %a %G"`).

> **Nota sobre pacotes pip do usuario:** se uma versao diferente do torch
> existir em `~/.local/lib/python3.12/site-packages` (por exemplo, de uma
> instalacao anterior), ela sera ignorada enquanto `PYTHONNOUSERSITE=1`
> estiver definido. Todos os scripts de job em `scripts/slurm/` ja exportam
> essa variavel.

### 2) Definir o caminho do dataset

Exporte a variavel `PDI_DATA_ROOT` apontando para o diretorio raiz dos
dados brutos. Os scripts de pre-processamento e treinamento usam essa
variavel para localizar as imagens:

```bash
export PDI_DATA_ROOT=/caminho/para/o/dataset
```

### 3) Pre-processar os dados (executar uma vez, compartilhado entre todos os workflows)

O pre-processamento converte os patches GeoTIFF para NPY e gera o arquivo
`preprocessed/manifest.csv` com os splits de treino, validacao e teste.

Opcao recomendada - submeter via SLURM (roda na particao `cpuq` com 16 CPUs
e 64 GB de RAM, tempo limite de 24h):

```bash
sbatch scripts/slurm/preprocess.sbatch
```

Opcao alternativa - executar diretamente (util para testes rapidos ou
ambientes sem SLURM):

```bash
python scripts/preprocess_dataset.py --resume
```

A flag `--resume` permite retomar o processamento de onde parou caso o
job seja interrompido.

### 4) Treinar todos os modelos de DL na GPU (SLURM)

Submeta o job de treinamento. Ele treina os 5 modelos (AE, VAE, GAN, U-Net,
ViT) sequencialmente na mesma alocacao de GPU. O job usa a particao `gpuq`
com 1 GPU A100, 16 CPUs, 64 GB de RAM e tempo limite de 72 horas:

```bash
sbatch scripts/slurm/train_all.sbatch
```

O job executa `python` diretamente (nao depende do `make`). Os
hiperparametros podem ser ajustados via variaveis de ambiente antes da
submissao:

```bash
# Exemplo: alterar epochs e batch size do VAE
VAE_EPOCHS=100 VAE_BATCH=16 sbatch scripts/slurm/train_all.sbatch
```

Variaveis disponiveis: `AE_EPOCHS`, `AE_BATCH`, `AE_LR`, `AE_PATIENCE`,
`VAE_EPOCHS`, `VAE_BATCH`, `VAE_LR`, `VAE_BETA`, `VAE_PATIENCE`,
`GAN_EPOCHS`, `GAN_BATCH`, `GAN_LR`, `GAN_LAMBDA_L1`, `GAN_LAMBDA_ADV`,
`GAN_PATIENCE`, `UNET_EPOCHS`, `UNET_BATCH`, `UNET_LR`, `UNET_WD`,
`UNET_PATIENCE`, `VIT_EPOCHS`, `VIT_BATCH`, `VIT_LR`, `VIT_WD`,
`VIT_PATIENCE`.

### 5) Monitorar jobs e inspecionar saida

```bash
# Listar jobs em execucao do usuario atual
squeue -u $USER

# Ver detalhes de um job especifico (tempo, memoria, status)
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS

# Acompanhar a saida em tempo real
tail -f slurm_pdi-train-all_<JOB_ID>.out

# Ver logs de erro
cat slurm_pdi-train-all_<JOB_ID>.err
```

### Diagnostico (cluster novo ou ambiente com problemas)

O `hello_cuda.sbatch` e um job apenas de diagnostico que coleta informacoes
do ambiente sem abortar em caso de erros. Use-o para descobrir nomes de
modulos, particoes disponiveis, GPUs acessiveis e estado do CUDA:

```bash
# Primeiro, verificar as particoes e GRES disponiveis no cluster
sinfo -o "%P %a %G"

# Submeter o diagnostico com os valores padrao
sbatch scripts/slurm/hello_cuda.sbatch

# Ou sobrescrever particao/GRES se os padroes nao corresponderem ao cluster
sbatch -p <particao> --gres=gpu:1 scripts/slurm/hello_cuda.sbatch
```

O arquivo de saida (`slurm_pdi-hello-cuda_<JOB_ID>.out`) contera secoes
para: modulos disponiveis, modulos carregados, informacoes de GPU via
nvidia-smi e status do PyTorch/CUDA.
