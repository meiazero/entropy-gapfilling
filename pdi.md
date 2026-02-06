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
3. Métodos geostáticos (kriging, DINEOF).
4. Métodos de transformadas/regularização (DCT, wavelets, TV inpainting).
5. Métodos patch-based (exemplar-based, non-local).
6. Métodos multi-temporais (spline temporal, Fourier temporal, kriging espaço-tempo).
7. Métodos de compressive sensing (L1 em wavelet/DCT).

## Base de Dados

Uso de Sentinel-2, Landsat-8/9 e MODIS. Simulação de lacunas (máscaras de nuvem realistas, patches de diferentes tamanhos, níveis de ruído). Validação com cenas limpas temporais.

## Pipeline

1. Coleta e pré-processamento.
2. Segmentação em janelas.
3. Cálculo de entropia local.
4. Aplicação dos métodos.
5. Avaliação (PSNR, SSIM, RMSE, IoU, SAM).
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

- Mapas de entropia, PSNR local (media, mediana, IC95%), SSIM local (media, mediana, IC95%).
- Scatterplots entropia vs PSNR.
- Boxplots por método e cenário.
- Hotspot maps (LISA).
- Tabelas médias ± IC95%.
- Sumário executivo com recomendações.
- Comparação com métodos baseados em aprendizado profundo
- Adição de variáveis auxiliares (NDVI, elevação DEM) para intelligent weighting na interpolação
- Tabela: média PSNR ± IC95% por método (colunas) × cenário (linhas).
- Matriz de correlação entre entropia (multi-escala) e métricas (PSNR, SSIM, RMSE) com p-values.
- Resultados de ANOVA/Kruskal: F/H e p para comparação global entre métodos.
- Regressão: coeficientes estimados (β), p-values, R² ajustado, VIF.
