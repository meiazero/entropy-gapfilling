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
7. Baselines de aprendizado profundo (AE, VAE, GAN, Transformer) - avaliados separadamente.

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
