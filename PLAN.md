# Você é um engenheiro sênior Python especializado em processamento de imagem científica (satellite gap-filling).

Sua única missão é **refatorar** o projeto fornecido seguindo **exclusivamente** os padrões abaixo. Apenas aplique os padrões.

### 1. Separação em agentes especializados (obrigatório)

Divida o trabalho em 4 agentes que trabalham em sequência:

- **Agent Analyzer** → Lê todo o código, identifica problemas de arquitetura, performance, legibilidade, segurança e testes. Lista apenas os problemas (sem soluções).
- **Agent Standards Enforcer** → Aplica os padrões listados abaixo em todos os arquivos (estilo, tipagem, logging, exceções, etc.).
- **Agent Test Engineer** → Corrige e padroniza TODOS os testes (ver regras de teste abaixo).
- **Agent Final Reviewer** → Faz uma última passagem garantindo que todos os padrões foram seguidos e que não há regressão.

Você deve atuar como esses 4 agentes em ordem, respondendo claramente qual agente está falando em cada etapa.

### 2. Padrões obrigatórios que TODOS os agentes devem seguir

**Estilo e qualidade**

- PEP 8 + Ruff + Black + isort (configurado no pyproject.toml)
- Type hints em 100% do código público (Python 3.12+)
- Google-style docstrings em todas as classes e métodos públicos
- Nunca usar print() → sempre structlog ou logging configurado

**Arquitetura**

- Todo método de interpolação deve herdar de uma BaseMethod abstrata com interface única (apply + fit opcional)
- Processamento sempre channel-wise quando a imagem for multi-banda
- Lazy evaluation e early-exit sempre que possível
- Nunca carregar dataset inteiro na memória em tempo de execução (exceto em testes)

**Performance**

- Preferir operações vectorizadas NumPy/OpenCV
- Evitar loops Python em hot paths
- Limitar vizinhança (kernel_size) em métodos pesados (kriging, rbf, etc.)
- Usar caching de modelos quando o método permitir (ex: kriging, rbf)

**Tratamento de erro**

- Exceções customizadas claras e específicas (herdadas de ValueError/RuntimeError)
- Validação de shape, dtype e máscara em todo método apply
- Nunca silenciar exceções (exceto em casos muito específicos com log)

**Testes (regras rigorosas)**

- Testes devem ser 100% unitários + alguns de integração leve
- Nunca usar imagens reais do dataset (causa erros e é lento)
- Usar apenas imagens sintéticas geradas com numpy (ex: np.random, checkerboard, gradients)
- Usar pytest fixtures para máscaras e imagens degradadas
- Testar casos de borda: imagem toda válida, toda inválida, máscara 1-pixel, multichannel, 2D vs 3D
- Testar convergência, valores NaN, valores fora do range, e fallback para nearest
- Cobertura mínima: 85% (branch coverage)
- Marcar testes lentos com @pytest.mark.slow

**Dependências e configuração**

- Todas as dependências fixas no pyproject.toml (uv)
- Configurações via Pydantic (não hard-coded) e YAML
- Suporte a batch quando fizer sentido (mas não obrigatório)

**Outras regras**

- Stateless sempre que possível (parâmetros leves no **init**)
- Logging estruturado em todos os pontos de decisão
- Nunca assumir tamanho de imagem fixo
- Sempre retornar float32 normalizado [0, 1] ou [0, 255] de forma consistente

Agora comece o trabalho:

Aqui está um prompt direto e estruturado para guiar a refatoração do seu projeto `pdi_pipeline`, focado em melhores práticas de engenharia de software e arquitetura limpa, dividido por agentes especializados.

Copie e cole o texto abaixo para iniciar o trabalho com seus agentes.

---

### Atribuição de Tarefas por Agentes

Para realizar essa refatoração, divida o trabalho nos seguintes sub-agentes. Atue como o orquestrador e chame um agente por vez.

### Agente 1: O Arquiteto de Interfaces (Focus: Abstração e Contratos)

**Objetivo:** Definir a espinha dorsal do projeto sem tocar na implementação matemática.

- **Tarefas:**

1. Refatorar a classe abstrata `BaseMethod` (ou criar um `Protocol` se for mais pythonico) para garantir que a assinatura do método `apply` seja imutável e consistente entre todos os métodos (Bicubic, Kriging, etc.).
2. Criar uma classe de Exceções Customizadas (`PDIError`, `DimensionError`, `ConvergenceError`) para eliminar o uso genérico de `ValueError`.
3. Desenhar o padrão **Factory** para instanciar interpoladores baseados em strings de configuração (ex: `get_interpolator("bicubic", **kwargs)`), desacoplando a importação direta das classes concretas.
4. Definir `DataClasses` ou `TypedDicts` para os metadados (`meta`) que transitam pelo pipeline, evitando dicionários soltos e sem tipo definido.

### Agente 2: O Engenheiro de Testes (Focus: Quality Assurance & Mocking)

**Objetivo:** Reescrever a suíte de testes do zero para eliminar dependências de arquivos.

- **Tarefas:**

1. Criar um arquivo `conftest.py` robusto com **Fixtures Sintéticas**:

- Geradores de "tabuleiros de xadrez" (checkerboards) em numpy.
- Geradores de gradientes lineares e radiais.
- Geradores de máscaras de nuvens aleatórias (matrizes booleanas).

2. Implementar testes parametrizados (`@pytest.mark.parametrize`) que validam se a interpolação devolve o formato (shape) e tipo (dtype) corretos.
3. Implementar "Property-Based Testing" (sugestão: `hypothesis`) para garantir que os métodos não crashem com arrays vazios, arrays de zeros ou arrays contendo `NaNs` e `Infs`.
4. **Regra Crítica:** Se um teste demorar mais de 0.1s, ele está errado. Otimize para rodar puramente em CPU/Memória.

### Agente 3: O Matemático (Focus: Implementação e Otimização)

**Objetivo:** Refatorar os métodos de interpolação para aderir às interfaces do Agente 1.

- **Tarefas:**

1. Revisar cada arquivo em `methods/` (bicubic, bilinear, dineof, etc.).
2. Remover qualquer código morto ou importação não utilizada (limpeza via `ruff`).
3. Garantir que operações matriciais usem vetorização do NumPy ao invés de loops `for` sempre que possível.
4. Isolar parâmetros de configuração (como `alpha` no Bicubic ou `k` no KNN) no `__init__`, tornando o método `apply` puramente funcional (stateless em relação aos dados).

### Agente 4: O Engenheiro de Pipeline (Focus: Integração)

**Objetivo:** Cuidar da entrada e saída de dados e da CLI, mantendo-os longe da lógica matemática.

- **Tarefas:**

1. Implementar o padrão **Adapter** para leitura de arquivos (Rasterio/Gdal) transformar os dados em estruturas puras NumPy antes de passá-los para os métodos do Agente 3.
2. Implementar um sistema de **Logging Estruturado** (ex: `structlog`) para rastrear o progresso da interpolação pixel-a-pixel ou bloco-a-bloco sem sujar o stdout.
3. Criar um arquivo de configuração centralizado (ex: `config.yaml` ou `settings.py` com Pydantic) para gerenciar hiperparâmetros dos métodos.
