# Decifrando a Caixa-Preta: Tornando Modelos de IA Explicáveis com LIME (German Credit – UCI)

> Projeto pronto para rodar: classificação de risco de crédito com **Random Forest** e explicações locais via **LIME**.

## 📦 O que vem no projeto

```
xai-lime-credit/
├─ data/                      # Onde o dataset será salvo automaticamente (UCI)
├─ imgs/                      # Espaço para imagens extras
├─ notebooks/                 # (opcional) você pode criar/usar notebooks aqui
├─ outputs/                   # Resultados (métricas, gráficos, explicações LIME)
├─ src/
│  └─ train_and_explain.py    # Código principal (baixa dados, treina e gera LIME)
├─ requirements.txt           # Dependências
└─ run.bat                    # (Windows) automatiza tudo
```

## 🧠 Objetivo

Você foi contratada(o) para explicar **por que** um modelo de crédito aprova ou nega clientes.  
Aqui usamos o dataset **Statlog (German Credit Data)** da UCI e aplicamos **LIME** para criar **explicações locais** (por cliente) mostrando as características que mais pesaram na decisão.

- Dataset (UCI): veja detalhes, formatos e custo/penalidade 1=Good, 2=Bad.  
- Artigo LIME (base teórica): *Why Should I Trust You?* (Ribeiro, Singh, Guestrin, 2016).  
- Documentação do LIME: exemplos e parâmetros.

## ✅ Entregáveis que este projeto cobre

- **Parte prática (4,0 pts)**: código limpo, funcional, outputs (HTML/PNG), `requirements.txt`, link para dados.
- **Parte teórica (2,0 pts)**: este README traz contexto, decisões de modelagem, análise das explicações e limitações.
- **Vídeo pitch (2,0 pts)**: roteiro sugerido ao final para gravar em até 4 minutos.

---

# 1) Passo a passo — como rodar **do zero** (Windows)

> Dica: se preferir o caminho rápido, apenas **clique duas vezes** em `run.bat` e pule para o passo 5.

## Passo 1 — Instalar o Python
1. Acesse https://www.python.org/downloads/  
2. Clique em **Download Python 3.x**.  
3. **Importante:** marque a caixinha **“Add python.exe to PATH”** antes de clicar em *Install Now*.

## Passo 2 — Baixar este projeto
1. Salve o arquivo 
2. Clique com o botão direito no `.zip` → **Extrair Tudo…** → escolha uma pasta fácil, por exemplo `C:\Users\SEU_USUARIO\Documents\xai-lime-credit\`.

## Passo 3 — Abrir a pasta do projeto
- Abra o **Explorador de Arquivos** e entre na pasta `xai-lime-credit` que você extraiu.

## Passo 4 — (Opcional) Rodar com 2 cliques
- Dê **dois cliques** em `run.bat`.  
  Isso vai:
  - criar um ambiente virtual (`.venv`),
  - instalar as dependências (`requirements.txt`),
  - **baixar o dataset automaticamente da UCI**, 
  - treinar o modelo e
  - gerar as explicações com LIME.
- Quando terminar, a janela vai mostrar “FIM!” — os resultados ficam em `outputs/`.

## Passo 5 — Rodar manualmente (linha de comando)
1. Clique no menu Iniciar e digite **cmd** → abra o **Prompt de Comando**.
2. Navegue até a pasta (substitua o caminho pelo seu):
   ```bat
   cd C:\Users\SEU_USUARIO\Documents\xai-lime-credit
   ```
3. Crie e ative o ambiente virtual:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Instale as dependências:
   ```bat
   pip install -r requirements.txt
   ```
5. Rode o script principal:
   ```bat
   python src\train_and_explain.py
   ```
6. Veja os arquivos gerados na pasta `outputs/`:
   - `metrics.txt` (AUC, relatório de classificação, matriz de confusão)
   - `graficos_importancias_top20.png`
   - `lime_explanation_*.html` e `lime_explanation_*.png`
   - `amostra_predicoes.csv` (20 primeiras previsões com probabilidade)

> **Observação importante:** o script usa o arquivo **categórico original** `german.data` (20 atributos) e mapeia o alvo para 0=Good, 1=Bad.  
> As features categóricas são codificadas via *One-Hot Encoder*, e as numéricas são padronizadas. O LIME é executado sobre o **espaço transformado** (pós-encoding), o que resulta em explicações interpretáveis como “Purpose=A44”, “Property=A121”, etc.

---

# 2) Como replicar as figuras e navegar nas explicações

- **Importâncias globais**: abra `outputs/graficos_importancias_top20.png` para ver quais variáveis gerais mais pesaram no modelo (RandomForest).
- **Explicações locais (LIME)**: abra `outputs/lime_explanation_*.html` no navegador (duplo clique).  
  Ali você verá:
  - Barras **verdes** empurrando para **Bad** (1) ou **vermelhas** para **Good** (0) — cores podem variar conforme tema;
  - os **valores das features** da pessoa específica;
  - a **predição local** e a probabilidade do modelo.

**Dica de leitura para um gerente:** escolha um caso “Bad” e aponte 3–5 fatores de maior peso no HTML do LIME; relacione com política de crédito (ex.: valor alto + histórico crítico + poucas garantias).

---

# 3) Contexto do problema, decisões e limitações (Parte Teórica)

## 3.1 Problema e Objetivos
- **Problema:** classificar clientes como bom (0) ou mau (1) risco de crédito e **explicar cada decisão** para clientes e compliance.
- **Objetivo:** construir um modelo preditivo **preciso** e **explicável**, gerando **explicações locais** (cliente a cliente) com **LIME**.

## 3.2 Modelo e Pipeline
- **Dados:** UCI Statlog (German Credit Data). 1.000 amostras, 20 atributos (mistura de categóricos e numéricos).  
- **Pré-processamento:** `OneHotEncoder` para categóricos (`handle_unknown='ignore'`) + `StandardScaler` para numéricos.
- **Modelo:** `RandomForestClassifier (class_weight='balanced_subsample')` — robusto, lida bem com não linearidades e interação de variáveis.
- **Métricas:** AUC, relatório de classificação e matriz de confusão.

## 3.3 LIME — como funciona (resumo)
O **LIME** aproxima o modelo **localmente** ao redor de uma instância por um modelo **interpretable** (linear), atribuindo pesos às features que mais explicam a predição naquele ponto.  
Pontos chave:
- Dependente da **vizinhaça local** amostrada e da **similaridade** definida;
- Explicações **estocásticas** (mudam um pouco entre execuções);
- Funciona **agnóstico ao modelo** (qualquer classificador).

## 3.4 Como usar as explicações na prática
- **Atendimento ao cliente**: explicar, em linguagem simples, os motivos da negação e que mudanças poderiam aumentar a chance de aprovação (ex.: renda maior, histórico sem atrasos).
- **Compliance**: registrar o *rationale* da decisão por caso, junto com controles de **viés** (auditar correlação com sexo/estado civil).
- **Produto de crédito**: revisar regras e limites quando o LIME indicar padrões recorrentes de negação em perfis específicos.

## 3.5 Limitações, riscos e mitigação
- **Estabilidade**: LIME pode variar em execuções diferentes. Use `random_state` e documente a versão do modelo.
- **Correlações**: explicações podem ser influenciadas por features correlacionadas — interprete em conjunto com análise global (SHAP/Permutation Importance).
- **Amostras fora da distribuição**: a vizinhança sintética do LIME pode gerar pontos pouco realistas. Mitigue com *discretize_continuous=True* e análise de sensibilidade.
- **Dados antigos e codificação**: a UCI aponta que a versão histórica tem **idiossincrasias de codificação**; existe um *South German Credit* atualizado. Use com cautela em decisões reais.
- **Justiça/Ética (LGPD)**: evite usar atributos sensíveis direta/indiretamente; ofereça canais de recurso e transparência sobre fatores acionáveis.

---

# 4) Publicar no GitHub (Parte Prática)

## Opção A — Interface Web (mais simples)
1. Crie conta em https://github.com e faça login.
2. Clique em **New** (canto esquerdo, ao lado de *Repositories*).
3. Em **Repository name**, digite: `xai-lime-credit` → **Create repository**.
4. Na página do repositório, clique em **Add file** → **Upload files**.
5. **Arraste** toda a pasta `xai-lime-credit` (ou compacte e envie o `.zip`) para a área de upload.
6. Desça a página e clique em **Commit changes**.
7. Copie a URL do repositório para enviar ao professor.

## Opção B — GitHub Desktop (gui)
1. Baixe https://desktop.github.com/ e instale.
2. **File → New repository** → Nome: `xai-lime-credit` → **Create repository**.
3. Copie os arquivos do projeto para essa pasta (arraste via Explorer).
4. Em **GitHub Desktop**, escreva uma mensagem e clique **Commit to main** → **Push origin**.

## Opção C — Linha de comando (git)
```bash
git init
git add .
git commit -m "XAI com LIME (German Credit)"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/xai-lime-credit.git
git push -u origin main
```

---

# 5) Roteiro do Vídeo Pitch (até 4 minutos)

1. **Abertura (20s)** — Problema: “Por que meu crédito foi negado?” e importância de explicabilidade.
2. **Dataset e modelo (40s)** — German Credit (UCI), 1.000 linhas, 20 atributos; RandomForest + pipeline de pré-processamento.
3. **LIME (60–90s)** — o que é e como estamos usando (explicações locais); mostrar 1 HTML do `outputs/`.
4. **Resultados (40s)** — 1–2 métricas (AUC, matriz de confusão) e 1 gráfico de importâncias globais.
5. **Discussão (30–40s)** — limitações do LIME, vieses, uso responsável (LGPD/compliance).
6. **Fecho (10s)** — onde está o código (GitHub) e como reproduzir.

---

# 6) Referências

- UCI Statlog (German Credit Data) — página do dataset (arquivos `german.data`, `german.data-numeric`, custo 1=Good, 2=Bad).
- Ribeiro, Singh, Guestrin (2016). *Why Should I Trust You?* (Artigo LIME).
- Documentação oficial do LIME.

> Nota: A UCI também disponibiliza o **South German Credit** com correções e contexto; útil para estudos mais rigorosos.
