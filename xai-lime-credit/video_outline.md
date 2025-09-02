# Roteiro sugerido (4 minutos)

**Slide 1 — Título**
- Decifrando a Caixa-Preta: LIME no Crédito Bancário
- Seu nome | Disciplina | Data

**Slide 2 — Problema**
- Clientes e reguladores pedem “por quê?”
- Precisamos de explicações por caso (transparência + compliance).

**Slide 3 — Dados e Modelo**
- UCI German Credit: 1.000 clientes, 20 atributos (mistos).
- Pipeline: OneHot para categóricos + StandardScaler para numéricos.
- Classificador: RandomForest (robusto e não-linear).

**Slide 4 — O que é LIME?**
- Modelo linear local sobre a vizinhança da instância.
- Explica contribuição (±) das features na predição.

**Slide 5 — Demonstração**
- Abrir 1 HTML do LIME (outputs/lime_explanation_*.html).

**Slide 6 — Resultados**
- AUC e matriz de confusão.
- Importâncias globais (gráfico top 20).

**Slide 7 — Limitações e Ética**
- Variação estocástica, correlação, OOD.
- LGPD, atributos sensíveis, canal de recurso.

**Slide 8 — Reprodutibilidade**
- Como rodar: run.bat / pip -r / python src/train_and_explain.py
- Link do GitHub.
