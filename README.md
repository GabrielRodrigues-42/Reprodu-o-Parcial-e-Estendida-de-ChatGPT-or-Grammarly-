# Reprodução Parcial e Estendida de "ChatGPT or Grammarly?"

Kit de reprodução do trabalho **"Reprodução Parcial e Estendida de 'ChatGPT or Grammarly?': Comparando Sistemas de Correção Gramatical em Múltiplos Benchmarks"**, desenvolvido para a disciplina de Reprodutibilidade em PLN (UFCG, 2026).

Este repositório reproduz e estende o estudo de Wu et al. (2023), comparando seis sistemas de Correção de Erros Gramaticais (GEC) — GECToR, Grammarly, LanguageTool, ChatGPT, Claude e DeepSeek — sobre três benchmarks (CoNLL-2014, BEA-2019, JFLEG), com repetições, testes estatísticos formais e intervalos de confiança ausentes no trabalho original.

> **Autor:** Gabriel Oliveira Rodrigues — Universidade Federal de Campina Grande
> **Contato:** gabriel.rodrigues@copin.ufcg.edu.br

---

## Sumário

- [Visão geral do experimento](#visão-geral-do-experimento)
- [Ambiente e dependências](#ambiente-e-dependências)
- [Como reproduzir](#como-reproduzir)
- [Sobre o Grammarly (consulta manual)](#sobre-o-grammarly-consulta-manual)
- [Dados](#dados)
- [Resultados](#resultados)
- [Hipóteses e onde encontrá-las nos resultados](#hipóteses-e-onde-encontrá-las-nos-resultados)
- [Limitações conhecidas](#limitações-conhecidas)
- [Citação](#citação)
- [Licença](#licença)

---

## Visão Geral do Experimento

| | |
|---|---|
| **Sistemas avaliados** | GECToR, Grammarly, LanguageTool, ChatGPT (gpt-4o), Claude (claude-sonnet-4-6), DeepSeek (deepseek-chat / DeepSeek V3) |
| **Datasets** | CoNLL-2014, BEA-2019, JFLEG |
| **Amostragem** | 200 sentenças por dataset, seed fixo, estratificadas por comprimento (curtas/médias/longas) |
| **Repetições** | 3 execuções por sentença para sistemas estocásticos (ChatGPT, Claude, DeepSeek); execução única para sistemas determinísticos (GECToR, LanguageTool) e para o Grammarly (consulta manual) |
| **Métricas** | Precisão, Recall, F0.5 (CoNLL-2014, BEA-2019) e GLEU (JFLEG) |
| **Scorers** | ERRANT (CoNLL-2014, BEA-2019) e GLEU scorer oficial (JFLEG) |
| **Análise estatística** | Shapiro-Wilk (normalidade), teste t pareado / Wilcoxon signed-rank, d de Cohen / r de Wilcoxon, IC 95%, coeficiente de variação |

O desenho experimental completo (variáveis independentes/dependentes/controladas, hipóteses H1–H5, protocolo de coleta) está descrito na Seção 2 do artigo (`docs/artigo.pdf`).

---

## Ambiente e Dependências

O experimento foi desenvolvido em um container Docker com dependências versionadas para garantir reprodutibilidade (evitando os problemas de incompatibilidade entre `torch`/`transformers` encontrados durante o desenvolvimento).

```bash
# Construir a imagem
docker build -t gec-reproducao .

# Rodar o container
docker run -it --rm -v $(pwd):/workspace gec-reproducao
```

Principais dependências (ver `requirements.txt` para versões exatas fixadas):

- Python 3.10+
- `torch`, `transformers` (para GECToR — rodando em modo CPU via `map_location` patch)
- `errant` (scoring CoNLL-2014 / BEA-2019)
- `language-tool-python` (LanguageTool)
- `anthropic`, `openai` (APIs de ChatGPT, Claude e DeepSeek — DeepSeek usa endpoint compatível com a API OpenAI)
- `scipy`, `pandas`, `numpy` (análise estatística)
- `pyyaml`

Chaves de API necessárias (não incluídas no repositório — configurar como variáveis de ambiente):

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

---

## Como Reproduzir

1. **Configurar o ambiente** (Docker, conforme acima, lembrando de baixar e adicionar GECToR, não incluído neste repositório) e exportar as chaves de API.
2. **Baixar os datasets originais** — CoNLL-2014, BEA-2019 e JFLEG não são redistribuídos neste repositório por questões de licenciamento; siga as instruções em `data/README.md` para obtê-los das fontes oficiais.
3. **Gerar (ou reutilizar) a amostra estratificada**:
   ```bash
   python scripts/sample_sentences.py --seed 2026 --n 200 --output data/raw/sampled_sentences.csv
   ```
   A amostra já utilizada neste trabalho está disponível em `data/raw/sampled_sentences.csv` para reprodução exata com o mesmo seed.
4. **Executar os sistemas automatizados**:
   ```bash
   bash scripts/run_pipeline.sh
   ```
   Isso executa GECToR e LanguageTool (execução única) e ChatGPT/Claude/DeepSeek (3 execuções por sentença) sobre os três datasets, salvando as saídas brutas em `results/scores.csv`.
5. **Consultar o Grammarly manualmente** — ver seção [Sobre o Grammarly](#sobre-o-grammarly-consulta-manual) abaixo.
6. **Rodar a análise estatística**:
   ```bash
   python scripts/stats_analysis.py
   ```
   Gera `results/stats_summary.csv`, `results/pairwise_tests.csv` e `results/hypothesis_results.csv`.

---

## Sobre o Grammarly (Consulta Manual)

O Grammarly não possui API pública, então as 200 sentenças do CoNLL-2014 foram consultadas manualmente através da interface web (`app.grammarly.com`, conta Grammarly Pro), seguindo um protocolo iterativo até estabilização (mesma metodologia do artigo original de Wu et al.).

O arquivo `data/raw/grammarly_queue.xlsx` contém:

- **Aba `Instructions`**: protocolo passo a passo para reprodução da consulta manual.
- **Aba `Grammarly_Queue`**: fila de trabalho com as 200 sentenças, colunas de iteração (`Iteration_1`–`Iteration_5`), correção final, contagem de iterações, e classificação de cada correção como `R` (reescrita/*rewrite*) ou `B` (básica/*basic*), com colunas calculadas automaticamente via fórmula (`=COUNTA(...)`, `=COUNTIF(...)`).

Por depender de fórmulas do Excel, recomenda-se também consultar `results/grammarly_final.csv` (exportação estática das colunas finais) para uso fora do Excel/LibreOffice. Grammarly foi avaliado apenas no CoNLL-2014, dado o esforço manual envolvido; não foi avaliado em BEA-2019 nem JFLEG.

---

## Dados

| Arquivo | Descrição |
|---|---|
| `data/sampled_sentences.csv` | 200 sentenças por dataset (600 no total), com `dataset`, `original_id`, `sentence`, `word_count` e `length_bucket` (curta/média/longa) |
| `data/grammarly_queue.xlsx` | Fila e protocolo de consulta manual ao Grammarly (ver seção acima) |

Os datasets originais (CoNLL-2014, BEA-2019, JFLEG) devem ser obtidos das fontes oficiais listadas em `data/README.md` — não são redistribuídos aqui.

---

## Resultados

| Arquivo | Conteúdo |
|---|---|
| `results/scores.csv` | Saída bruta (TP, FP, FN, precisão, recall, F0.5/GLEU) por sistema, dataset e execução |
| `results/stats_summary.csv` | Média, desvio padrão, CV, IC 95% e teste de Shapiro-Wilk por sistema/dataset |
| `results/pairwise_tests.csv` | Testes pareados (t de Student ou Wilcoxon signed-rank) entre todos os pares de sistemas, com tamanho de efeito |
| `results/hypothesis_results.csv` | Veredito (suportada / parcialmente suportada / rejeitada) para cada uma das hipóteses H1–H5, com justificativa |

Os principais achados estão consolidados nas Tabelas 2–7 do artigo (`docs/artigo.pdf`), incluindo a hierarquia de desempenho entre sistemas, a análise de *over-correction*, a variabilidade estocástica das LLMs e a generalização entre datasets.

---

## Hipóteses e Onde Encontrá-las nos Resultados

| Hipótese | Resumo | Veredito | Seção do artigo |
|---|---|---|---|
| H1 | Sistemas determinísticos (GECToR, Grammarly) superam o ChatGPT no CoNLL-2014 | Parcialmente suportada | 3.2, 3.2.1 |
| H2 | LLMs diferentes têm padrões distintos de *over-correction* | Parcialmente suportada | 3.3 |
| H3 | LLMs apresentam variabilidade estocástica entre execuções | Suportada | 3.4 |
| H4 | LLMs degradam em sentenças longas | Rejeitada | 3.5 |
| H5 | Ranqueamento relativo dos sistemas é consistente entre datasets | Suportada | 3.6 |

Ver `results/hypothesis_results.csv` para a justificativa quantitativa de cada veredito.

---

## Limitações Conhecidas

- **Amostra de n=3 execuções por sistema estocástico**: limita o poder estatístico dos testes de hipótese, especialmente do Wilcoxon signed-rank contra sistemas determinísticos (ver Seção 3.2.1 e 4.2 do artigo).
- **Grammarly avaliado apenas no CoNLL-2014**, por depender de consulta manual sem API.
- **Prompt único e fixo** para os três LLMs, o que pode favorecer sistematicamente modelos mais alinhados ao estilo do prompt escolhido (ver Seção 4.2).
- **Sem taxonomia de tipo de erro**: a análise de *over-correction* (H2) usa um proxy indireto (recall − precisão) em vez de categorização manual por tipo de erro.
Discussão completa em `docs/artigo.pdf`, Seção 4 (Ameaças à Validade).

---

## Citação

Se este trabalho for útil para sua pesquisa, por favor cite:

```bibtex
@misc{rodrigues2026reproducao,
  author = {Rodrigues, Gabriel Oliveira},
  title  = {Reprodução Parcial e Estendida de "ChatGPT or Grammarly?": Comparando Sistemas de Correção Gramatical em Múltiplos Benchmarks},
  year   = {2026},
  institution = {Universidade Federal de Campina Grande}
}
```

Trabalho original reproduzido:

> Wu, H., Wang, W., Wan, Y., Jiao, W., Lyu, M. R. (2023). ChatGPT or Grammarly? Evaluating ChatGPT on Grammatical Error Correction Benchmark. *arXiv preprint arXiv:2303.13648*.
