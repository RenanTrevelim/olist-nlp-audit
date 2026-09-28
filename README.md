# Olist NLP Audit — Análise de Sentimentos, Embeddings & Modelagem de Tópicos

Projeto de Ciência de Dados para compreender avaliações de consumidores brasileiros por meio de **Processamento de Linguagem Natural, Machine Learning e análise de tópicos**.

O trabalho percorre a preparação dos dados, a comparação de representações textuais, o treinamento e a otimização de classificadores, a análise de generalização e a investigação dos assuntos presentes nos comentários.

Na auditoria final, **BoW com n-gramas + SVM alcançou F1 macro de 0,9188 e acurácia de 93,13%**. Os **embeddings + SVM identificaram mais avaliações negativas**, evidenciando que a escolha também depende do tipo de erro que se deseja reduzir.

Projeto desenvolvido como trabalho prático do módulo de NLP da **Pós-Tech em AI Scientist**.

---

## Objetivo do Projeto

> Como transformar avaliações de e-commerce em informações sobre satisfação dos clientes e temas recorrentes, comparando abordagens clássicas e representações semânticas?

Para responder a essa pergunta, foram exploradas:

- **TF-IDF e Bag of Words com n-gramas**, para representar palavras e suas combinações;
- **embeddings de sentenças**, para representar características semânticas dos comentários;
- **Naive Bayes, Regressão Logística e SVM**, para classificação de sentimentos;
- **LDA e NMF**, para descoberta de tópicos sem utilizar rótulos de sentimento no treinamento.

---

## Dataset

O projeto utiliza a tabela de avaliações do [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), no arquivo `olist_order_reviews_dataset.csv`.

| Etapa de preparação | Quantidade |
|---|---:|
| Registros na base original | 99.224 |
| Avaliações com comentário preenchido | 40.977 |
| Avaliações após excluir a nota 3 | **37.420** |
| Avaliações positivas | 26.530 |
| Avaliações negativas | 10.890 |

As variáveis centrais são `review_comment_message`, que contém o texto, e `review_score`, que contém a nota. A coluna `sentimento` é criada a partir dessa nota.

**Os rótulos são derivados das avaliações numéricas, não de uma anotação manual dos textos.** Portanto, podem existir divergências entre o conteúdo de um comentário e o sentimento atribuído pela regra.

---

## Estrutura do Projeto

```text
olist-nlp-audit/
├── assets/
│   └── images/                       # Gráficos apresentados neste README
├── data/
│   ├── olist_order_reviews_dataset.csv
│   ├── dataset_tratado.csv
│   └── embedding/                    # Matrizes de embeddings salvas em .npy
├── models/
│   ├── modelo_tfidf_final.pkl
│   ├── modelo_bow_ngram_final.pkl
│   └── modelo_embeddings_final.pkl
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_tfidf_models.ipynb
│   ├── 03_bow_ngram_models.ipynb
│   ├── 04_embeddings.ipynb
│   ├── 05_topic_modeling.ipynb
│   └── 06_final_nlp_audit.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

Os arquivos tratados, embeddings e modelos são produzidos ao executar os notebooks. Caso não estejam disponíveis na sua cópia do projeto, siga a sequência de execução ao final deste documento.

---

## 1. Validação e Preparação dos Dados

Notebook: [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb)

A preparação inclui inspeção da estrutura, verificação de valores ausentes e registros duplicados, seleção das colunas e remoção de avaliações sem comentário. Não foram encontrados registros inteiramente duplicados na verificação inicial; isso não significa ausência de textos repetidos.

### 1.1. Distribuição das Notas

A nota **5** predomina entre os comentários disponíveis. Essa concentração já indica que as classes de sentimento terão tamanhos diferentes.

![Distribuição das notas das 40.977 avaliações com comentário](assets/images/01_distribuicao_notas.png)

### 1.2. Definição dos Sentimentos

| Nota | Sentimento | Classe |
|---|---|---:|
| 1 ou 2 | Negativo | 0 |
| 3 | Excluída do problema binário | — |
| 4 ou 5 | Positivo | 1 |

A base final contém aproximadamente **70,9% de avaliações positivas** e **29,1% de negativas**. Esse desequilíbrio motiva o uso de estratificação, pesos de classe e métricas macro.

![Distribuição dos sentimentos na base preparada](assets/images/02_distribuicao_sentimentos.png)

O resultado da preparação é salvo em `data/dataset_tratado.csv`, utilizado pelos demais experimentos.

---

## 2. Pré-processamento dos Textos

As etapas de limpeza incluem conversão para minúsculas, remoção de HTML, URLs e e-mails, expansão de abreviações e normalização de espaços.

O tratamento é adaptado à representação utilizada:

| Abordagem | Tratamento aplicado |
|---|---|
| TF-IDF e BoW | Tokenização, remoção de stopwords e stemização com RSLP |
| Embeddings | Preservação das palavras completas, stopwords e pontuação básica |
| Modelagem de tópicos | Tokenização e remoção de stopwords, sem stemização |

As listas de stopwords preservam termos de negação como **não, nunca e sem**, relevantes para interpretar insatisfação. Nos modelos lexicais, a stemização reduz as palavras a radicais; por exemplo, “produto” aparece como “produt” nos textos tratados.

---

## 3. Estratégia de Treinamento e Avaliação

Os experimentos supervisionados utilizam divisão estratificada com `random_state=42`:

```text
37.420 avaliações
        ↓
Treinamento: 26.194 (70%)
Teste:       11.226 (30%)
```

A otimização utiliza **GridSearchCV**, **StratifiedKFold com 5 divisões** e **F1 macro** como critério de seleção. Para TF-IDF e BoW, o vetorizador integra o pipeline avaliado na validação cruzada. Os embeddings são gerados por um modelo pré-treinado mantido fixo.

| Métrica | O que permite observar |
|---|---|
| Acurácia | Proporção total de classificações corretas |
| Precisão por classe | Quantas previsões daquela classe estavam corretas |
| Recall por classe | Quantos exemplos reais daquela classe foram identificados |
| F1 macro | Média do F1 das classes, com o mesmo peso para cada uma |
| Matriz de confusão | Quantidade e direção dos erros de classificação |

O teste também foi consultado durante a comparação dos modelos. Por isso, a auditoria final consolida resultados conhecidos e **não representa uma avaliação independente em dados novos**.

---

## 4. Modelos com TF-IDF

Notebook: [02_tfidf_models.ipynb](notebooks/02_tfidf_models.ipynb)

O TF-IDF atribui pesos aos termos considerando sua frequência no documento e sua ocorrência no conjunto de textos. A configuração utiliza **unigramas e bigramas**, frequência mínima de 5 documentos, frequência máxima de 85% e limite de 10.000 características, com `sublinear_tf=True`.

Foram comparados **Naive Bayes Multinomial, Regressão Logística e SVM Linear**. O SVM selecionado utiliza **C = 0,5** e é salvo junto ao vetorizador em `modelo_tfidf_final.pkl`.

Na auditoria final, essa abordagem alcançou **93,10% de acurácia** e **F1 macro de 0,9186**.

---

## 5. Modelos com Bag of Words e N-gramas

Notebook: [03_bow_ngram_models.ipynb](notebooks/03_bow_ngram_models.ipynb)

O `CountVectorizer` transforma os comentários em contagens de **palavras e pares de palavras**. Utilizamos os mesmos limites de frequência e tamanho máximo do vocabulário adotados no TF-IDF.

Também foram comparados Naive Bayes, Regressão Logística e SVM. O **SVM otimizado com C = 0,01** foi escolhido nesta etapa por priorizar a identificação de avaliações negativas, embora a Regressão Logística tenha apresentado ligeiramente menos erros totais.

### 5.1. Curva de Aprendizado

O F1 macro de validação cresce com o número de exemplos e termina em aproximadamente **0,915**, frente a **0,928 no treino**. A diferença de **0,013** sugere generalização razoável nessa validação, com ganhos progressivamente menores nas maiores amostras.

![Curva de aprendizado do SVM com Bag of Words e n-gramas](assets/images/03_curva_aprendizado_bow.png)

Na auditoria final, o pipeline salvo em `modelo_bow_ngram_final.pkl` alcançou **93,13% de acurácia** e **F1 macro de 0,9188**.

---

## 6. Modelos com Embeddings

Notebook: [04_embeddings.ipynb](notebooks/04_embeddings.ipynb)

Utilizamos o modelo **paraphrase-multilingual-MiniLM-L12-v2**, por meio de Sentence Transformers, para transformar cada comentário em um vetor normalizado de **384 dimensões**. Seus pesos não são ajustados neste projeto.

```text
Comentário tratado
        ↓
Sentence Transformer pré-treinado
        ↓
Embedding normalizado de 384 dimensões
        ↓
Regressão Logística ou SVM
        ↓
Sentimento previsto
```

Após otimização, o **SVM com C = 10** apresentou F1 macro médio de **0,9112 na validação cruzada**, frente a **0,9100** da Regressão Logística.

### 6.1. Curva de Aprendizado

Com mais exemplos, as curvas de treino e validação se aproximam. Os resultados impressos, arredondados a duas casas, são **0,92 no treino**, **0,91 na validação** e diferença de **0,01**. O gráfico mostra redução dessa diferença e estabilização gradual.

![Curva de aprendizado do SVM com embeddings](assets/images/04_curva_aprendizado_embeddings.png)

Na auditoria final, os embeddings alcançaram **92,78% de acurácia**, **F1 macro de 0,9159** e **recall de aproximadamente 95% para avaliações negativas**.

---

## 7. Comparação Final das Representações

Notebook: [06_final_nlp_audit.ipynb](notebooks/06_final_nlp_audit.ipynb)

A comparação abaixo reúne os **três SVMs selecionados** nos experimentos. Ela não inclui todas as configurações de Naive Bayes e Regressão Logística testadas anteriormente.

| Modelo | Acurácia | Precisão macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| TF-IDF + SVM | 93,10% | 0,9088 | 0,9307 | 0,9186 |
| **BoW + n-gramas + SVM** | **93,13%** | **0,9098** | 0,9296 | **0,9188** |
| Embeddings + SVM | 92,78% | 0,9031 | **0,9335** | 0,9159 |

![Comparação do F1 macro e dos tipos de erro dos três modelos](assets/images/08_comparacao_modelos.png)

O BoW com n-gramas apresentou o maior F1 macro, mas a diferença para TF-IDF é de apenas **0,0002** nos valores arredondados. Sem uma análise de incerteza, essa diferença não comprova superioridade estatística.

### 7.1. Matrizes de Confusão

Nas matrizes, **0 representa negativo** e **1 representa positivo**.

![Matrizes de confusão dos três modelos finais](assets/images/06_matrizes_finais.png)

| Modelo | Negativas previstas como positivas | Positivas previstas como negativas | Total de erros |
|---|---:|---:|---:|
| TF-IDF + SVM | 229 | 546 | 775 |
| BoW + n-gramas + SVM | 243 | **528** | **771** |
| Embeddings + SVM | **173** | 637 | 810 |

**Para priorizar o F1 macro**, BoW com n-gramas é a referência de maior desempenho observado. **Para deixar passar menos insatisfações**, embeddings é uma alternativa relevante: identifica mais avaliações negativas, mas também gera mais falsos alertas sobre comentários positivos.

---

## 8. Modelagem de Tópicos

Notebook: [05_topic_modeling.ipynb](notebooks/05_topic_modeling.ipynb)

Além de prever sentimentos, investigamos **sobre o que os consumidores escrevem**. Foram utilizados LDA sobre contagens de palavras e NMF sobre TF-IDF, sem utilizar os rótulos de sentimento no ajuste dos tópicos.

### 8.1. LDA e Número de Tópicos

| Número de tópicos | Coerência c_v |
|---|---:|
| 3 | 0,4323 |
| **5** | **0,4701** |
| 7 | 0,4513 |

A configuração com **5 tópicos** apresentou a maior coerência entre as alternativas testadas. Essa métrica auxilia a análise da associação entre palavras; não representa acurácia de classificação.

![Coerência do LDA para três, cinco e sete tópicos](assets/images/05_coerencia_lda.png)

### 8.2. Interpretação dos Tópicos do NMF

O NMF utiliza uma matriz TF-IDF com **37.420 documentos e 3.158 termos**, organizada em cinco componentes. Sua coerência foi de **0,469**, próxima à do LDA, sem estabelecer uma vantagem conclusiva entre as abordagens.

Os tópicos receberam nomes a partir das palavras mais relevantes e da inspeção de comentários representativos:

| Tema interpretado | Avaliações | Sentimento negativo |
|---|---:|---:|
| Problemas de Recebimento | 14.802 | 64,5% |
| Qualidade e Satisfação | 8.222 | 8,0% |
| Entrega e Prazo | 7.464 | 5,5% |
| Recomendação e Experiência com a loja | 3.858 | 4,2% |
| Produto, preço e atendimento | 3.074 | 3,9% |

![Distribuição dos tópicos do NMF e proporção dos sentimentos em cada grupo](assets/images/07_topicos_sentimentos.png)

O grupo **Problemas de Recebimento** concentra a maior parcela de avaliações e apresenta predominância negativa, sugerindo uma prioridade de investigação. Entretanto, **35,5% dos comentários desse grupo são positivos**: o nome é uma interpretação temática, não uma confirmação de falha de entrega.

Cada documento recebe o tópico de maior peso. Esse peso não é uma probabilidade calibrada, e comentários curtos ou com vários assuntos podem exigir revisão manual.

---

## 9. Artefatos e Reutilização

| Arquivo | Conteúdo salvo | Entrada necessária |
|---|---|---|
| `modelo_tfidf_final.pkl` | Vetorizador TF-IDF e SVM | Texto já pré-processado |
| `modelo_bow_ngram_final.pkl` | CountVectorizer e SVM | Texto já pré-processado |
| `modelo_embeddings_final.pkl` | Classificador SVM | Embeddings gerados pelo mesmo modelo e normalização |

Os modelos são serializados com **Joblib**. A limpeza textual é executada separadamente; o classificador de embeddings também depende do Sentence Transformer. Portanto, os arquivos salvos ainda não formam um único fluxo de previsão a partir do texto bruto.

As matrizes de embeddings são armazenadas em `data/embedding/`. Sua reutilização exige manter a correspondência entre a ordem dos vetores, os textos e os rótulos.

---

## 10. Possíveis Aplicações

Os resultados podem apoiar a triagem de avaliações negativas, a identificação de temas recorrentes e o acompanhamento de problemas relatados pelos consumidores.

Uma aplicação futura poderia seguir o fluxo:

```text
Novas avaliações
        ↓
Validação e pré-processamento
        ↓
Classificação de sentimento + identificação de temas
        ↓
Revisão de casos ambíguos
        ↓
Dashboard de satisfação e prioridades de atendimento
```

Essa arquitetura é uma possibilidade de evolução. O escopo implementado neste repositório está nos notebooks e nos artefatos de modelagem.

---

## 11. Limitações e Próximas Evoluções

| Limitação observada | Próximo passo |
|---|---|
| Sentimentos derivados das notas e exclusão da nota 3 | Validar uma amostra manualmente e estudar a classe neutra |
| Teste consultado durante a comparação | Reservar novos dados para avaliação independente |
| Possibilidade de textos repetidos entre divisões | Auditar duplicatas textuais e avaliar separação por grupos |
| Pequenas diferenças entre modelos | Estimar incerteza e comparar previsões de forma pareada |
| Limpeza definida separadamente dos modelos | Centralizar o pré-processamento e verificar equivalência entre treino e inferência |
| Ausência de medição de latência e memória | Comparar custo de execução antes de uma escolha operacional |
| Tópicos atribuídos pelo maior peso | Revisar casos ambíguos, sem evidência temática ou com múltiplos assuntos |

Outras extensões incluem análise dos erros, integração com uma API ou dashboard, rastreamento de experimentos e monitoramento de mudanças no perfil dos comentários. **NER e BERTopic permanecem possibilidades futuras**, sem resultados apresentados nos seis notebooks atuais.

---

## 12. Tecnologias Utilizadas

**Python · Pandas · NumPy · NLTK · Scikit-learn · Sentence Transformers · Gensim · Matplotlib · Seaborn · Joblib · Jupyter Notebook**

---

## 13. Como Executar o Projeto

### 13.1. Ambiente

A partir da raiz do projeto, crie um ambiente virtual. O ambiente utilizado no desenvolvimento foi baseado em **Python 3.12**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Em Linux ou macOS, a ativação equivalente é `source .venv/bin/activate`. O arquivo de dependências reflete o ambiente Windows e inclui `pywinpty`; ambientes de outros sistemas podem exigir adaptação dessa dependência.

### 13.2. Dados e Recursos

Baixe o dataset na página da Olist indicada acima e disponibilize `olist_order_reviews_dataset.csv` na pasta `data/`. Garanta a existência das pastas de saída:

```powershell
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/embedding', 'models']]"
python -m nltk.downloader punkt punkt_tab stopwords rslp
```

A primeira utilização dos embeddings requer acesso à internet para baixar o modelo pré-treinado. O recurso `rslp` é necessário para a stemização em português.

### 13.3. Ordem dos Notebooks

| Ordem | Notebook | Entrega principal |
|---|---|---|
| 1 | [Preparação dos dados](notebooks/01_data_preparation.ipynb) | Base tratada e rótulos de sentimento |
| 2 | [Modelos TF-IDF](notebooks/02_tfidf_models.ipynb) | Pipeline TF-IDF + SVM |
| 3 | [Modelos BoW e n-gramas](notebooks/03_bow_ngram_models.ipynb) | Pipeline BoW + SVM |
| 4 | [Embeddings](notebooks/04_embeddings.ipynb) | Vetores e classificador SVM |
| 5 | [Modelagem de tópicos](notebooks/05_topic_modeling.ipynb) | Temas e relação com os sentimentos |
| 6 | [Auditoria final](notebooks/06_final_nlp_audit.ipynb) | Comparação das representações |

Para respeitar os caminhos relativos `../data` e `../models`, execute os notebooks com o diretório de trabalho em `notebooks/`:

```powershell
cd notebooks
python -m jupyter lab
```

Execute cada notebook do início ao fim com o kernel do ambiente configurado. A auditoria final depende dos três arquivos de modelos produzidos nas etapas anteriores.

---

## 14. Conclusão

O projeto combinou **classificação supervisionada** e **modelagem de tópicos** para analisar satisfação e assuntos recorrentes nas avaliações da Olist.

As representações clássicas apresentaram desempenho competitivo: **BoW com n-gramas obteve o maior F1 macro entre os SVMs finais**, com resultado muito próximo ao TF-IDF. Os **embeddings favoreceram a identificação de avaliações negativas**, ao custo de mais falsos alertas.

A análise de tópicos complementou os sentimentos ao destacar temas relacionados a entrega, recebimento, qualidade e recomendação. O principal aprendizado é que a utilidade de uma solução de NLP depende da qualidade dos rótulos, da validação e da interpretação dos erros, além da escolha do modelo.

---

## Autor

**Renan Assis Trevelim**

Projeto de portfólio e aplicação prática de **Data Science, NLP, Machine Learning, análise de sentimentos e modelagem de tópicos**.

**LinkedIn:** [Renan Assis Trevelim](https://www.linkedin.com/in/renan-trevelim)  
**GitHub:** [RenanTrevelim](https://github.com/RenanTrevelim)

---

<sub>Resultados extraídos das saídas salvas dos notebooks. Os gráficos de comparação e de tópicos foram organizados a partir desses mesmos resultados; as demais figuras foram exportadas diretamente dos notebooks.</sub>
