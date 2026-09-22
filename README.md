# Olist NLP Audit

Auditoria de técnicas de Processamento de Linguagem Natural aplicada a reviews reais de e-commerce em português brasileiro.

## Contexto

Este projeto foi desenvolvido como trabalho prático do módulo de NLP da Pós-Tech em AI Scientist.

O objetivo é aplicar e avaliar diferentes abordagens de NLP sobre o dataset público Brazilian E-Commerce Public Dataset by Olist, utilizando avaliações escritas por consumidores brasileiros.

O projeto percorre diferentes etapas de um pipeline de NLP, desde o tratamento do texto bruto até classificação supervisionada, embeddings, reconhecimento de entidades e modelagem de tópicos.

---

## Objetivo

Investigar como diferentes técnicas de NLP podem ser utilizadas para compreender e estruturar feedback textual de consumidores.

O projeto busca responder questões como:

- É possível classificar automaticamente o sentimento de um review?
- Qual representação funciona melhor como baseline: Bag of Words ou TF-IDF?
- Como Naive Bayes, Regressão Logística e SVM se comportam nesse problema?
- Representações semânticas agregam valor em relação às abordagens clássicas?
- Quais entidades podem ser extraídas automaticamente dos comentários?
- Quais assuntos aparecem naturalmente nos reviews sem utilizar rótulos?
- Quais limitações específicas surgem ao trabalhar com português brasileiro?

---

## Dataset

Será utilizado o Brazilian E-Commerce Public Dataset by Olist.

Para NLP, a principal fonte será a tabela de avaliações dos pedidos, contendo informações como:

- nota da avaliação;
- título do review;
- comentário escrito pelo consumidor;
- data da avaliação.

O dataset original não será versionado neste repositório.

---

## Etapas do projeto

### 1. Entendimento dos dados

Análise do corpus, valores ausentes, distribuição das avaliações, tamanho dos comentários e definição do problema.

### 2. Pré-processamento

Avaliação de técnicas de limpeza e normalização de texto, tokenização, stopwords e particularidades do português brasileiro.

### 3. Vetorização clássica

Experimentação com:

- Bag of Words;
- n-grams;
- TF-IDF.

### 4. Modelagem supervisionada

Construção e comparação de modelos para análise de sentimento:

- Multinomial Naive Bayes;
- Logistic Regression;
- Linear SVM.

Avaliação utilizando precision, recall, F1-score, matriz de confusão e validação cruzada.

### 5. Embeddings e representações semânticas

Exploração de embeddings pré-treinados e comparação entre representações baseadas em frequência e representações semânticas.

### 6. Named Entity Recognition

Aplicação de NER em português para identificar entidades nos reviews e análise das limitações do modelo para o domínio de e-commerce.

### 7. Modelagem de tópicos

Descoberta dos principais assuntos presentes nos comentários utilizando abordagens clássicas e modernas, como LDA/NMF e BERTopic.

### 8. Auditoria de NLP

Avaliação das seis dimensões propostas no trabalho:

1. Fundamentos e definição do problema;
2. Pré-processamento;
3. Vetorização clássica;
4. Embeddings e representações semânticas;
5. Modelagem supervisionada e NER;
6. Análise não supervisionada e interpretação.

### 9. Gaps e plano de ação

Identificação das principais limitações encontradas e proposta de melhorias de curto, médio e longo prazo, considerando:

- qualidade;
- custo computacional;
- interpretabilidade;
- disponibilidade de dados;
- complexidade operacional.

---

## Resultado esperado

Ao final, o projeto deverá fornecer uma visão comparativa das técnicas estudadas e demonstrar como diferentes abordagens de NLP podem ser combinadas para transformar reviews de consumidores em informações úteis para análise e tomada de decisão.

O objetivo não é encontrar um algoritmo universalmente melhor, mas compreender os trade-offs entre diferentes representações, modelos, custos e níveis de interpretabilidade.
