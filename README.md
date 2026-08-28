# Atividade — Fluxo básico com scikit-learn

## Registro explicativo

Nesta atividade foi criado um modelo de aprendizado supervisionado para prever se um aluno será aprovado ou reprovado.
Os dados de entrada (`X`) representam horas de estudo e número de faltas de cada aluno.
A variável `y` representa o resultado conhecido: `0` para reprovado e `1` para aprovado.
Os dados foram separados em conjuntos de treino e teste usando `train_test_split`.
Essa separação é importante porque testar com os mesmos dados usados no treinamento pode gerar uma avaliação enganosa.
Foi escolhida a Regressão Logística por ser um modelo simples e adequado para classificação binária.
Treinar significa ajustar os parâmetros do modelo a partir dos exemplos disponíveis no conjunto de treino.
Depois do treinamento, o modelo recebeu os dados de teste para gerar previsões sem utilizar seus resultados durante o aprendizado.
A avaliação foi feita com a acurácia, que representa a proporção de previsões corretas em relação ao total de previsões.
Neste conjunto pequeno, a acurácia obtida foi de 100% (2 acertos em 2 exemplos de teste).
Esse resultado não garante o mesmo desempenho em novos alunos, pois a quantidade de dados é muito pequena.
Além disso, o modelo considera somente horas de estudo e faltas, ignorando outros fatores que podem influenciar a aprovação.
Em uma aplicação real, seriam necessários mais dados, validação mais robusta e outras métricas de avaliação.
Portanto, este exercício é apenas um primeiro passo para compreender o fluxo de um projeto de Machine Learning.

## Como executar

Instale a dependência:

```bash
pip install -r requirements.txt
```

Execute:

```bash
python atividade_scikit_learn.py
```
