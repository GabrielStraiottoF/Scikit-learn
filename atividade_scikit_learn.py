"""
Atividade: Ferramenta scikit-learn - fluxo básico de Machine Learning.
Contexto: prever aprovação ou reprovação de alunos a partir de horas de estudo e faltas.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Etapa 1 - Preparação dos dados
# X contém as características (features) usadas pelo modelo:
# coluna 1 = horas de estudo e coluna 2 = número de faltas.
X = [
    [2, 1],
    [4, 0],
    [1, 3],
    [3, 1],
    [5, 0],
    [2, 2],
]

# y contém o resultado conhecido de cada aluno (rótulo/target):
# 0 = reprovado e 1 = aprovado.
y = [0, 1, 0, 1, 1, 0]


# Etapa 2 - Separação em treino e teste
# train_test_split separa os dados para que o modelo aprenda com uma parte
# e seja avaliado em exemplos que não foram usados durante o treinamento.
# Não devemos treinar e testar com os mesmos dados porque isso pode esconder
# erros de generalização e produzir uma avaliação artificialmente otimista.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y,
)


# Etapa 3 - Criação do modelo
# A Regressão Logística foi escolhida por ser simples, supervisionada e adequada
# para uma classificação binária como aprovado (1) ou reprovado (0).
modelo = LogisticRegression(random_state=42)

# Treinar um modelo significa ajustar seus parâmetros usando os dados de treino,
# procurando aprender a relação entre as características X e os resultados y.


# Etapa 4 - Treinamento
# Os dados de treino são os exemplos que o algoritmo usa para aprender esse padrão.
modelo.fit(X_train, y_train)


# Etapa 5 - Avaliação
# Agora o modelo faz previsões sobre os dados de teste, que ficaram separados.
y_pred = modelo.predict(X_test)

# A acurácia indica a proporção de previsões corretas entre todas as previsões.
acuracia = accuracy_score(y_test, y_pred)

print("Resultados da avaliação")
print(f"Dados de teste: {X_test}")
print(f"Resultados reais: {y_test}")
print(f"Previsões: {y_pred.tolist()}")
print(f"Acurácia: {acuracia:.2%}")


# Etapa 6 - Análise final
# A acurácia deste pequeno conjunto de teste foi de 100%, mas isso NÃO significa
# que o modelo terá 100% de acerto em novos alunos. A amostra é muito pequena,
# portanto o resultado tem alta incerteza. Este exercício representa apenas um
# primeiro passo: modelos reais precisam de mais dados, validação e análise.
