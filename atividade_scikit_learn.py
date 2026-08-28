"""
Atividade: Ferramenta scikit-learn - fluxo básico de Machine Learning.
Contexto: prever aprovação ou reprovação de alunos a partir de horas de estudo e faltas.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Etapa 1 - Preparação dos dados
# X contém as características (features): horas de estudo e número de faltas.
# y contém o resultado conhecido: 0 = reprovado e 1 = aprovado.
# Foram adicionados 200 alunos ao conjunto original, totalizando 206 registros.
dados = [
    (2, 1, 0), (4, 0, 1), (1, 3, 0), (3, 1, 1), (5, 0, 1), (2, 2, 0),
]

# Dados adicionais: 200 exemplos sintéticos para tornar o conjunto maior.
# A regra usada para gerar os rótulos é: maior estudo e menos faltas tendem a favorecer aprovação.
for aluno in range(200):
    horas = 1 + (aluno * 3) % 6
    faltas = (aluno * 2 + 1) % 5
    aprovado = 1 if horas >= 3 and faltas <= 2 else 0
    dados.append((horas, faltas, aprovado))

# DataFrame do pandas: tabela organizada para visualizar e manipular os dados.
df = pd.DataFrame(dados, columns=["horas_estudo", "faltas", "resultado"])

print("Tabela dos alunos:")
print(df.to_string(index=False))
print(f"\nTotal de alunos: {len(df)}")

X = df[["horas_estudo", "faltas"]]
y = df["resultado"]


# Etapa 2 - Separação em treino e teste
# O conjunto é dividido para que o modelo aprenda com uma parte e seja avaliado
# em exemplos que não foram usados durante o treinamento.
# Não devemos usar os mesmos dados para treino e teste, pois isso pode gerar uma
# avaliação artificialmente otimista e não medir corretamente a generalização.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Etapa 3 - Criação do modelo
# A Regressão Logística foi escolhida por ser simples, supervisionada e adequada
# para uma classificação binária: aprovado (1) ou reprovado (0).
modelo = LogisticRegression(random_state=42)

# Treinar significa ajustar os parâmetros do modelo usando os dados de treino,
# buscando aprender a relação entre as características e os resultados conhecidos.


# Etapa 4 - Treinamento
# Os dados de treino são os exemplos utilizados pelo algoritmo para aprender padrões.
modelo.fit(X_train, y_train)


# Etapa 5 - Avaliação
# O modelo faz previsões sobre dados de teste que ficaram separados do treinamento.
y_pred = modelo.predict(X_test)

# A acurácia representa a proporção de previsões corretas entre todas as previsões.
acuracia = accuracy_score(y_test, y_pred)

print("\nResultados da avaliação")
print(f"Quantidade de dados de treino: {len(X_train)}")
print(f"Quantidade de dados de teste: {len(X_test)}")
print(f"Acurácia: {acuracia:.2%}")


# Etapa 6 - Análise final
# O conjunto maior permite uma avaliação mais representativa que os 6 exemplos iniciais,
# mas os dados continuam sendo sintéticos e seguem uma regra artificial.
# Portanto, uma acurácia alta não garante bom desempenho com alunos reais.
# Em um projeto real seriam necessários dados reais, mais variáveis, validação cruzada
# e análise de métricas como precisão, recall e matriz de confusão.
