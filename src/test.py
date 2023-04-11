import pandas as pd

# leia o arquivo CSV
df = pd.read_csv('assets/items.csv', sep=';', encoding='iso-8859-1')
df['Peso'] = df['Peso'].str.replace(',', '.').astype(float)
df['Preço'] = df['Preço'].str.replace(',', '.').astype(float)
# ordene os itens pela relação utilidade-peso em ordem decrescente
df['utilidade_peso'] = df['Utilidade'] / df['Peso']
df = df.sort_values('utilidade_peso', ascending=False)

# inicialize as variáveis
peso_atual = 0
utilidade_atual = 0
itens_na_mochila = []

# percorra todos os itens
for index, row in df.iterrows():
    # se o item não exceder a capacidade da mochila, adicione-o
    if peso_atual + row['Peso'] <= 30:
        itens_na_mochila.append(row['item'])
        peso_atual += row['Peso']
        utilidade_atual += row['Utilidade']

# imprima os resultados
print("Itens na mochila: ", itens_na_mochila)
print("Peso total: ", peso_atual, "kg")
print("Utilidade total: ", utilidade_atual)