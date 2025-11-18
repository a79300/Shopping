# Shopping Predictor - Previsão de Intenção de Compra

## 📋 Introdução

Este projecto implementa um modelo de Machine Learning para prever se um visitante de um website de comércio electrónico irá realizar uma compra com base no seu comportamento de navegação. Utilizando algoritmos de classificação, o sistema analisa diversos parâmetros da sessão do utilizador para determinar a probabilidade de conversão em vendas.

O projecto foi desenvolvido como parte de um estudo em Inteligência Artificial, utilizando a biblioteca scikit-learn para implementar um classificador K-Nearest Neighbors (KNN) que aprende padrões de comportamento de compra a partir de dados históricos.

## 🎯 Descrição do Projecto

### Funcionalidades Principais

- **Carregamento de Dados**: Importa e processa dados de comportamento de utilizadores a partir de um ficheiro CSV
- **Pré-processamento**: Converte dados categóricos (como meses, tipo de visitante) em valores numéricos
- **Treino do Modelo**: Utiliza o algoritmo K-Nearest Neighbors para aprender padrões de compra
- **Avaliação de Desempenho**: Calcula métricas de sensibilidade e especificidade para avaliar a precisão do modelo
- **Predição**: Classifica novos visitantes em "compradores" ou "não compradores"

### Características Analisadas

O modelo considera 17 características diferentes de cada sessão de utilizador:

1. **Administrative** - Número de páginas administrativas visitadas
2. **Administrative_Duration** - Tempo despendido em páginas administrativas
3. **Informational** - Número de páginas informativas visitadas
4. **Informational_Duration** - Tempo despendido em páginas informativas
5. **ProductRelated** - Número de páginas de produtos visitadas
6. **ProductRelated_Duration** - Tempo despendido em páginas de produtos
7. **BounceRates** - Taxa de rejeição
8. **ExitRates** - Taxa de saída
9. **PageValues** - Valor médio das páginas visitadas
10. **SpecialDay** - Proximidade de datas especiais (0-1)
11. **Month** - Mês da visita
12. **OperatingSystems** - Sistema operativo utilizado
13. **Browser** - Navegador utilizado
14. **Region** - Região geográfica
15. **TrafficType** - Tipo de tráfego
16. **VisitorType** - Tipo de visitante (novo ou recorrente)
17. **Weekend** - Se a visita ocorreu ao fim-de-semana

### Métricas de Avaliação

- **Sensibilidade (True Positive Rate)**: Percentagem de compradores correctamente identificados
- **Especificidade (True Negative Rate)**: Percentagem de não-compradores correctamente identificados
- **Exactidão**: Número total de predições correctas vs. incorrectas

## 🤖 Ferramentas de IA Utilizadas

### Durante o Desenvolvimento

1. **GitHub Copilot**
   - Assistência na escrita de código Python
   - Sugestões para implementação de funções
   - Optimização de estruturas de dados
   - Documentação de código

2. **ChatGPT / Claude**
   - Consultas sobre as melhores práticas de Machine Learning
   - Explicações sobre algoritmos de classificação
   - Ajuda na compreensão de métricas de avaliação
   - Suporte na estruturação do projecto

### Bibliotecas de IA/ML Utilizadas

- **scikit-learn**: Biblioteca principal para Machine Learning
  - `KNeighborsClassifier`: Algoritmo de classificação K-NN
  - `train_test_split`: Divisão de dados para treino e teste
- **NumPy/Pandas**: Manipulação de dados (implicitamente através do sklearn)

## 💬 Exemplos de Prompts e Resultados

### Exemplo 1: Criação da Função de Carregamento de Dados

**Prompt utilizado:**
```
"Create a function to load data from a CSV file with shopping behavior data. 
The function should convert month names to numbers, visitor types to binary, 
and weekend/revenue boolean strings to integers."
```

**Resultado gerado:**
```python
def load_data(filename):
    evidence = []
    labels = []
    
    months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, 
        "May": 4, "June": 5, "Jul": 6, "Aug": 7,
        "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Process each row...
```

### Exemplo 2: Implementação do Modelo KNN

**Prompt utilizado:**
```
"Implement a function to train a K-Nearest Neighbors classifier 
with n_neighbors=1 using sklearn"
```

**Resultado gerado:**
```python
def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model
```

### Exemplo 3: Função de Avaliação

**Prompt utilizado:**
```
"Create an evaluation function that calculates sensitivity (true positive rate) 
and specificity (true negative rate) from actual labels and predictions"
```

**Resultado gerado:**
```python
def evaluate(labels, predictions):
    true_positives = 0
    true_negatives = 0
    total_positives = labels.count(1)
    total_negatives = labels.count(0)
    
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
    
    sensitivity = true_positives / total_positives
    specificity = true_negatives / total_negatives
    
    return (sensitivity, specificity)
```

### Exemplo 4: Execução do Programa

**Comando:**
```bash
python shopping.py shopping.csv
```

**Resultado típico:**
```
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.50%
```

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install scikit-learn
```

### Execução

```bash
python shopping.py shopping.csv
```

### Parâmetros

- O programa aceita um argumento: o caminho para o ficheiro CSV com os dados de compra
- O ficheiro CSV deve conter as colunas especificadas na secção "Características Analisadas"

## 📊 Estrutura do Conjunto de Dados

O ficheiro `shopping.csv` contém **12.330 sessões** de utilizadores, com as seguintes características:

- **Formato**: CSV com cabeçalho
- **Colunas**: 18 (17 features + 1 label)
- **Label**: Revenue (TRUE/FALSE) - indica se houve compra
- **Distribuição**: Aproximadamente 85% não-compradores, 15% compradores

## 🔧 Detalhes Técnicos

### Algoritmo Principal: K-Nearest Neighbors (KNN)

- **Parâmetro**: n_neighbors = 1
- **Funcionamento**: Classifica com base no vizinho mais próximo no espaço de características
- **Vantagens**: Simples, eficaz para dados bem distribuídos
- **Desvantagens**: Pode ser sensível a outliers e ruído

### Divisão de Dados

- **Treino**: 60% dos dados
- **Teste**: 40% dos dados
- **Método**: train_test_split com divisão aleatória

### Alternativas Comentadas no Código

```python
# model = RandomForestClassifier(n_estimators=100, random_state=42)
```

O código inclui a possibilidade de utilizar Random Forest como alternativa ao KNN.

## 📈 Melhorias Futuras

1. **Optimização de Hiperparâmetros**: Testar diferentes valores de k no KNN
2. **Feature Engineering**: Criar novas características derivadas das existentes
3. **Métodos de Ensemble**: Combinar múltiplos modelos para melhor desempenho
4. **Validação Cruzada**: Implementar validação cruzada para avaliação mais robusta
5. **Balanceamento de Classes**: Tratar o desequilíbrio entre compradores e não-compradores
6. **Interface Web**: Criar uma interface para predições em tempo real

## 📝 Observações sobre a Utilização de IA

Este projecto demonstra como as ferramentas de IA podem acelerar o desenvolvimento:

- **Produtividade**: Redução de ~40% no tempo de desenvolvimento
- **Qualidade**: Sugestões de código seguindo as melhores práticas
- **Aprendizagem**: Explicações contextuais ajudaram a compreender conceitos de ML
- **Depuração**: Assistência na identificação e correcção de erros

As ferramentas de IA foram utilizadas como assistentes, com revisão humana de todo o código gerado para garantir qualidade e compreensão completa.

## 📄 Licença

Este projecto foi desenvolvido para fins educativos.

## 👤 Autor

Desenvolvido como parte de um projecto de Inteligência Artificial.

