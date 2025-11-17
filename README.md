# Shopping Predictor - Previsão de Intenção de Compra

## 📋 Introdução

Este projeto implementa um modelo de Machine Learning para prever se um visitante de um site de e-commerce irá realizar uma compra com base no seu comportamento de navegação. Utilizando algoritmos de classificação, o sistema analisa diversos parâmetros da sessão do usuário para determinar a probabilidade de conversão em vendas.

O projeto foi desenvolvido como parte de um estudo em Inteligência Artificial, utilizando a biblioteca scikit-learn para implementar um classificador K-Nearest Neighbors (KNN) que aprende padrões de comportamento de compra a partir de dados históricos.

## 🎯 Descrição do Projeto

### Funcionalidades Principais

- **Carregamento de Dados**: Importa e processa dados de comportamento de usuários a partir de um arquivo CSV
- **Pré-processamento**: Converte dados categóricos (como meses, tipo de visitante) em valores numéricos
- **Treinamento de Modelo**: Utiliza o algoritmo K-Nearest Neighbors para aprender padrões de compra
- **Avaliação de Performance**: Calcula métricas de sensibilidade e especificidade para avaliar a precisão do modelo
- **Predição**: Classifica novos visitantes em "compradores" ou "não compradores"

### Características Analisadas

O modelo considera 17 características diferentes de cada sessão de usuário:

1. **Administrative** - Número de páginas administrativas visitadas
2. **Administrative_Duration** - Tempo gasto em páginas administrativas
3. **Informational** - Número de páginas informacionais visitadas
4. **Informational_Duration** - Tempo gasto em páginas informacionais
5. **ProductRelated** - Número de páginas de produtos visitadas
6. **ProductRelated_Duration** - Tempo gasto em páginas de produtos
7. **BounceRates** - Taxa de rejeição
8. **ExitRates** - Taxa de saída
9. **PageValues** - Valor médio das páginas visitadas
10. **SpecialDay** - Proximidade de datas especiais (0-1)
11. **Month** - Mês da visita
12. **OperatingSystems** - Sistema operacional utilizado
13. **Browser** - Navegador utilizado
14. **Region** - Região geográfica
15. **TrafficType** - Tipo de tráfego
16. **VisitorType** - Tipo de visitante (novo ou retornante)
17. **Weekend** - Se a visita ocorreu no fim de semana

### Métricas de Avaliação

- **Sensibilidade (True Positive Rate)**: Percentual de compradores corretamente identificados
- **Especificidade (True Negative Rate)**: Percentual de não-compradores corretamente identificados
- **Acurácia**: Número total de predições corretas vs. incorretas

## 🤖 Ferramentas de IA Usadas

### Durante o Desenvolvimento

1. **GitHub Copilot**
   - Assistência na escrita de código Python
   - Sugestões para implementação de funções
   - Otimização de estruturas de dados
   - Documentação de código

2. **ChatGPT / Claude**
   - Consultas sobre melhores práticas de Machine Learning
   - Explicações sobre algoritmos de classificação
   - Ajuda na compreensão de métricas de avaliação
   - Suporte na estruturação do projeto

### Bibliotecas de IA/ML Utilizadas

- **scikit-learn**: Biblioteca principal para Machine Learning
  - `KNeighborsClassifier`: Algoritmo de classificação K-NN
  - `train_test_split`: Divisão de dados para treino e teste
- **NumPy/Pandas**: Manipulação de dados (implicitamente via sklearn)

## 💬 Exemplos de Prompts e Outputs

### Exemplo 1: Criação da Função de Carregamento de Dados

**Prompt usado:**
```
"Create a function to load data from a CSV file with shopping behavior data. 
The function should convert month names to numbers, visitor types to binary, 
and weekend/revenue boolean strings to integers."
```

**Output gerado:**
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

**Prompt usado:**
```
"Implement a function to train a K-Nearest Neighbors classifier 
with n_neighbors=1 using sklearn"
```

**Output gerado:**
```python
def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model
```

### Exemplo 3: Função de Avaliação

**Prompt usado:**
```
"Create an evaluation function that calculates sensitivity (true positive rate) 
and specificity (true negative rate) from actual labels and predictions"
```

**Output gerado:**
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

**Output típico:**
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

- O programa aceita um argumento: o caminho para o arquivo CSV com os dados de compra
- O arquivo CSV deve conter as colunas especificadas na seção "Características Analisadas"

## 📊 Estrutura do Dataset

O arquivo `shopping.csv` contém **12.330 sessões** de usuários, com as seguintes características:

- **Formato**: CSV com cabeçalho
- **Colunas**: 18 (17 features + 1 label)
- **Label**: Revenue (TRUE/FALSE) - indica se houve compra
- **Distribuição**: Aproximadamente 85% não-compradores, 15% compradores

## 🔧 Detalhes Técnicos

### Algoritmo Principal: K-Nearest Neighbors (KNN)

- **Parâmetro**: n_neighbors = 1
- **Funcionamento**: Classifica baseado no vizinho mais próximo no espaço de features
- **Vantagens**: Simples, efetivo para dados bem distribuídos
- **Desvantagens**: Pode ser sensível a outliers e ruído

### Divisão de Dados

- **Treino**: 60% dos dados
- **Teste**: 40% dos dados
- **Método**: train_test_split com divisão aleatória

### Alternativas Comentadas no Código

```python
# model = RandomForestClassifier(n_estimators=100, random_state=42)
```

O código inclui a possibilidade de usar Random Forest como alternativa ao KNN.

## 📈 Melhorias Futuras

1. **Otimização de Hiperparâmetros**: Testar diferentes valores de k no KNN
2. **Feature Engineering**: Criar novas features derivadas das existentes
3. **Ensemble Methods**: Combinar múltiplos modelos para melhor performance
4. **Cross-Validation**: Implementar validação cruzada para avaliação mais robusta
5. **Balanceamento de Classes**: Tratar o desbalanceamento entre compradores e não-compradores
6. **Interface Web**: Criar uma interface para predições em tempo real

## 📝 Observações sobre o Uso de IA

Este projeto demonstra como ferramentas de IA podem acelerar o desenvolvimento:

- **Produtividade**: Redução de ~40% no tempo de desenvolvimento
- **Qualidade**: Sugestões de código seguindo best practices
- **Aprendizado**: Explicações contextuais ajudaram a entender conceitos de ML
- **Debug**: Assistência na identificação e correção de erros

As ferramentas de IA foram usadas como assistentes, com revisão humana de todo o código gerado para garantir qualidade e entendimento completo.

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais.

## 👤 Autor

Desenvolvido como parte de um projeto de Inteligência Artificial.

