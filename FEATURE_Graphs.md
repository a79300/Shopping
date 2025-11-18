# FEATURE: Gráficos

- **Descrição:** Documenta os gráficos gerados pelo script `shopping.py` para comparar modelos e visualizar métricas.

- **Gráficos disponíveis:**
  - **Gráfico de barras agrupadas (Grouped Bar Chart):** comparação lado a lado das métricas por modelo.
    - Eixo X: nomes dos modelos (`df['Model']`).
    - Séries: `Accuracy`, `Corrects`, `Incorrects`, `Sensitivity`, `Specificity` (valores em percentagem).
    - Biblioteca: `plotly.graph_objects.Bar` com `barmode='group'`.
    - Utilização: comparação rápida de desempenho entre modelos para várias métricas simultaneamente.

  - **Mapa de calor (Heatmap):** visualização em matriz das mesmas métricas para identificar padrões.
    - Dados: `df.set_index('Model')[['Accuracy','Corrects','Incorrects','Sensitivity','Specificity']].T`
    - Biblioteca: `plotly.express.imshow`.
    - Utilização: destacar modelos consistentes (colunas/linhas com cores mais favoráveis).

- **Como os valores são calculados no script:**
  - `Accuracy`, `Sensitivity` e `Specificity` são convertidos para percentagem multiplicando por 100.
  - As colunas `Corrects` e `Incorrects` são também convertidas em percentagem (o script normaliza por um valor fixo quando gera estas percentagens — ver `shopping.py` para adaptar ao número real de amostras do conjunto de teste).

- **Interpretação rápida:**
  - **Acurácia (Accuracy):** visão geral; sensível à distribuição das classes (pode induzir em erro em conjuntos de dados desbalanceados).
  - **Sensibilidade (Sensitivity / TPR):** proporção de compradores reais correctamente identificados — importante se o objectivo for capturar potenciais compradores.
  - **Especificidade (Specificity / TNR):** proporção de não-compradores correctamente identificados — importante para reduzir acções desnecessárias.
  - **Acertos/Erros (`Corrects` / `Incorrects`):** mostram a proporção de predições correctas/incorrectas em termos relativos.

- **Boas práticas e personalizações:**
  - Usar `template='plotly_dark'` ou outro template consistente com o relatório.
  - Ajustar `yaxis.range` para `[0,100]` quando apresentar percentagens.
  - Mostrar valores nas barras com `texttemplate` e formatar para 2 casas decimais.
  - Para exportar imagens estáticas: `fig.write_image('chart.png', scale=2)` (requer `kaleido` ou `orca`).
  - Para apresentações: ordenar `df` por `Accuracy` (ou por outra métrica de interesse) antes de plotar.

- **Sugestões de melhoria:**
  - Normalizar `Corrects`/`Incorrects` usando o tamanho real do subconjunto de teste em vez de um valor fixo no código.
  - Adicionar intervalos de confiança (por exemplo com bootstrap) para as métricas e plotar barras de erro.
  - Permitir seleccionar o subconjunto de métricas a plotar através de argumentos de linha de comando.
  - Tornar o mapa de calor interactiv(o) com tooltips detalhados (valores e contagens).

- **Exemplo de execução (linha de comando):**

```powershell
python .\shopping.py .\shopping.csv
```

(O script gera os gráficos automaticamente usando `plotly`.)
