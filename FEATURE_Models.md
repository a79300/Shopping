# FEATURE: Modelos

- **Descrição:** Documentação dos modelos treinados e avaliados pelo script `shopping.py`.

- **Modelos incluídos no script:**
  - `KNN` — K-Nearest Neighbors (`KNeighborsClassifier(n_neighbors=3)`).
  - `Random Forest` — `RandomForestClassifier(n_estimators=100, random_state=42)`.
  - `Random Forest (Tuned)` — estimador obtido por `GridSearchCV` sobre `n_estimators` e `max_depth`.
  - `Logistic Regression` — `LogisticRegression(max_iter=5000, solver='saga')`.
  - `SVM` — `SVC(kernel='linear')`.
  - `Decision Tree` — `DecisionTreeClassifier(random_state=42)`.
  - `Gradient Boosting` — `GradientBoostingClassifier(random_state=42)`.
  - `Extra Trees` — `ExtraTreesClassifier(n_estimators=100, random_state=42)`.
  - `Naive Bayes` — `GaussianNB()`.
  - `MLP Neural Net` — `MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, early_stopping=True, random_state=42)`.

- **Pré-processamento aplicado no fluxo:**
  - Divisão treino/teste: `train_test_split(..., test_size=0.4)` (60% treino, 40% teste).
  - Escalonamento: `StandardScaler()` aplicado aos dados antes de treinar os modelos.
  - Ajuste de hiperparâmetros (apenas Random Forest): `GridSearchCV(RandomForestClassifier(...), param_grid, cv=3)`.

- **Métricas calculadas pelo script:**
  - **Acurácia (Accuracy):** proporção de predições correctas.
  - **Acertos / Erros (`Corrects` / `Incorrects`):** contagens de acertos/erros convertidas em percentagem no relatório final.
  - **Sensibilidade (Sensitivity / TPR):** proporção de compradores reais correctamente identificados.
  - **Especificidade (Specificity / TNR):** proporção de não-compradores correctamente identificados.

- **Interpretação e recomendações:**
  - Em conjuntos de dados desbalanceados (muitos não-compradores), a acurácia tende a favorecer a classe maioritária — prefira métricas como sensibilidade, especificidade e outras baseadas na matriz de confusão (precisão, recall, F1) para análise.
  - Compare modelos utilizando validação cruzada estratificada (`StratifiedKFold`) para obter estimativas mais robustas.
  - Para `Random Forest` e `Extra Trees`, utilize `feature_importances_` para identificar as variáveis com maior contributo.
  - Para `MLP` e `SVM`, o escalonamento dos dados é crítico (o script já aplica `StandardScaler`).
  - Para `Logistic Regression`, verifique a regularização (`C`) e o solver; para conjuntos grandes, `saga` é uma boa opção.

- **Sugestões de melhoria e validação:**
  - Utilizar `GridSearchCV` ou `RandomizedSearchCV` para cada modelo (por exemplo: número de vizinhos no KNN, `max_depth`/`n_estimators` em árvores, `C`/`gamma` em SVM, arquitectura e learning_rate no MLP).
  - Implementar `StratifiedKFold` para validação cruzada e reportar métricas médias com desvio-padrão.
  - Aplicar técnicas de balanceamento (SMOTE, undersampling) se o desbalanceamento prejudicar a sensibilidade.
  - Guardar modelos treinados com `joblib.dump` para utilização futura em inferência.
  - Calibrar probabilidades (por exemplo com `CalibratedClassifierCV`) se as probabilidades previstas forem utilizadas em decisões de negócio.

- **Dicas práticas para reprodutibilidade:**
  - Fixar `random_state` onde aplicável.
  - Registar a versão das bibliotecas (`pip freeze > requirements.txt`).
  - Documentar o `test_size` e a forma como as percentagens (`Corrects`/`Incorrects`) são calculadas.

- **Comando de exemplo para executar o script:**

```powershell
python .\shopping.py .\shopping.csv
```

(O script treina os modelos, avalia, grava `results_models.csv` e gera gráficos.)
