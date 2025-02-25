# Análise e Predição de Crédito

Este repositório contém um notebook Jupyter (`predcit.ipynb`) que realiza a análise e predição de crédito utilizando um modelo XGBoost. O objetivo é prever o status de crédito (aprovado ou não) com base em diversas características dos clientes.

## Descrição do Projeto

O notebook realiza as seguintes etapas:

1. **Conexão com o Banco de Dados**: Conecta-se a um banco de dados PostgreSQL para carregar os dados de crédito.
2. **Exploração dos Dados**: Realiza uma análise inicial dos dados, incluindo a verificação de valores nulos e a descrição das colunas.
3. **Pré-processamento**: Preenche valores nulos com a mediana das colunas numéricas.
4. **Treinamento do Modelo**: Utiliza o algoritmo XGBoost para treinar um modelo de classificação binária.
5. **Avaliação do Modelo**: Calcula métricas de desempenho como precisão, recall, F1-score e AUC-ROC.

## Requisitos

Para executar o notebook, você precisará das seguintes bibliotecas Python:

- `pandas`
- `psycopg2`
- `xgboost`
- `scikit-learn`
- `numpy`

Você pode instalar as dependências usando o seguinte comando:

```bash
pip install pandas psycopg2 xgboost scikit-learn numpy
```

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/DerekWillyan/Predicao_Credito.git
   ```

2. Navegue até o diretório do projeto:
   ```bash
   cd Predicao_Credito
   ```

3. Execute o notebook Jupyter:
   ```bash
   jupyter notebook predcit.ipynb
   ```

4. Siga as instruções no notebook para carregar os dados, treinar o modelo e avaliar os resultados.

## Resultados

O modelo XGBoost alcançou as seguintes métricas de desempenho:

- **Acurácia**: 75%
- **AUC-ROC**: 0.73582
- **Precisão (Classe 1)**: 77.55%
- **Recall (Classe 1)**: 90.48%
- **F1-Score (Classe 1)**: 83.52%

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorar o código ou adicionar novas funcionalidades.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
