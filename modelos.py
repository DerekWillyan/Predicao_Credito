import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def mlp_model(data, target_column, one_hot_columns=None, label_columns=None, hidden_layer_sizes=(100,)):
    """
    Função para criar e treinar um modelo MLP.

    Parâmetros:
    - data: DataFrame do Pandas com os dados.
    - target_column: Nome da coluna alvo.
    - one_hot_columns: Lista de colunas categóricas para One-Hot Encoding (opcional).
    - label_columns: Lista de colunas categóricas para Label Encoding (opcional).
    - hidden_layer_sizes: Tupla com o número de neurônios em cada camada oculta.

    Retorna:
    - metrics: Dicionário com as métricas de avaliação.
    - model: Modelo MLP treinado.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if one_hot_columns:
        X = pd.get_dummies(X, columns=one_hot_columns, drop_first=True)

    if label_columns:
        label_encoder = LabelEncoder()
        for col in label_columns:
            X[col] = label_encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics, model


def knn_model(data, target_column, one_hot_columns=None, label_columns=None, n_neighbors=5):
    """
    Função para criar e treinar um modelo KNN.

    Parâmetros:
    - data: DataFrame do Pandas com os dados.
    - target_column: Nome da coluna alvo.
    - one_hot_columns: Lista de colunas categóricas para One-Hot Encoding (opcional).
    - label_columns: Lista de colunas categóricas para Label Encoding (opcional).
    - n_neighbors: Número de vizinhos (k) para o KNN.

    Retorna:
    - metrics: Dicionário com as métricas de avaliação.
    - model: Modelo KNN treinado.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if one_hot_columns:
        X = pd.get_dummies(X, columns=one_hot_columns, drop_first=True)

    if label_columns:
        label_encoder = LabelEncoder()
        for col in label_columns:
            X[col] = label_encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics, model


def naive_bayes_model(data, target_column, one_hot_columns=None, label_columns=None):
    """
    Função para criar e treinar um modelo Naive Bayes.

    Parâmetros:
    - data: DataFrame do Pandas com os dados.
    - target_column: Nome da coluna alvo.
    - one_hot_columns: Lista de colunas categóricas para One-Hot Encoding (opcional).
    - label_columns: Lista de colunas categóricas para Label Encoding (opcional).

    Retorna:
    - metrics: Dicionário com as métricas de avaliação.
    - model: Modelo Naive Bayes treinado.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if one_hot_columns:
        X = pd.get_dummies(X, columns=one_hot_columns, drop_first=True)

    if label_columns:
        label_encoder = LabelEncoder()
        for col in label_columns:
            X[col] = label_encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics, model


def decision_tree_model(data, target_column, one_hot_columns=None, label_columns=None, max_depth=None):
    """
    Função para criar e treinar um modelo de Árvore de Decisão.

    Parâmetros:
    - data: DataFrame do Pandas com os dados.
    - target_column: Nome da coluna alvo.
    - one_hot_columns: Lista de colunas categóricas para One-Hot Encoding (opcional).
    - label_columns: Lista de colunas categóricas para Label Encoding (opcional).
    - max_depth: Profundidade máxima da árvore (opcional).

    Retorna:
    - metrics: Dicionário com as métricas de avaliação.
    - model: Modelo de Árvore de Decisão treinado.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if one_hot_columns:
        X = pd.get_dummies(X, columns=one_hot_columns, drop_first=True)

    if label_columns:
        label_encoder = LabelEncoder()
        for col in label_columns:
            X[col] = label_encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics, model


def mlp_model_pytorch(
    data, 
    target_column, 
    hidden_units_list, 
    task='binary', 
    learning_rate=0.001, 
    epochs=100, 
    batch_size=32,
    dropout_rate=0.5,
    test_size=0.3
):
    """
    Função para criar e treinar um modelo MLP usando PyTorch com diferentes números de neurônios por camada e Dropout.
    Retorna o modelo treinado e as métricas de desempenho no conjunto de teste.

    Parâmetros:
        data (pd.DataFrame): Dados de entrada no formato de DataFrame do Pandas.
        target_column (str): Nome da coluna alvo.
        hidden_units_list (list): Lista de inteiros, onde cada inteiro representa o número de neurônios em cada camada oculta.
        task (str): Tipo de tarefa ('binary', 'multiclass' ou 'regression').
        learning_rate (float): Taxa de aprendizado.
        epochs (int): Número de épocas de treinamento.
        batch_size (int): Tamanho do lote para treinamento.
        dropout_rate (float): Taxa de dropout (fração de neurônios desativados). Valor entre 0 e 1.
        test_size (float): Fração dos dados a ser usada como conjunto de teste (padrão: 0.3).

    Retorna:
        model: Modelo treinado.
        metrics: Dicionário contendo as métricas de desempenho no conjunto de teste.
    """
    # Separa os dados em features (X) e target (y)
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Converte os dados para tensores do PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32 if task == 'regression' else torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32 if task == 'regression' else torch.long)

    # Define a dimensão de entrada com base nos dados
    input_dim = X_train.shape[1]

    # Define a dimensão de saída com base na tarefa
    if task == 'binary':
        output_dim = 1
        criterion = nn.BCEWithLogitsLoss()  # Loss para classificação binária
    elif task == 'multiclass':
        output_dim = len(torch.unique(y_train))  # Número de classes únicas
        criterion = nn.CrossEntropyLoss()  # Loss para classificação multiclasse
    elif task == 'regression':
        output_dim = 1
        criterion = nn.MSELoss()  # Loss para regressão
    else:
        raise ValueError("O parâmetro 'task' deve ser 'binary', 'multiclass' ou 'regression'.")

    # Cria o modelo MLP
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            layers = []
            # Primeira camada oculta
            layers.append(nn.Linear(input_dim, hidden_units_list[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Adiciona Dropout após a ativação
            # Camadas ocultas intermediárias
            for i in range(1, len(hidden_units_list)):
                layers.append(nn.Linear(hidden_units_list[i-1], hidden_units_list[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))  # Adiciona Dropout após a ativação
            # Camada de saída
            layers.append(nn.Linear(hidden_units_list[-1], output_dim))
            if task == 'binary':
                layers.append(nn.Sigmoid())  # Ativação final para classificação binária
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = MLP()

    # Define o otimizador
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepara os dados para treinamento
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Treina o modelo
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            if task == 'binary':
                loss = criterion(outputs.squeeze(), batch_y.float())
            elif task == 'multiclass':
                loss = criterion(outputs, batch_y)
            elif task == 'regression':
                loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader)}")

    # Avalia o modelo no conjunto de teste
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        if task == 'binary':
            preds = (outputs.squeeze() > 0.5).float().numpy()  # Limiar de 0.5 para classificação binária
            y_true = y_test.numpy()
            metrics = {
                'accuracy': accuracy_score(y_true, preds),
                'precision': precision_score(y_true, preds),
                'recall': recall_score(y_true, preds),
                'f1_score': f1_score(y_true, preds)
            }
        elif task == 'multiclass':
            preds = torch.argmax(outputs, dim=1).numpy()  # Classe com maior probabilidade
            y_true = y_test.numpy()
            metrics = {
                'accuracy': accuracy_score(y_true, preds),
                'precision': precision_score(y_true, preds, average='weighted'),
                'recall': recall_score(y_true, preds, average='weighted'),
                'f1_score': f1_score(y_true, preds, average='weighted')
            }
        elif task == 'regression':
            preds = outputs.squeeze().numpy()
            y_true = y_test.numpy()
            metrics = {
                'mae': mean_absolute_error(y_true, preds),
                'mse': mean_squared_error(y_true, preds),
                'rmse': mean_squared_error(y_true, preds, squared=False)
            }

    return model, metrics

# ------------------------------------------
# Função para Regressão Múltiplas
# ------------------------------------------

def regressao_multipla(df, dependente, independentes):
    """
    Realiza uma regressão múltipla usando statsmodels.

    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    dependente (str): O nome da variável dependente.
    independentes (list): Lista com os nomes das variáveis independentes.

    Retorna:
    O resumo do modelo de regressão.
    """
    # Adiciona uma constante ao modelo (intercepto)
    X = sm.add_constant(df[independentes])  # Variáveis independentes
    y = df[dependente]  # Variável dependente

    # Ajusta o modelo de regressão
    modelo = sm.OLS(y, X).fit()

    # Retorna o resumo do modelo
    return modelo

# ------------------------------------------
# Função para Regressão Linear
# ------------------------------------------

def regressao_linear(df=None, x_col=None, y_col=None, n_clusters=None, show_trendline=False, 
                 title="Scatter Plot", xlabel="X Axis", ylabel="Y Axis",
                 trendline_color='red', show_correlation=True):
    
    """
    Cria um gráfico scatter com opções de clustering e linha de tendência.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    x_col (str): Nome da coluna no DataFrame para o eixo X.
    y_col (str): Nome da coluna no DataFrame para o eixo Y.
    n_clusters (int): Número de clusters para K-Means.
    show_trendline (bool): Se True, mostra a linha de regressão linear.
    title (str): Título do gráfico.
    xlabel (str): Rótulo do eixo X.
    ylabel (str): Rótulo do eixo Y.
    trendline_color (str): Cor da linha de tendência.
    show_correlation (bool): Se True, exibe o coeficiente de correlação no gráfico.
    
    Retorna:
    model: Modelo de regressão linear treinado (se show_trendline=True).
    """
    # Verifica se o DataFrame foi fornecido
    if df is None or x_col is None or y_col is None:
        raise ValueError("Você deve fornecer um DataFrame e os nomes das colunas para x e y.")
    
    # Extrai os dados das colunas especificadas
    x = df[x_col].values
    y = df[y_col].values
    
    # Aplica K-Means se especificado
    if n_clusters is not None:
        data = np.column_stack((x, y))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_
        colors = plt.cm.viridis(labels / (n_clusters - 1)) if n_clusters > 1 else 'blue'
    else:
        colors = 'blue'
        centroids = None
    
    # Calcula o coeficiente de correlação de Pearson
    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]
    
    # Configura o gráfico
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=colors, alpha=0.6, edgecolors='w', label='Dados')
    
    # Inicializa o modelo de regressão linear
    model = None
    if show_trendline:
        # Treina o modelo de regressão linear
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)  # Ajusta o modelo aos dados
        
        # Calcula a linha de tendência
        trendline_x = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        trendline_y = model.predict(trendline_x)
        
        # Calcula o R²
        r_squared = r2_score(y, model.predict(x.reshape(-1, 1)))
        
        # Plota a linha de tendência
        plt.plot(trendline_x, trendline_y, color=trendline_color, linestyle='--', 
                 label=f'Linha de Tendência\n$R^2 = {r_squared:.2f}$')
    
    # Centróides
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', 
                    s=100, linewidths=1.5, label='Centroids')
    
    # Exibe o coeficiente de correlação DENTRO do gráfico
    if show_correlation:
        # Posiciona o texto no canto superior direito do gráfico
        plt.text(
            0.95, 0.95,  # Posição relativa (95% da largura e altura do gráfico)
            f'Correlação: {correlation_coefficient:.2f}', 
            transform=plt.gca().transAxes,  # Usa coordenadas relativas ao eixo
            fontsize=12, 
            ha='right',  # Alinhamento horizontal à direita
            va='top',    # Alinhamento vertical ao topo
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')  # Caixa de texto
        )
    
    # Configurações finais
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    if show_trendline or centroids is not None:
        plt.legend()
    plt.show()
    
    # Retorna o modelo treinado (se aplicável)
    return model