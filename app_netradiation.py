import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import joblib
from datetime import datetime

# ============================================================
# CLASSE PREDITOR (adaptada para carregar modelo por horizonte)
# ============================================================
class RadiationPredictor:
    def __init__(self, base_dir=".", horizon=1):
        self.horizon = horizon
        self.model_dir = os.path.join(base_dir, f"modelos_treinados_{horizon}h")
        self.metadata = {}
        
        # Carregar modelo
        model_path = os.path.join(self.model_dir, "ML_GradientBoosting.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        self.model = joblib.load(model_path)
        self.feature_importances = self.model.feature_importances_
        
        # Carregar scaler
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado em {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Carregar feature names
        features_path = os.path.join(self.model_dir, "feature_names.json")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature names não encontrado em {features_path}")
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        
        # Carregar metadata (opcional)
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'R2': 0.89, 'RMSE': 61, 'MAE': 30, 'model': 'GradientBoosting'}
    
    def _compute_features(self, df):
        """Calcula features a partir do DataFrame (última linha)."""
        df = df.sort_values('datetime').reset_index(drop=True)
        last_idx = df.index[-1]
        
        # Calcular albedo
        df['albedo'] = np.where(df['SWTop'] > 10, df['SWBot'] / df['SWTop'], np.nan)
        df['albedo'] = df['albedo'].clip(0, 1)
        
        if df['T_C'].mean() > 200:
            df['T_C'] = df['T_C'] - 273.15
        
        dt = df.loc[last_idx, 'datetime']
        hour = dt.hour
        dayofyear = dt.dayofyear
        
        features_dict = {}
        # Temporais
        features_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features_dict['day_of_year_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
        features_dict['day_of_year_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
        features_dict['month'] = dt.month
        features_dict['day_of_week'] = dt.dayofweek
        features_dict['season'] = (dt.month % 12 + 3) // 3
        features_dict['is_daylight'] = 1 if 6 <= hour <= 18 else 0
        
        if 5 <= hour < 12:
            features_dict['time_period'] = 1
        elif 12 <= hour < 17:
            features_dict['time_period'] = 2
        elif 17 <= hour < 21:
            features_dict['time_period'] = 3
        else:
            features_dict['time_period'] = 4
        
        # Variáveis exógenas
        exogenous = ['SWTop', 'SWBot', 'LWTop', 'LWBot', 'T_C', 'albedo']
        max_lag = 12
        windows = [3, 6, 12]
        
        self._raw_values = {}
        for var in exogenous:
            if var not in df.columns:
                continue
            series = df[var].values
            for lag in range(1, max_lag + 1):
                idx = -(lag + 1)
                if len(series) > lag:
                    val = series[idx]
                else:
                    val = np.nan
                features_dict[f'{var}_lag{lag}'] = val
                self._raw_values[f'{var}_lag{lag}'] = val
            for window in windows:
                if len(series) >= window:
                    window_data = series[-(window+1):-1]
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                else:
                    mean_val = np.nan
                    std_val = np.nan
                features_dict[f'{var}_MA{window}'] = mean_val
                features_dict[f'{var}_Std{window}'] = std_val
                self._raw_values[f'{var}_MA{window}'] = mean_val
                self._raw_values[f'{var}_Std{window}'] = std_val
            if len(series) >= 3:
                diff_val = series[-2] - series[-3]
            else:
                diff_val = np.nan
            features_dict[f'{var}_diff'] = diff_val
            self._raw_values[f'{var}_diff'] = diff_val
        
        # Montar vetor
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(features_dict.get(name, np.nan))
        
        self._feature_vector_raw = np.array(feature_vector).reshape(1, -1)
        return self._feature_vector_raw
    
    def predict(self, df_input):
        """Faz a previsão para a última linha do DataFrame."""
        if len(df_input) < 13:
            raise ValueError(f"Precisa de pelo menos 13 linhas. Foram {len(df_input)}.")
        
        features_raw = self._compute_features(df_input)
        has_nan = np.any(np.isnan(features_raw))
        features_scaled = self.scaler.transform(features_raw)
        pred = self.model.predict(features_scaled)[0]
        
        # Contexto físico
        last_hour = df_input.iloc[-1]['datetime'].hour
        if 6 <= last_hour <= 18:
            time_context = "Dia (com radiação solar ativa)"
        else:
            time_context = "Noite (sem radiação solar direta)"
        
        if pred > 300:
            magnitude = "🔆 Alta (forte aquecimento diurno)"
        elif pred > 100:
            magnitude = "☀️ Média (aquecimento moderado)"
        elif pred > -20:
            magnitude = "🌤️ Baixa ou próxima de zero (transição ou nebulosidade)"
        else:
            magnitude = "❄️ Negativa (perda de calor noturna)"
        
        return {
            'prediction': pred,
            'time_context': time_context,
            'magnitude': magnitude,
            'has_missing_data': has_nan,
            'input_summary': df_input.tail(3),
            'top_features': self._get_top_features(),
            'model_metrics': self.metadata
        }
    
    def _get_top_features(self, top_n=10):
        """Retorna as top N features com seus valores atuais."""
        if not hasattr(self, '_feature_vector_raw'):
            return []
        importances = []
        for i, name in enumerate(self.feature_names):
            imp = self.feature_importances[i] if i < len(self.feature_importances) else 0.0
            raw_val = self._raw_values.get(name, np.nan)
            importances.append((name, imp, raw_val))
        importances.sort(key=lambda x: x[1], reverse=True)
        return importances[:top_n]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def normalize_columns(df):
    """Normaliza nomes das colunas para o padrão esperado."""
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('-', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    return df

def map_columns(df):
    """Mapeia colunas para o formato final."""
    col_map = {}
    required_lower = ['datetime', 'swtop', 'swbot', 'lwtop', 'lwbot', 't_c']
    for req in required_lower:
        matches = [c for c in df.columns if req in c]
        if matches:
            col_map[req] = matches[0]
    # Renomear
    rename = {
        col_map['datetime']: 'datetime',
        col_map['swtop']: 'SWTop',
        col_map['swbot']: 'SWBot',
        col_map['lwtop']: 'LWTop',
        col_map['lwbot']: 'LWBot',
        col_map['t_c']: 'T_C'
    }
    df = df.rename(columns=rename)
    # Garantir que datetime seja datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def load_data(uploaded_file):
    """Carrega e processa o arquivo CSV enviado."""
    try:
        # Tentar vários separadores
        for sep in [';', ',', '\t', '|']:
            try:
                df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                break
            except:
                continue
        else:
            raise ValueError("Não foi possível ler o CSV")
        
        df = normalize_columns(df)
        df = map_columns(df)
        # Verificar colunas obrigatórias
        required = ['datetime', 'SWTop', 'SWBot', 'LWTop', 'LWBot', 'T_C']
        if not all(col in df.columns for col in required):
            missing = set(required) - set(df.columns)
            raise ValueError(f"Colunas faltando: {missing}")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.set_page_config(page_title="Preditor de Radiação Líquida", layout="wide")
st.title("🌞 Preditor de Radiação Líquida (Rn)")
st.markdown("Faça upload de um arquivo CSV com dados horários e obtenha a previsão para o horizonte desejado.")

# Sidebar - configurações
st.sidebar.header("⚙️ Configurações")
horizon = st.sidebar.selectbox("Horizonte de previsão (horas)", [1, 3, 6, 12], index=0)
uploaded_file = st.sidebar.file_uploader("📂 Escolha um arquivo CSV", type=['csv'])

# Botão de previsão
if st.sidebar.button("🔮 Prever") and uploaded_file is not None:
    # Carregar dados
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    
    if len(df) < 13:
        st.error(f"O arquivo tem apenas {len(df)} linhas. São necessárias pelo menos 13 (12 horas históricas + atual).")
        st.stop()
    
    # Carregar modelo
    try:
        predictor = RadiationPredictor(base_dir=".", horizon=horizon)
    except FileNotFoundError as e:
        st.error(f"Modelo não encontrado para horizonte {horizon}h. Certifique-se de que a pasta 'modelos_treinados_{horizon}h' existe.")
        st.stop()
    
    # Fazer previsão
    with st.spinner(f"Carregando modelo e processando..."):
        result = predictor.predict(df)
    
    # ============================================================
    # EXIBIÇÃO DOS RESULTADOS
    # ============================================================
    st.success("✅ Previsão realizada com sucesso!")
    
    # Layout em colunas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(label="🌞 Radiação Líquida Prevista", 
                  value=f"{result['prediction']:.2f} W/m²",
                  delta=f"Horizonte: {horizon}h")
        st.write(f"**Contexto:** {result['time_context']}")
        st.write(f"**Magnitude:** {result['magnitude']}")
        if result['has_missing_data']:
            st.warning("⚠️ Dados históricos com NaN detectados - previsão pode ser menos confiável.")
    
    with col2:
        st.subheader("📊 Desempenho do Modelo")
        met = result['model_metrics']
        st.write(f"**R²:** {met.get('R2', 0.89):.4f}")
        st.write(f"**RMSE:** {met.get('RMSE', 61):.2f} W/m²")
        st.write(f"**MAE:** {met.get('MAE', 30):.2f} W/m²")
    
    # Gráfico: série temporal (últimas 12 horas + previsão)
    st.subheader("📈 Série Temporal - Últimas 12 horas e Previsão")
    fig, ax = plt.subplots(figsize=(12, 5))
    # Pegar as últimas 12 linhas
    last_12 = df.tail(12).copy()
    # Adicionar a previsão como um ponto futuro
    last_dt = last_12['datetime'].iloc[-1]
    future_dt = last_dt + pd.Timedelta(hours=horizon)
    
    ax.plot(last_12['datetime'], last_12['Rn'], 'b-o', label='Rn Real')
    ax.scatter([future_dt], [result['prediction']], color='red', s=100, zorder=5, label=f'Previsão ({horizon}h)')
    ax.axvline(x=last_dt, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Horário')
    ax.set_ylabel('Rn (W/m²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Tabela com features importantes
    st.subheader("🏆 Principais Features que influenciaram a previsão")
    top_feat = result['top_features']
    if top_feat:
        df_feat = pd.DataFrame(top_feat, columns=['Feature', 'Importância', 'Valor Atual'])
        df_feat['Importância'] = df_feat['Importância'] * 100
        df_feat = df_feat.round({'Importância': 2, 'Valor Atual': 2})
        st.dataframe(df_feat, use_container_width=True)
    else:
        st.info("Não foi possível extrair a importância das features.")
    
    # Mostrar os dados de entrada (últimas 3 linhas)
    with st.expander("📋 Ver dados de entrada (últimas 3 horas)"):
        st.dataframe(result['input_summary'].round(2))
    
else:
    if uploaded_file is None:
        st.info("👈 Faça upload de um arquivo CSV na barra lateral e clique em 'Prever'.")
    else:
        st.warning("Clique em 'Prever' para gerar a previsão.")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido com Streamlit | Modelo GradientBoosting")