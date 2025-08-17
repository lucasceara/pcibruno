# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import math

# ===================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CARREGAMENTO DOS MODELOS
# ===================================================================

st.set_page_config(
    page_title="Ferramenta PCI - Análise de Pavimentos",
    page_icon="ରା",
    layout="wide"
)

# @st.cache_resource garante que os modelos sejam carregados da memória apenas uma vez.
@st.cache_resource
def carregar_modelos():
    try:
        model = tf.keras.models.load_model("modelo_pci.keras")
        preprocessor = joblib.load("preprocessor_pci.joblib")
        scaler_y = joblib.load("scaler_y_pci.joblib")
        return model, preprocessor, scaler_y, True
    except Exception as e:
        # Exibe um erro persistente se os modelos não puderem ser carregados.
        st.error(f"Erro CRÍTICO ao carregar arquivos do modelo: {e}")
        st.stop() # Interrompe a execução do app se os modelos não existirem.

loaded_model, loaded_preprocessor, loaded_scaler_y, model_pronto = carregar_modelos()

# Inicialização do st.session_state para guardar os dados das tabelas e resultados
if 'amostras_data' not in st.session_state:
    st.session_state.amostras_data = {}
if 'pci_results' not in st.session_state:
    st.session_state.pci_results = {}

# ===================================================================
# 2. FUNÇÕES DE CÁLCULO (AMOSTRAGEM E PCI)
# ===================================================================

def calcular_amostras_params(CV, W, e, s, modo_area, area_manual):
    AREA_ALVO, AREA_MIN, AREA_MAX = 225.0, 135.0, 315.0
    if CV <= 0 or W <= 0: return {"error": "CV e W devem ser > 0."}
    if modo_area == 'Manual' and not (AREA_MIN <= area_manual <= AREA_MAX):
        return {"error": f"Área manual inválida. Permitido: {AREA_MIN:.0f}–{AREA_MAX:.0f} m²."}
    area = AREA_ALVO if modo_area == 'Automático' else area_manual
    ca = area / W
    if (CV/ca) <= 1: n_cont = 1.0
    else: n_cont = ( ( (CV/ca) * s**2 ) / ( ((e**2)/4) * ( (CV/ca) - 1 ) + s**2 ) )
    n_min = max(1, math.ceil(n_cont))
    espacamento = (CV / n_min) if n_min > 0 else 0
    posicoes = [math.floor((i * espacamento) * 10) / 10 for i in range(n_min)]
    return {"n_minimo": n_min, "Area_m2": area, "Posicoes_m": posicoes}

def classify_pci_and_get_color(pci_value):
    if pd.isna(pci_value): return "Não Calculado", "#808080"
    if 85 <= pci_value <= 100: return "Bom", "darkgreen"
    if 70 <= pci_value < 85: return "Satisfatório", "limegreen"
    if 55 <= pci_value < 70: return "Regular", "gold"
    if 40 <= pci_value < 55: return "Ruim", "deeppink"
    if 25 <= pci_value < 40: return "Muito ruim", "red"
    if 10 <= pci_value < 25: return "Péssimo", "saddlebrown"
    if 0 <= pci_value < 10: return "Perda funcional", "dimgray"
    return "Fora do Intervalo", "black"

def calcular_pci_para_amostra(df_amostra):
    dv_col_name = ('VALOR DEDUZIDO', '')
    if df_amostra.empty or dv_col_name not in df_amostra.columns: return np.nan
    df_amostra[dv_col_name] = pd.to_numeric(df_amostra[dv_col_name], errors='coerce')
    df_ordenada = df_amostra.sort_values(by=dv_col_name, ascending=False, na_position='last').reset_index(drop=True)
    dv_validos = df_ordenada[dv_col_name].dropna()
    if dv_validos.empty: return 100
    hdv = dv_validos.iloc[0]
    m_calc = 1 + (9/98) * (100 - hdv)
    m = round(min(10, m_calc))
    df_filtrada = df_ordenada.head(m)
    q = sum(1 for v in df_filtrada[dv_col_name] if v > 2)
    if q <= 1:
        cdv = df_filtrada[dv_col_name].sum()
    else:
        vdt_total = df_filtrada[dv_col_name].sum()
        # Simplificação da fórmula de VDC para uso no app
        cdv_calc = -0.0018 * (vdt_total**2) + 0.9187 * vdt_total - 18.047
        cdv = max(cdv_calc, hdv)
    return max(0, 100 - cdv)

def prever_valor_deduzido(defeito_fmt, severidade_code, densidade):
    if not model_pronto: return np.nan
    try:
        defeito = defeito_fmt.split(' - ', 1)[1]
        dados = pd.DataFrame([[defeito, severidade_code, densidade]], columns=['DEFEITO', 'SEVERIDADE', 'DENSIDADE (%)'])
        pred_scaled = loaded_model.predict(loaded_preprocessor.transform(dados), verbose=0)
        valor_final = loaded_scaler_y.inverse_transform(pred_scaled)
        return round(float(valor_final[0][0]), 2)
    except Exception: return np.nan

# ===================================================================
# 3. INTERFACE GRÁFICA (SIDEBAR PARA ENTRADAS E CONTROLES)
# ===================================================================
with st.sidebar:
    st.header("1. Parâmetros da Via")
    cv = st.number_input('Comprimento da Via (CV, m)', value=1000.0, format="%.2f")
    largura = st.number_input('Largura da VIA (m)', value=7.0, format="%.2f")
    erro = st.number_input('Erro aceitável (e)', value=5.0, format="%.2f")
    desvio_padrao = st.number_input('Desvio padrão (s)', value=10.0, format="%.2f")
    modo_area = st.radio('Modo da Área', ['Automático', 'Manual'], horizontal=True)
    area_manual = st.number_input('Área Manual (m²)', value=225.0, format="%.2f", disabled=(modo_area == 'Automático'))

    if st.button("Calcular Amostragem e Gerar Tabelas", type="primary", use_container_width=True):
        res = calcular_amostras_params(cv, largura, erro, desvio_padrao, modo_area, area_manual)
        if "error" in res:
            st.error(res["error"])
        else:
            st.session_state.amostras_data.clear()
            st.session_state.pci_results.clear()
            n_amostras, area, posicoes = res['n_minimo'], res['Area_m2'], res['Posicoes_m']
            st.session_state.area_amostra_calculada = area
            st.session_state.posicoes = {f"Amostra_{i+1}": posicoes[i] for i in range(n_amostras)}
            for i in range(n_amostras):
                amostra_id = f"Amostra_{i+1}"
                st.session_state.amostras_data[amostra_id] = pd.DataFrame(columns=[('DEFEITO', ''), ('SEVERIDADE', ''), ('Q1', ''), ('Q2', ''), ('Q3', ''), ('Q4', ''), ('TOTAL', ''), ('DENSIDADE', ''), ('VALOR DEDUZIDO', '')])
            st.success(f"{n_amostras} amostras geradas com área de {area:.2f} m².")

    st.header("2. Gerenciar Amostras")
    if st.button("Adicionar Amostra Extra", use_container_width=True):
        area_extra = area_manual if modo_area == 'Manual' else 225.0
        idx = len(st.session_state.amostras_data) + 1
        amostra_id = f"Amostra_Extra_{idx}"
        st.session_state.amostras_data[amostra_id] = pd.DataFrame(columns=[('DEFEITO', ''), ('SEVERIDADE', ''), ('Q1', ''), ('Q2', ''), ('Q3', ''), ('Q4', ''), ('TOTAL', ''), ('DENSIDADE', ''), ('VALOR DEDUZIDO', '')])
        st.session_state.posicoes[amostra_id] = 0.0 # Posição padrão para extras
        st.rerun()

    if st.session_state.amostras_data:
        amostra_a_excluir = st.selectbox("Excluir Amostra:", options=[""] + list(st.session_state.amostras_data.keys()))
        if st.button("Confirmar Exclusão", use_container_width=True):
            if amostra_a_excluir and amostra_a_excluir in st.session_state.amostras_data:
                del st.session_state.amostras_data[amostra_a_excluir]
                if amostra_a_excluir in st.session_state.pci_results:
                    del st.session_state.pci_results[amostra_a_excluir]
                st.rerun()

# ===================================================================
# 4. ÁREA PRINCIPAL (EXIBIÇÃO DAS TABELAS E RESULTADOS)
# ===================================================================
st.title("Ferramenta Integrada de Análise de Pavimentos")

if not st.session_state.amostras_data:
    st.info("⬅️ Utilize o painel à esquerda para calcular o número de amostras.")
else:
    # --- Cálculo e Exibição do PCI Médio ---
    pcis_validos = [pci for pci in st.session_state.pci_results.values() if pd.notna(pci)]
    if len(pcis_validos) > 0:
        pci_medio = np.mean(pcis_validos)
        classificacao, cor = classify_pci_and_get_color(pci_medio)
        st.header(f"Resultado Final da Via")
        col1, col2 = st.columns(2)
        col1.metric(label="PCI Médio da Via", value=f"{pci_medio:.2f}")
        col2.markdown(f"#### Classificação: <span style='color:{cor};'>{classificacao}</span>", unsafe_allow_html=True)
        st.markdown("---")

    st.header("3. Coleta de Dados e Análise por Amostra")
    
    mapa_defeitos = {'BLOCOS DANIFICADOS': 1, 'DEPRESSÕES': 2, 'DANO DE CONTENÇÃO': 3, 'ESPAÇAMENTO EXCESSIVO DAS JUNTAS': 4, 'DIFERENÇA DE ALTURA DO BLOCO': 5, 'ONDULAÇÃO': 6, 'DESLOCAMENTO HORIZONTAL': 7, 'PERDA DE MATERIAL DE REJUNTAMENTO': 8, 'PERDA DE BLOCOS': 9, 'REMENDO': 10, 'DEFORMAÇÃO DE TRILHA DE RODA': 11}
    opcoes_defeito = [''] + [f"{mapa_defeitos.get(d, '??')} - {d}" for d in sorted(mapa_defeitos.keys())]
    opcoes_severidade = [('', ''), ('Alto (A)', 'A'), ('Médio (M)', 'M'), ('Baixo (L)', 'L')]

    for amostra_id, df in st.session_state.amostras_data.items():
        pos = st.session_state.posicoes.get(amostra_id, 0.0)
        area = st.session_state.get('area_amostra_calculada', 225.0)
        
        with st.expander(f"**{amostra_id.replace('_', ' ')}** (Posição: {pos:.1f} m | Área: {area:.2f} m²)", expanded=True):
            
            st.dataframe(df.style.format("{:.2f}", na_rep=""), use_container_width=True)
            
            with st.form(key=f"form_{amostra_id}", clear_on_submit=True):
                c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 2])
                defeito = c1.selectbox("Defeito", options=opcoes_defeito, label_visibility="collapsed")
                severidade = c2.selectbox("Severidade", options=opcoes_severidade, format_func=lambda x: x[0], label_visibility="collapsed")
                q1 = c3.number_input("Q1", value=0.0, format="%.2f", label_visibility="collapsed")
                q2 = c4.number_input("Q2", value=0.0, format="%.2f", label_visibility="collapsed")
                add_button = c5.form_submit_button("Adicionar Linha", use_container_width=True)

                if add_button and defeito and severidade[1]:
                    quantidades = [q1, q2, 0.0, 0.0] # Simplificado para 2 Qs, ajuste se necessário
                    total = sum(quantidades)
                    densidade = (total / area) * 100
                    valor = prever_valor_deduzido(defeito, severidade[1], densidade)
                    nova_linha = {('DEFEITO', ''): defeito, ('SEVERIDADE', ''): severidade[0], ('Q1', ''): q1, ('Q2', ''): q2, ('Q3', ''): 0.0, ('Q4', ''): 0.0, ('TOTAL', ''): total, ('DENSIDADE', ''): densidade, ('VALOR DEDUZIDO', ''): valor}
                    st.session_state.amostras_data[amostra_id] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
                    st.rerun()

            col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
            idx_excluir = col_b1.number_input("Índice p/ Excluir", min_value=0, max_value=max(0, len(df)-1), step=1, key=f"del_idx_{amostra_id}")
            if col_b2.button("Excluir Linha", key=f"del_btn_{amostra_id}", use_container_width=True):
                if 0 <= idx_excluir < len(df):
                    st.session_state.amostras_data[amostra_id] = df.drop(index=idx_excluir).reset_index(drop=True)
                    st.rerun()
            
            if col_b3.button("Calcular PCI desta Amostra", type="primary", key=f"pci_btn_{amostra_id}", use_container_width=True):
                pci_calculado = calcular_pci_para_amostra(df)
                st.session_state.pci_results[amostra_id] = pci_calculado
                st.rerun()

            pci_individual = st.session_state.pci_results.get(amostra_id)
            if pd.notna(pci_individual):
                class_ind, cor_ind = classify_pci_and_get_color(pci_individual)
                st.metric(label=f"PCI da Amostra", value=f"{pci_individual:.2f}", help=f"Classificação: {class_ind}")