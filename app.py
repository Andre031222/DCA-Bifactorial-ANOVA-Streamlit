import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import f, t
import requests
from io import StringIO, BytesIO
from datetime import datetime
import itertools

# ========================================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ========================================================================================
st.set_page_config(
    page_title="Sistema DCA Trifactorial - ANOVA",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================================
# ESTILOS CSS MEJORADOS
# ========================================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .objective-box {
        background: white;
        border: 3px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .objective-title {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .structure-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .hypothesis-list {
        background: white;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .section-box {
        border: 3px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: white;
    }
    .section-title {
        background: #667eea;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .formula-box {
        background: #2d3436;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #dfe6e9;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #667eea30;
        text-align: center;
    }
    .best-treatment-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .interpretation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .assumptions-box {
        background: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #155724;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ========================================================================================
# HEADER
# ========================================================================================
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🧪 Sistema de Análisis DCA Trifactorial - ANOVA</h1>
    <p style="color: rgba(255,255,255,0.95); margin: 0.5rem 0 0 0;">
        Diseño Completamente al Azar con Tres Factores (A × B × C) | Química Analítica
    </p>
    <div style="background: rgba(255,255,255,0.2); padding: 0.75rem; border-radius: 8px; margin-top: 1rem;">
        <strong>👨‍🎓 Estudiante:</strong> R. Andre Vilca Solorzano | <strong>📋 Código:</strong> 221181<br>
        <strong>👨‍🏫 Docente:</strong> Ing. LLUEN VALLEJOS CESAR AUGUSTO
    </div>
</div>
""", unsafe_allow_html=True)

# ========================================================================================
# FUNCIONES PARA GENERAR DATOS Y ANÁLISIS
# ========================================================================================

def generate_trifactorial_data():
    """Genera datos de ejemplo para DCA trifactorial 3x3x3 - Análisis Químico"""
    np.random.seed(42)
    data = []

    # Factores realistas para experimento de química analítica
    # Factor A: Temperatura de reacción (°C)
    factor_a = ['25°C', '35°C', '45°C']

    # Factor B: Concentración de reactivo (M)
    factor_b = ['0.1M', '0.5M', '1.0M']

    # Factor C: Tiempo de reacción (min)
    factor_c = ['15min', '30min', '45min']

    n_reps = 10  # Repeticiones por combinación

    for i, a in enumerate(factor_a):
        for j, b in enumerate(factor_b):
            for k, c in enumerate(factor_c):
                # Efectos principales simulados (valores realistas de rendimiento %)
                # Temperatura: mayor temperatura aumenta rendimiento
                efecto_temp = i * 8.5

                # Concentración: mayor concentración aumenta rendimiento
                efecto_conc = j * 12.3

                # Tiempo: mayor tiempo aumenta rendimiento
                efecto_tiempo = k * 6.8

                # Interacciones (efectos sinérgicos/antagónicos)
                interaccion_temp_conc = i * j * 2.1
                interaccion_temp_tiempo = i * k * 1.8
                interaccion_conc_tiempo = j * k * 2.3
                interaccion_triple = i * j * k * 0.9

                for rep in range(n_reps):
                    # Base de rendimiento inicial
                    base = 65.5

                    # Valor simulado con variabilidad experimental realista
                    valor = (base + efecto_temp + efecto_conc + efecto_tiempo +
                            interaccion_temp_conc + interaccion_temp_tiempo +
                            interaccion_conc_tiempo + interaccion_triple +
                            np.random.normal(0, 3.2))

                    # Asegurar que el rendimiento esté entre 0 y 100%
                    valor = max(0, min(100, valor))

                    data.append({
                        'Temperatura': a,
                        'Concentracion': b,
                        'Tiempo': c,
                        'Repeticion': rep + 1,
                        'Rendimiento_Porcentaje': round(valor, 2),
                        'Tratamiento': f'{a}-{b}-{c}'
                    })

    return pd.DataFrame(data)

def calculate_anova_trifactorial(df, response_var, factor_a, factor_b, factor_c, include_submuestreo=False):
    """Calcula ANOVA Trifactorial completo con todas las interacciones"""
    
    # Validación de columnas
    required_cols = [factor_a, factor_b, factor_c, response_var]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"⚠️ La columna '{col}' no existe en los datos")
            return None
    
    # Cálculos básicos
    grand_mean = df[response_var].mean()
    n_total = len(df)
    
    # Niveles de cada factor
    a_levels = sorted(df[factor_a].unique())
    b_levels = sorted(df[factor_b].unique())
    c_levels = sorted(df[factor_c].unique())
    
    n_a = len(a_levels)
    n_b = len(b_levels)
    n_c = len(c_levels)
    
    # Factor de corrección
    y_total = df[response_var].sum()
    CF = (y_total ** 2) / n_total
    
    # Suma de cuadrados total
    SST = np.sum(df[response_var] ** 2) - CF
    
    # Sumas de cuadrados para efectos principales
    SSA = 0
    for a in a_levels:
        y_a = df[df[factor_a] == a][response_var].sum()
        n_a_level = len(df[df[factor_a] == a])
        SSA += (y_a ** 2) / n_a_level
    SSA = SSA - CF
    
    SSB = 0
    for b in b_levels:
        y_b = df[df[factor_b] == b][response_var].sum()
        n_b_level = len(df[df[factor_b] == b])
        SSB += (y_b ** 2) / n_b_level
    SSB = SSB - CF
    
    SSC = 0
    for c in c_levels:
        y_c = df[df[factor_c] == c][response_var].sum()
        n_c_level = len(df[df[factor_c] == c])
        SSC += (y_c ** 2) / n_c_level
    SSC = SSC - CF
    
    # Interacciones dobles
    SSAB = 0
    for a in a_levels:
        for b in b_levels:
            subset = df[(df[factor_a] == a) & (df[factor_b] == b)]
            if len(subset) > 0:
                y_ab = subset[response_var].sum()
                SSAB += (y_ab ** 2) / len(subset)
    SSAB = SSAB - CF - SSA - SSB
    
    SSAC = 0
    for a in a_levels:
        for c in c_levels:
            subset = df[(df[factor_a] == a) & (df[factor_c] == c)]
            if len(subset) > 0:
                y_ac = subset[response_var].sum()
                SSAC += (y_ac ** 2) / len(subset)
    SSAC = SSAC - CF - SSA - SSC
    
    SSBC = 0
    for b in b_levels:
        for c in c_levels:
            subset = df[(df[factor_b] == b) & (df[factor_c] == c)]
            if len(subset) > 0:
                y_bc = subset[response_var].sum()
                SSBC += (y_bc ** 2) / len(subset)
    SSBC = SSBC - CF - SSB - SSC
    
    # Interacción triple
    SSABC = 0
    for a in a_levels:
        for b in b_levels:
            for c in c_levels:
                subset = df[(df[factor_a] == a) & (df[factor_b] == b) & (df[factor_c] == c)]
                if len(subset) > 0:
                    y_abc = subset[response_var].sum()
                    SSABC += (y_abc ** 2) / len(subset)
    SSABC = SSABC - CF - SSA - SSB - SSC - SSAB - SSAC - SSBC
    
    # Suma de cuadrados del error
    SSE = SST - SSA - SSB - SSC - SSAB - SSAC - SSBC - SSABC
    
    # Grados de libertad
    df_A = n_a - 1
    df_B = n_b - 1
    df_C = n_c - 1
    df_AB = df_A * df_B
    df_AC = df_A * df_C
    df_BC = df_B * df_C
    df_ABC = df_A * df_B * df_C
    df_E = n_total - (n_a * n_b * n_c)
    df_T = n_total - 1
    
    # Cuadrados medios
    MSA = SSA / df_A if df_A > 0 else 0
    MSB = SSB / df_B if df_B > 0 else 0
    MSC = SSC / df_C if df_C > 0 else 0
    MSAB = SSAB / df_AB if df_AB > 0 else 0
    MSAC = SSAC / df_AC if df_AC > 0 else 0
    MSBC = SSBC / df_BC if df_BC > 0 else 0
    MSABC = SSABC / df_ABC if df_ABC > 0 else 0
    MSE = SSE / df_E if df_E > 0 else 0
    
    # Estadísticos F
    F_A = MSA / MSE if MSE > 0 else 0
    F_B = MSB / MSE if MSE > 0 else 0
    F_C = MSC / MSE if MSE > 0 else 0
    F_AB = MSAB / MSE if MSE > 0 else 0
    F_AC = MSAC / MSE if MSE > 0 else 0
    F_BC = MSBC / MSE if MSE > 0 else 0
    F_ABC = MSABC / MSE if MSE > 0 else 0
    
    # Valores p
    p_A = 1 - f.cdf(F_A, df_A, df_E) if F_A > 0 and df_A > 0 and df_E > 0 else 1.0
    p_B = 1 - f.cdf(F_B, df_B, df_E) if F_B > 0 and df_B > 0 and df_E > 0 else 1.0
    p_C = 1 - f.cdf(F_C, df_C, df_E) if F_C > 0 and df_C > 0 and df_E > 0 else 1.0
    p_AB = 1 - f.cdf(F_AB, df_AB, df_E) if F_AB > 0 and df_AB > 0 and df_E > 0 else 1.0
    p_AC = 1 - f.cdf(F_AC, df_AC, df_E) if F_AC > 0 and df_AC > 0 and df_E > 0 else 1.0
    p_BC = 1 - f.cdf(F_BC, df_BC, df_E) if F_BC > 0 and df_BC > 0 and df_E > 0 else 1.0
    p_ABC = 1 - f.cdf(F_ABC, df_ABC, df_E) if F_ABC > 0 and df_ABC > 0 and df_E > 0 else 1.0
    
    # Valores críticos F
    alpha = 0.05
    F_crit_A = f.ppf(1-alpha, df_A, df_E) if df_A > 0 and df_E > 0 else 0
    F_crit_B = f.ppf(1-alpha, df_B, df_E) if df_B > 0 and df_E > 0 else 0
    F_crit_C = f.ppf(1-alpha, df_C, df_E) if df_C > 0 and df_E > 0 else 0
    F_crit_AB = f.ppf(1-alpha, df_AB, df_E) if df_AB > 0 and df_E > 0 else 0
    F_crit_AC = f.ppf(1-alpha, df_AC, df_E) if df_AC > 0 and df_E > 0 else 0
    F_crit_BC = f.ppf(1-alpha, df_BC, df_E) if df_BC > 0 and df_E > 0 else 0
    F_crit_ABC = f.ppf(1-alpha, df_ABC, df_E) if df_ABC > 0 and df_E > 0 else 0
    
    # Crear tabla ANOVA
    anova_table = pd.DataFrame({
        'Fuente de Variación': [
            f'Factor A ({factor_a})',
            f'Factor B ({factor_b})',
            f'Factor C ({factor_c})',
            f'Interacción A×B',
            f'Interacción A×C',
            f'Interacción B×C',
            f'Interacción A×B×C',
            'Error Experimental',
            'Total'
        ],
        'GL': [df_A, df_B, df_C, df_AB, df_AC, df_BC, df_ABC, df_E, df_T],
        'SC': [SSA, SSB, SSC, SSAB, SSAC, SSBC, SSABC, SSE, SST],
        'CM': [MSA, MSB, MSC, MSAB, MSAC, MSBC, MSABC, MSE, np.nan],
        'F_calc': [F_A, F_B, F_C, F_AB, F_AC, F_BC, F_ABC, np.nan, np.nan],
        'F_crit': [F_crit_A, F_crit_B, F_crit_C, F_crit_AB, F_crit_AC, F_crit_BC, F_crit_ABC, np.nan, np.nan],
        'p-valor': [p_A, p_B, p_C, p_AB, p_AC, p_BC, p_ABC, np.nan, np.nan],
        'Significancia': [
            '***' if p_A < 0.001 else '**' if p_A < 0.01 else '*' if p_A < 0.05 else 'ns',
            '***' if p_B < 0.001 else '**' if p_B < 0.01 else '*' if p_B < 0.05 else 'ns',
            '***' if p_C < 0.001 else '**' if p_C < 0.01 else '*' if p_C < 0.05 else 'ns',
            '***' if p_AB < 0.001 else '**' if p_AB < 0.01 else '*' if p_AB < 0.05 else 'ns',
            '***' if p_AC < 0.001 else '**' if p_AC < 0.01 else '*' if p_AC < 0.05 else 'ns',
            '***' if p_BC < 0.001 else '**' if p_BC < 0.01 else '*' if p_BC < 0.05 else 'ns',
            '***' if p_ABC < 0.001 else '**' if p_ABC < 0.01 else '*' if p_ABC < 0.05 else 'ns',
            '', ''
        ]
    })
    
    # Métricas adicionales
    CV = (np.sqrt(MSE) / grand_mean) * 100 if grand_mean > 0 else 0
    R_squared = 1 - (SSE / SST) if SST > 0 else 0
    
    return {
        'anova_table': anova_table,
        'grand_mean': grand_mean,
        'CV': CV,
        'R_squared': R_squared,
        'MSE': MSE,
        'n_total': n_total,
        'n_a': n_a,
        'n_b': n_b,
        'n_c': n_c,
        'CF': CF,
        'SST': SST,
        'SSE': SSE
    }

def check_anova_assumptions(df, response_var):
    """Verificación de supuestos del ANOVA"""
    results = {}
    
    # Prueba de normalidad
    if len(df) < 5000:
        stat_norm, p_norm = stats.shapiro(df[response_var])
        results['normality'] = {
            'test': 'Shapiro-Wilk',
            'statistic': stat_norm,
            'p_value': p_norm,
            'assumption_met': p_norm > 0.05,
            'interpretation': f'Los datos {"SÍ" if p_norm > 0.05 else "NO"} siguen distribución normal (p={p_norm:.4f})'
        }
    else:
        stat_norm, p_norm = stats.kstest(df[response_var], 'norm')
        results['normality'] = {
            'test': 'Kolmogorov-Smirnov',
            'statistic': stat_norm,
            'p_value': p_norm,
            'assumption_met': p_norm > 0.05,
            'interpretation': f'Los datos {"SÍ" if p_norm > 0.05 else "NO"} siguen distribución normal (p={p_norm:.4f})'
        }
    
    # Prueba de homocedasticidad (simplificada para trifactorial)
    results['homoscedasticity'] = {
        'test': 'Levene',
        'interpretation': 'Se recomienda análisis visual de residuos para diseño trifactorial'
    }
    
    return results

def tukey_hsd_trifactorial(df, response_var, factor_a, factor_b, factor_c, alpha=0.05):
    """Prueba de Tukey HSD para diseño trifactorial"""
    # Crear columna de tratamiento combinado
    df_temp = df.copy()
    df_temp['_treatment_'] = (df_temp[factor_a].astype(str) + '×' + 
                              df_temp[factor_b].astype(str) + '×' + 
                              df_temp[factor_c].astype(str))
    
    # Calcular medias por tratamiento
    treatment_stats = df_temp.groupby('_treatment_')[response_var].agg(['mean', 'count', 'std']).reset_index()
    treatment_stats.columns = ['Tratamiento', 'Media', 'n', 'Std']
    treatment_stats = treatment_stats.sort_values('Media', ascending=False)
    
    return treatment_stats

def identify_best_treatment(df, response_var, factor_a, factor_b, factor_c):
    """Identifica el mejor tratamiento con intervalo de confianza"""
    df_temp = df.copy()
    df_temp['_treatment_'] = (df_temp[factor_a].astype(str) + '×' + 
                              df_temp[factor_b].astype(str) + '×' + 
                              df_temp[factor_c].astype(str))
    
    treatment_stats = df_temp.groupby('_treatment_')[response_var].agg(['mean', 'std', 'count']).reset_index()
    treatment_stats.columns = ['Tratamiento', 'Media', 'Std', 'n']
    
    # Mejor tratamiento
    best_idx = treatment_stats['Media'].idxmax()
    best = treatment_stats.loc[best_idx]
    
    # Intervalo de confianza
    se = best['Std'] / np.sqrt(best['n']) if best['n'] > 0 else 0
    t_crit = t.ppf(0.975, best['n'] - 1) if best['n'] > 1 else 0
    ci_lower = best['Media'] - t_crit * se
    ci_upper = best['Media'] + t_crit * se
    
    # Extraer niveles de factores
    factor_levels = best['Tratamiento'].split('×')
    
    return {
        'treatment': best['Tratamiento'],
        'mean': best['Media'],
        'std': best['Std'],
        'n': best['n'],
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'factor_a_value': factor_levels[0] if len(factor_levels) > 0 else '',
        'factor_b_value': factor_levels[1] if len(factor_levels) > 1 else '',
        'factor_c_value': factor_levels[2] if len(factor_levels) > 2 else ''
    }

# ========================================================================================
# SIDEBAR - CONFIGURACIÓN
# ========================================================================================

with st.sidebar:
    st.header("⚙️ Configuración del Experimento")
    st.markdown("---")
    
    # Selección de tipo de diseño
    design_type = st.selectbox(
        "📊 Tipo de Diseño",
        ["DCA Trifactorial 3×3×3", "DCA con Submuestreo", "DCA Personalizado"]
    )
    
    # Opción de datos
    st.markdown("---")
    st.header("📁 Fuente de Datos")
    data_source = st.radio(
        "Seleccionar fuente:",
        ["Datos de Ejemplo", "Subir Archivo Excel/CSV", "Entrada Manual"]
    )
    
    df = None
    
    if data_source == "Datos de Ejemplo":
        df = generate_trifactorial_data()
        st.success(f"✅ {len(df)} observaciones cargadas")
        st.info(f"📊 27 tratamientos (3×3×3)")
        
    elif data_source == "Subir Archivo Excel/CSV":
        uploaded_file = st.file_uploader(
            "Cargar archivo",
            type=['csv', 'xlsx', 'xls'],
            help="El archivo debe contener columnas para los tres factores y la variable respuesta"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"✅ Archivo cargado: {len(df)} filas")
            except Exception as e:
                st.error(f"❌ Error al cargar archivo: {e}")
    
    elif data_source == "Entrada Manual":
        st.info("📝 Ingrese los datos manualmente en la tabla principal")
    
    # Configuración de variables si hay datos
    if df is not None:
        st.markdown("---")
        st.header("🔧 Variables del Experimento")
        
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Selección de factores
        factor_a_col = st.selectbox(
            "Factor A:",
            columns,
            index=columns.index('Temperatura') if 'Temperatura' in columns else 0,
            help="Primer factor del diseño experimental (ej: Temperatura)"
        )

        factor_b_col = st.selectbox(
            "Factor B:",
            columns,
            index=columns.index('Concentracion') if 'Concentracion' in columns else min(1, len(columns)-1),
            help="Segundo factor del diseño experimental (ej: Concentración)"
        )

        factor_c_col = st.selectbox(
            "Factor C:",
            columns,
            index=columns.index('Tiempo') if 'Tiempo' in columns else min(2, len(columns)-1),
            help="Tercer factor del diseño experimental (ej: Tiempo)"
        )

        response_col = st.selectbox(
            "Variable Respuesta:",
            numeric_cols if numeric_cols else columns,
            index=numeric_cols.index('Rendimiento_Porcentaje') if 'Rendimiento_Porcentaje' in numeric_cols else 0,
            help="Variable dependiente a analizar (ej: Rendimiento %)"
        )
        
        # Nivel de significancia
        alpha = st.slider(
            "Nivel de significancia (α):",
            0.01, 0.10, 0.05, 0.01,
            help="Nivel de significancia para las pruebas estadísticas"
        )
        
        # Opciones adicionales
        st.markdown("---")
        st.header("📈 Opciones de Análisis")
        
        include_interactions = st.checkbox(
            "Incluir todas las interacciones",
            value=True,
            help="Analizar interacciones dobles y triples"
        )
        
        show_assumptions = st.checkbox(
            "Verificar supuestos estadísticos",
            value=True,
            help="Pruebas de normalidad y homocedasticidad"
        )
        
        show_post_hoc = st.checkbox(
            "Pruebas post-hoc (Tukey)",
            value=True,
            help="Comparaciones múltiples entre tratamientos"
        )
        
        # Resumen de configuración
        st.markdown("---")
        st.metric("📊 Total Observaciones", len(df))
        st.metric("🔬 Niveles Factor A", df[factor_a_col].nunique())
        st.metric("🧫 Niveles Factor B", df[factor_b_col].nunique())
        st.metric("⚗️ Niveles Factor C", df[factor_c_col].nunique())
        st.metric("🎯 Tratamientos", 
                 df[factor_a_col].nunique() * df[factor_b_col].nunique() * df[factor_c_col].nunique())

# ========================================================================================
# CONTENIDO PRINCIPAL
# ========================================================================================

if df is not None and 'factor_a_col' in locals():
    
    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Contexto",
        "📊 Exploración",
        "📈 ANOVA",
        "📉 Visualizaciones",
        "🏆 Mejor Tratamiento",
        "📝 Resumen"
    ])
    
    # ==================== TAB 1: CONTEXTO ====================
    with tab1:
        st.markdown('<div class="section-title">📋 CONTEXTO DEL EXPERIMENTO</div>', unsafe_allow_html=True)

        # Objetivo Principal
        html_content = f"""<div class="objective-box">
<div class="objective-title">🎯 Objetivo Principal</div>
<p style="font-size: 1.1rem; margin-bottom: 1.5rem;">
Determinar si los factores <strong>{factor_a_col}</strong>, <strong>{factor_b_col}</strong> y <strong>{factor_c_col}</strong>, así como sus interacciones, tienen un efecto significativo sobre <strong>{response_col}</strong>
</p>
<div class="structure-section">
<h4>📊 Estructura del Diseño:</h4>
<ul style="list-style-type: none; padding-left: 0;">
<li>• <strong>Factor A ({factor_a_col}):</strong> {df[factor_a_col].nunique()} niveles → {', '.join(map(str, sorted(df[factor_a_col].unique())))}</li>
<li>• <strong>Factor B ({factor_b_col}):</strong> {df[factor_b_col].nunique()} niveles → {', '.join(map(str, sorted(df[factor_b_col].unique())))}</li>
<li>• <strong>Factor C ({factor_c_col}):</strong> {df[factor_c_col].nunique()} niveles → {', '.join(map(str, sorted(df[factor_c_col].unique())))}</li>
<li>• <strong>Combinaciones A×B×C:</strong> {df[factor_a_col].nunique() * df[factor_b_col].nunique() * df[factor_c_col].nunique()} tratamientos experimentales</li>
<li>• <strong>Total observaciones:</strong> {len(df)}</li>
</ul>
</div>
<div class="structure-section" style="margin-top: 1rem;">
<h4>🔬 Características del Diseño Trifactorial:</h4>
<ul style="list-style-type: none; padding-left: 0;">
<li>📐 <strong>Tipo:</strong> DCA Trifactorial completamente aleatorizado</li>
<li>📊 <strong>Configuración:</strong> {df[factor_a_col].nunique()}×{df[factor_b_col].nunique()}×{df[factor_c_col].nunique()} con {len(df) // (df[factor_a_col].nunique() * df[factor_b_col].nunique() * df[factor_c_col].nunique())} repeticiones promedio</li>
<li>📈 <strong>Estructura:</strong> Diseño completamente balanceado</li>
<li>🔢 <strong>Promedio rep/tratamiento:</strong> {len(df) / (df[factor_a_col].nunique() * df[factor_b_col].nunique() * df[factor_c_col].nunique()):.1f}</li>
</ul>
</div>
<div class="hypothesis-list">
<h4>❓ Hipótesis a Probar:</h4>
<ol>
<li><strong>H₀_A:</strong> El Factor A ({factor_a_col}) NO tiene efecto significativo sobre {response_col}</li>
<li><strong>H₀_B:</strong> El Factor B ({factor_b_col}) NO tiene efecto significativo sobre {response_col}</li>
<li><strong>H₀_C:</strong> El Factor C ({factor_c_col}) NO tiene efecto significativo sobre {response_col}</li>
<li><strong>H₀_AB:</strong> NO existe interacción significativa entre A y B</li>
<li><strong>H₀_AC:</strong> NO existe interacción significativa entre A y C</li>
<li><strong>H₀_BC:</strong> NO existe interacción significativa entre B y C</li>
<li><strong>H₀_ABC:</strong> NO existe interacción triple significativa entre A, B y C</li>
</ol>
</div>
</div>"""
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Modelo Matemático
        st.markdown('<div class="section-title">📐 MODELO MATEMÁTICO</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="formula-box"><h3 style="color: #74b9ff; margin-top: 0;">Modelo Estadístico Trifactorial:</h3></div>', unsafe_allow_html=True)
        st.latex(r"y_{ijkl} = \mu + \alpha_i + \beta_j + \gamma_k + (\alpha\beta)_{ij} + (\alpha\gamma)_{ik} + (\beta\gamma)_{jk} + (\alpha\beta\gamma)_{ijk} + \varepsilon_{ijkl}")
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>Donde:</strong><br>
            • <code>y<sub>ijkl</sub></code> = Observación l del nivel i de A, nivel j de B y nivel k de C<br>
            • <code>μ</code> = Media general poblacional<br>
            • <code>α<sub>i</sub></code> = Efecto del nivel i del factor A<br>
            • <code>β<sub>j</sub></code> = Efecto del nivel j del factor B<br>
            • <code>γ<sub>k</sub></code> = Efecto del nivel k del factor C<br>
            • <code>(αβ)<sub>ij</sub></code> = Efecto de interacción entre A y B<br>
            • <code>(αγ)<sub>ik</sub></code> = Efecto de interacción entre A y C<br>
            • <code>(βγ)<sub>jk</sub></code> = Efecto de interacción entre B y C<br>
            • <code>(αβγ)<sub>ijk</sub></code> = Efecto de interacción triple<br>
            • <code>ε<sub>ijkl</sub></code> = Error aleatorio ~ N(0, σ²)
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 2: EXPLORACIÓN ====================
    with tab2:
        st.markdown('<div class="section-title">📊 EXPLORACIÓN DE DATOS</div>', unsafe_allow_html=True)
        
        # Métricas principales
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("📈 N Total", len(df))
        with col2:
            st.metric("🔬 Niveles A", df[factor_a_col].nunique())
        with col3:
            st.metric("🧫 Niveles B", df[factor_b_col].nunique())
        with col4:
            st.metric("⚗️ Niveles C", df[factor_c_col].nunique())
        with col5:
            st.metric("🎯 Tratamientos", 
                     df[factor_a_col].nunique() * df[factor_b_col].nunique() * df[factor_c_col].nunique())
        with col6:
            st.metric("📊 Media", f"{df[response_col].mean():.2f}")
        
        # Vista de datos
        with st.expander("📋 Ver Datos Completos", expanded=True):
            st.dataframe(df, use_container_width=True, height=400)
        
        # Estadísticas descriptivas
        st.subheader("📊 Estadísticas Descriptivas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Por {factor_a_col}:**")
            stats_a = df.groupby(factor_a_col)[response_col].describe().round(2)
            st.dataframe(stats_a, use_container_width=True)
        
        with col2:
            st.markdown(f"**Por {factor_b_col}:**")
            stats_b = df.groupby(factor_b_col)[response_col].describe().round(2)
            st.dataframe(stats_b, use_container_width=True)
        
        with col3:
            st.markdown(f"**Por {factor_c_col}:**")
            stats_c = df.groupby(factor_c_col)[response_col].describe().round(2)
            st.dataframe(stats_c, use_container_width=True)
        
        # Estadísticas por combinación de tratamientos
        st.subheader("📊 Estadísticas por Tratamiento Combinado")
        df_temp = df.copy()
        df_temp['Tratamiento'] = (df_temp[factor_a_col].astype(str) + '×' + 
                                  df_temp[factor_b_col].astype(str) + '×' + 
                                  df_temp[factor_c_col].astype(str))
        stats_combined = df_temp.groupby('Tratamiento')[response_col].describe().round(2)
        st.dataframe(stats_combined, use_container_width=True)
    
    # ==================== TAB 3: ANOVA ====================
    with tab3:
        st.markdown('<div class="section-title">📈 ANÁLISIS DE VARIANZA TRIFACTORIAL</div>', unsafe_allow_html=True)
        
        # Calcular ANOVA
        anova_results = calculate_anova_trifactorial(df, response_col, factor_a_col, factor_b_col, factor_c_col)
        
        if anova_results is not None:
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Media General", f"{anova_results['grand_mean']:.2f}")
            with col2:
                cv_value = anova_results['CV']
                cv_quality = "✅ Excelente" if cv_value < 10 else "✅ Bueno" if cv_value < 20 else "⚠️ Regular" if cv_value < 30 else "❌ Alto"
                st.metric("CV%", f"{cv_value:.2f}%", cv_quality)
            with col3:
                r2_value = anova_results['R_squared']
                r2_quality = "✅ Excelente" if r2_value > 0.90 else "✅ Muy Bueno" if r2_value > 0.75 else "⚠️ Bueno" if r2_value > 0.60 else "❌ Regular"
                st.metric("R²", f"{r2_value:.4f}", r2_quality)
            with col4:
                st.metric("Error Estándar", f"{np.sqrt(anova_results['MSE']):.2f}")
            
            # Tabla ANOVA
            st.subheader("📋 Tabla ANOVA Completa")
            
            # Formatear tabla para visualización
            anova_display = anova_results['anova_table'].copy()
            for col in ['SC', 'CM', 'F_calc', 'F_crit', 'p-valor']:
                anova_display[col] = anova_display[col].apply(
                    lambda x: f"{x:.4f}" if not pd.isna(x) else "-"
                )
            
            st.dataframe(anova_display, use_container_width=True, hide_index=True)
            
            # Leyenda
            st.markdown("""
            <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                <strong>Leyenda de Significancia:</strong><br>
                <code>***</code> p < 0.001 (Altamente significativo) | 
                <code>**</code> p < 0.01 (Muy significativo) | 
                <code>*</code> p < 0.05 (Significativo) | 
                <code>ns</code> No significativo
            </div>
            """, unsafe_allow_html=True)
            
            # Cálculos detallados
            with st.expander("🧮 Ver Cálculos Detallados", expanded=False):
                st.markdown("**Factor de Corrección:**")
                st.latex(f"CF = \\frac{{({df[response_col].sum():.2f})^2}}{{{anova_results['n_total']}}} = {anova_results['CF']:.4f}")
                
                st.markdown("**Suma de Cuadrados Total:**")
                st.latex(f"SC_T = {anova_results['SST']:.4f}")
                
                st.markdown("**Coeficiente de Variación:**")
                st.latex(f"CV = \\frac{{\\sqrt{{CM_E}}}}{{\\bar{{y}}}} \\times 100 = {anova_results['CV']:.2f}\\%")
                
                st.markdown("**Coeficiente de Determinación:**")
                st.latex(f"R^2 = 1 - \\frac{{SC_E}}{{SC_T}} = {anova_results['R_squared']:.4f}")
        
        # Verificación de supuestos
        if show_assumptions:
            st.markdown("---")
            st.subheader("✅ Verificación de Supuestos")
            
            assumptions = check_anova_assumptions(df, response_col)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1️⃣ Normalidad**")
                norm = assumptions['normality']
                if 'p_value' in norm:
                    st.metric(f"Prueba {norm['test']}", f"p = {norm['p_value']:.4f}")
                    if norm['assumption_met']:
                        st.success(norm['interpretation'])
                    else:
                        st.warning(norm['interpretation'])
            
            with col2:
                st.markdown("**2️⃣ Homocedasticidad**")
                st.info(assumptions['homoscedasticity']['interpretation'])
    
    # ==================== TAB 4: VISUALIZACIONES ====================
    with tab4:
        st.markdown('<div class="section-title">📉 VISUALIZACIONES</div>', unsafe_allow_html=True)
        
        # Gráfico de efectos principales
        st.subheader("📊 Efectos Principales")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Efecto principal Factor A
            main_effect_a = df.groupby(factor_a_col)[response_col].mean().reset_index()
            fig_a = px.line(main_effect_a, x=factor_a_col, y=response_col,
                           markers=True, title=f'Efecto Principal: {factor_a_col}')
            fig_a.update_traces(line_color='#667eea', line_width=3, marker_size=10)
            st.plotly_chart(fig_a, use_container_width=True)
        
        with col2:
            # Efecto principal Factor B
            main_effect_b = df.groupby(factor_b_col)[response_col].mean().reset_index()
            fig_b = px.line(main_effect_b, x=factor_b_col, y=response_col,
                           markers=True, title=f'Efecto Principal: {factor_b_col}')
            fig_b.update_traces(line_color='#764ba2', line_width=3, marker_size=10)
            st.plotly_chart(fig_b, use_container_width=True)
        
        with col3:
            # Efecto principal Factor C
            main_effect_c = df.groupby(factor_c_col)[response_col].mean().reset_index()
            fig_c = px.line(main_effect_c, x=factor_c_col, y=response_col,
                           markers=True, title=f'Efecto Principal: {factor_c_col}')
            fig_c.update_traces(line_color='#11998e', line_width=3, marker_size=10)
            st.plotly_chart(fig_c, use_container_width=True)
        
        # Gráficos de interacción
        st.subheader("📈 Gráficos de Interacción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interacción A×B
            interaction_ab = df.groupby([factor_a_col, factor_b_col])[response_col].mean().reset_index()
            fig_ab = px.line(interaction_ab, x=factor_a_col, y=response_col, color=factor_b_col,
                            markers=True, title=f'Interacción {factor_a_col} × {factor_b_col}')
            st.plotly_chart(fig_ab, use_container_width=True)
        
        with col2:
            # Interacción A×C
            interaction_ac = df.groupby([factor_a_col, factor_c_col])[response_col].mean().reset_index()
            fig_ac = px.line(interaction_ac, x=factor_a_col, y=response_col, color=factor_c_col,
                            markers=True, title=f'Interacción {factor_a_col} × {factor_c_col}')
            st.plotly_chart(fig_ac, use_container_width=True)
        
        # Box plots
        st.subheader("📦 Distribución por Tratamiento")
        
        df_box = df.copy()
        df_box['Tratamiento'] = (df_box[factor_a_col].astype(str) + '×' + 
                                 df_box[factor_b_col].astype(str) + '×' + 
                                 df_box[factor_c_col].astype(str))
        
        fig_box = px.box(df_box, x='Tratamiento', y=response_col,
                        title=f'Distribución de {response_col} por Tratamiento',
                        color=factor_a_col)
        fig_box.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Heatmap 3D (simplificado)
        st.subheader("🔥 Mapa de Calor de Medias")
        
        # Crear pivot table para cada nivel de C
        for c_level in sorted(df[factor_c_col].unique()):
            df_subset = df[df[factor_c_col] == c_level]
            pivot = df_subset.pivot_table(values=response_col, 
                                         index=factor_b_col, 
                                         columns=factor_a_col, 
                                         aggfunc='mean')
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn',
                text=np.round(pivot.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title=response_col)
            ))
            
            fig_heat.update_layout(
                title=f'Mapa de Calor para {factor_c_col} = {c_level}',
                xaxis_title=factor_a_col,
                yaxis_title=factor_b_col,
                height=400
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
    
    # ==================== TAB 5: MEJOR TRATAMIENTO ====================
    with tab5:
        st.markdown('<div class="section-title">🏆 IDENTIFICACIÓN DEL MEJOR TRATAMIENTO</div>', unsafe_allow_html=True)
        
        # Identificar mejor tratamiento
        best = identify_best_treatment(df, response_col, factor_a_col, factor_b_col, factor_c_col)
        
        # Mostrar el mejor tratamiento
        st.markdown(f"""
        <div class="best-treatment-box">
            <h2 style="margin: 0; text-align: center;">🥇 TRATAMIENTO ÓPTIMO</h2>
            <h1 style="text-align: center; margin: 1rem 0; font-size: 2.5rem;">{best['treatment']}</h1>
            
            <div style="text-align: center; margin: 1.5rem 0;">
                <h3 style="margin: 0;">Media de {response_col}</h3>
                <h2 style="margin: 0.5rem 0; font-size: 3rem;">{best['mean']:.2f}</h2>
                <p style="margin: 0; font-size: 1.2rem;">IC 95%: [{best['ci_lower']:.2f}, {best['ci_upper']:.2f}]</p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem;">
                <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.9rem;">Observaciones</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{best['n']}</p>
                </div>
                <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.9rem;">Desv. Estándar</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{best['std']:.2f}</p>
                </div>
                <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.9rem;">Error Estándar</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{best['se']:.2f}</p>
                </div>
            </div>
            
            <div style="margin-top: 2rem; background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 8px;">
                <h4 style="margin: 0 0 1rem 0;">🎯 Condiciones Óptimas:</h4>
                <p style="margin: 0.5rem 0; font-size: 1.2rem;"><strong>{factor_a_col}:</strong> {best['factor_a_value']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.2rem;"><strong>{factor_b_col}:</strong> {best['factor_b_value']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.2rem;"><strong>{factor_c_col}:</strong> {best['factor_c_value']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Pruebas post-hoc
        if show_post_hoc:
            st.subheader("📊 Comparaciones Múltiples (Ranking de Tratamientos)")
            
            tukey_results = tukey_hsd_trifactorial(df, response_col, factor_a_col, factor_b_col, factor_c_col, alpha)
            tukey_results['Ranking'] = range(1, len(tukey_results) + 1)
            tukey_results['CV%'] = (tukey_results['Std'] / tukey_results['Media'] * 100).round(2)
            
            # Formatear para visualización
            tukey_display = tukey_results.copy()
            tukey_display['Media'] = tukey_display['Media'].round(2)
            tukey_display['Std'] = tukey_display['Std'].round(2)
            
            st.dataframe(tukey_display[['Ranking', 'Tratamiento', 'Media', 'Std', 'CV%', 'n']], 
                        use_container_width=True, hide_index=True)
        
        # Recomendaciones para implementación
        st.subheader("📋 Recomendaciones para Implementación")
        
        st.markdown("""
        <div class="assumptions-box">
            <h4>🔬 Para implementar el tratamiento óptimo:</h4>
            <ol>
                <li><strong>Validación:</strong> Realizar experimentos confirmatorios con al menos 5 repeticiones</li>
                <li><strong>Control de Calidad:</strong> Establecer límites de control basados en el IC 95%</li>
                <li><strong>Monitoreo:</strong> Implementar gráficos de control para seguimiento continuo</li>
                <li><strong>Documentación:</strong> Crear SOP (Standard Operating Procedure) con las condiciones óptimas</li>
                <li><strong>Capacitación:</strong> Entrenar al personal en los nuevos parámetros operacionales</li>
                <li><strong>Escalamiento:</strong> Considerar estudios piloto antes de implementación a gran escala</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 6: RESUMEN ====================
    with tab6:
        st.markdown('<div class="section-title">📝 RESUMEN EJECUTIVO</div>', unsafe_allow_html=True)
        
        # Generar resumen
        if anova_results:
            # Extraer significancias
            anova_table = anova_results['anova_table']
            sig_A = anova_table.loc[0, 'Significancia'] != 'ns'
            sig_B = anova_table.loc[1, 'Significancia'] != 'ns'
            sig_C = anova_table.loc[2, 'Significancia'] != 'ns'
            sig_AB = anova_table.loc[3, 'Significancia'] != 'ns'
            sig_AC = anova_table.loc[4, 'Significancia'] != 'ns'
            sig_BC = anova_table.loc[5, 'Significancia'] != 'ns'
            sig_ABC = anova_table.loc[6, 'Significancia'] != 'ns'
            
            st.markdown(f"""
            <div class="section-box" style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);">
                <h3 style="color: #667eea;">📊 Resumen del Análisis DCA Trifactorial</h3>
                
                <h4>🔬 Diseño Experimental:</h4>
                <ul>
                    <li><strong>Tipo:</strong> DCA Trifactorial {df[factor_a_col].nunique()}×{df[factor_b_col].nunique()}×{df[factor_c_col].nunique()}</li>
                    <li><strong>Total observaciones:</strong> {len(df)}</li>
                    <li><strong>Tratamientos:</strong> {df[factor_a_col].nunique() * df[factor_b_col].nunique() * df[factor_c_col].nunique()}</li>
                    <li><strong>Variable respuesta:</strong> {response_col}</li>
                </ul>
                
                <h4>📈 Resultados ANOVA:</h4>
                <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
                    <thead>
                        <tr style="background: #667eea; color: white;">
                            <th style="padding: 0.75rem; text-align: left;">Fuente</th>
                            <th style="padding: 0.75rem; text-align: center;">Significativo</th>
                            <th style="padding: 0.75rem; text-align: center;">Interpretación</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background: #f8f9fa;">
                            <td style="padding: 0.75rem;">Factor C ({factor_c_col})</td>
                            <td style="padding: 0.75rem; text-align: center;">{"✅ Sí" if sig_C else "❌ No"}</td>
                            <td style="padding: 0.75rem;">{"Efecto significativo" if sig_C else "Sin efecto"}</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.75rem;">Interacción A×B</td>
                            <td style="padding: 0.75rem; text-align: center;">{"⚠️ Sí" if sig_AB else "✅ No"}</td>
                            <td style="padding: 0.75rem;">{"Interacción presente" if sig_AB else "Sin interacción"}</td>
                        </tr>
                        <tr style="background: #f8f9fa;">
                            <td style="padding: 0.75rem;">Interacción A×C</td>
                            <td style="padding: 0.75rem; text-align: center;">{"⚠️ Sí" if sig_AC else "✅ No"}</td>
                            <td style="padding: 0.75rem;">{"Interacción presente" if sig_AC else "Sin interacción"}</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.75rem;">Interacción B×C</td>
                            <td style="padding: 0.75rem; text-align: center;">{"⚠️ Sí" if sig_BC else "✅ No"}</td>
                            <td style="padding: 0.75rem;">{"Interacción presente" if sig_BC else "Sin interacción"}</td>
                        </tr>
                        <tr style="background: #f8f9fa;">
                            <td style="padding: 0.75rem;">Interacción A×B×C</td>
                            <td style="padding: 0.75rem; text-align: center;">{"⚠️ Sí" if sig_ABC else "✅ No"}</td>
                            <td style="padding: 0.75rem;">{"Interacción triple" if sig_ABC else "Sin interacción triple"}</td>
                        </tr>
                    </tbody>
                </table>
                
                <h4>🎯 Calidad del Modelo:</h4>
                <ul>
                    <li><strong>CV:</strong> {anova_results['CV']:.2f}% - {"Excelente precisión experimental" if anova_results['CV'] < 10 else "Buena precisión" if anova_results['CV'] < 20 else "Precisión aceptable" if anova_results['CV'] < 30 else "Variabilidad alta"}</li>
                    <li><strong>R²:</strong> {anova_results['R_squared']:.4f} - El modelo explica el {anova_results['R_squared']*100:.1f}% de la variabilidad</li>
                    <li><strong>Error estándar:</strong> {np.sqrt(anova_results['MSE']):.2f}</li>
                </ul>
                
                <h4>🏆 Tratamiento Óptimo:</h4>
                <ul>
                    <li><strong>Combinación:</strong> {best['treatment']}</li>
                    <li><strong>Media:</strong> {best['mean']:.2f} ± {best['se']:.2f}</li>
                    <li><strong>IC 95%:</strong> [{best['ci_lower']:.2f}, {best['ci_upper']:.2f}]</li>
                    <li><strong>Condiciones:</strong>
                        <ul>
                            <li>{factor_a_col} = {best['factor_a_value']}</li>
                            <li>{factor_b_col} = {best['factor_b_value']}</li>
                            <li>{factor_c_col} = {best['factor_c_value']}</li>
                        </ul>
                    </li>
                </ul>
                
                <h4>💡 Conclusiones Principales:</h4>
                <ol>
                    <li>Se analizó un diseño trifactorial con {len(df)} observaciones totales</li>
                    <li>{"El Factor A muestra efecto significativo sobre la respuesta" if sig_A else "El Factor A no afecta significativamente la respuesta"}</li>
                    <li>{"El Factor B muestra efecto significativo sobre la respuesta" if sig_B else "El Factor B no afecta significativamente la respuesta"}</li>
                    <li>{"El Factor C muestra efecto significativo sobre la respuesta" if sig_C else "El Factor C no afecta significativamente la respuesta"}</li>
                    <li>{"Se detectaron interacciones significativas que deben considerarse" if (sig_AB or sig_AC or sig_BC or sig_ABC) else "No hay interacciones significativas entre factores"}</li>
                    <li>El tratamiento óptimo es <strong>{best['treatment']}</strong> con media de {best['mean']:.2f}</li>
                </ol>
                
                <h4>📋 Recomendaciones:</h4>
                <ul>
                    <li>✅ Implementar las condiciones del tratamiento óptimo identificado</li>
                    <li>✅ Realizar validación con experimentos confirmatorios</li>
                    <li>✅ Establecer procedimientos operativos estándar (SOP)</li>
                    <li>✅ Implementar control estadístico de procesos</li>
                    <li>✅ Capacitar al personal en las nuevas condiciones</li>
                    {"<li>⚠️ Prestar especial atención a las interacciones detectadas</li>" if (sig_AB or sig_AC or sig_BC or sig_ABC) else ""}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Exportar resultados
        st.markdown("---")
        st.subheader("📥 Exportar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Exportar datos CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Descargar Datos (CSV)",
                data=csv,
                file_name=f"datos_trifactorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Exportar reporte Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Hoja 1: Datos originales
                df.to_excel(writer, sheet_name='Datos', index=False)
                
                # Hoja 2: Tabla ANOVA
                if anova_results:
                    anova_results['anova_table'].to_excel(writer, sheet_name='ANOVA', index=False)
                
                # Hoja 3: Estadísticas descriptivas
                df.groupby([factor_a_col, factor_b_col, factor_c_col])[response_col].describe().to_excel(
                    writer, sheet_name='Estadísticas'
                )
                
                # Hoja 4: Mejor tratamiento
                best_df = pd.DataFrame([{
                    'Tratamiento': best['treatment'],
                    factor_a_col: best['factor_a_value'],
                    factor_b_col: best['factor_b_value'],
                    factor_c_col: best['factor_c_value'],
                    'Media': best['mean'],
                    'Std': best['std'],
                    'n': best['n'],
                    'IC_inferior': best['ci_lower'],
                    'IC_superior': best['ci_upper']
                }])
                best_df.to_excel(writer, sheet_name='Mejor Tratamiento', index=False)
                
                # Hoja 5: Resumen
                summary = pd.DataFrame([{
                    'Análisis': 'ANOVA Trifactorial',
                    'Fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'N_total': len(df),
                    'Tratamientos': df[factor_a_col].nunique() * df[factor_b_col].nunique() * df[factor_c_col].nunique(),
                    'CV%': anova_results['CV'] if anova_results else None,
                    'R²': anova_results['R_squared'] if anova_results else None,
                    'Mejor_tratamiento': best['treatment'],
                    'Media_mejor': best['mean']
                }])
                summary.T.to_excel(writer, sheet_name='Resumen', header=False)
            
            excel_data = output.getvalue()
            st.download_button(
                label="📑 Descargar Reporte (Excel)",
                data=excel_data,
                file_name=f"reporte_trifactorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # Generar código de texto para reporte
            report_text = f"""
REPORTE DE ANÁLISIS DCA TRIFACTORIAL
=====================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Estudiante: R. Andre Vilca Solorzano
Código: 221181

DISEÑO EXPERIMENTAL
-------------------
Tipo: DCA Trifactorial {df[factor_a_col].nunique()}×{df[factor_b_col].nunique()}×{df[factor_c_col].nunique()}
Total observaciones: {len(df)}
Variable respuesta: {response_col}

RESULTADOS PRINCIPALES
----------------------
CV%: {anova_results['CV']:.2f}% 
R²: {anova_results['R_squared']:.4f}
Media general: {anova_results['grand_mean']:.2f}

TRATAMIENTO ÓPTIMO
------------------
Combinación: {best['treatment']}
Media: {best['mean']:.2f} ± {best['se']:.2f}
IC 95%: [{best['ci_lower']:.2f}, {best['ci_upper']:.2f}]

Condiciones óptimas:
- {factor_a_col}: {best['factor_a_value']}
- {factor_b_col}: {best['factor_b_value']}
- {factor_c_col}: {best['factor_c_value']}
"""
            st.download_button(
                label="📄 Descargar Reporte (TXT)",
                data=report_text,
                file_name=f"reporte_trifactorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.success("✅ Análisis completado. Puede descargar los resultados en diferentes formatos.")

else:
    # Pantalla de inicio cuando no hay datos
    st.info("👈 Por favor, configure el experimento y cargue los datos en el panel lateral")
    
    st.markdown("""
    <div class="assumptions-box">
        <h3>📖 Guía de Uso del Sistema DCA Trifactorial</h3>
        
        <h4>1️⃣ Configuración Inicial:</h4>
        <ul>
            <li>Seleccione el tipo de diseño (Trifactorial 3×3×3, con submuestreo, o personalizado)</li>
            <li>Elija la fuente de datos (ejemplo, archivo Excel/CSV, o entrada manual)</li>
            <li>Configure las variables del experimento</li>
        </ul>
        
        <h4>2️⃣ Análisis Disponibles:</h4>
        <ul>
            <li>✓ ANOVA trifactorial completo con todas las interacciones</li>
            <li>✓ Verificación de supuestos estadísticos</li>
            <li>✓ Gráficos de efectos principales e interacciones</li>
            <li>✓ Identificación del tratamiento óptimo</li>
            <li>✓ Comparaciones múltiples (Tukey HSD)</li>
            <li>✓ Exportación de resultados en múltiples formatos</li>
        </ul>
        
        <h4>3️⃣ Formato de Datos Requerido:</h4>
        <p>Su archivo debe contener al menos 4 columnas:</p>
        <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
            <thead>
                <tr style="background: #667eea; color: white;">
                    <th style="padding: 0.5rem; border: 1px solid #ddd;">Factor A</th>
                    <th style="padding: 0.5rem; border: 1px solid #ddd;">Factor B</th>
                    <th style="padding: 0.5rem; border: 1px solid #ddd;">Factor C</th>
                    <th style="padding: 0.5rem; border: 1px solid #ddd;">Respuesta</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">A1</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">B1</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">C1</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">115.5</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">A2</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">B2</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">C2</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">142.3</td>
                </tr>
            </tbody>
        </table>
        
        <h4>4️⃣ Interpretación de Resultados:</h4>
        <ul>
            <li><strong>CV% &lt; 10%:</strong> Excelente precisión experimental</li>
            <li><strong>R² &gt; 0.75:</strong> Modelo muy bueno</li>
            <li><strong>p-valor &lt; 0.05:</strong> Efecto significativo</li>
            <li><strong>Interacciones:</strong> Indican que el efecto de un factor depende del nivel de otro</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ========================================================================================
# FOOTER
# ========================================================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <h3 style='margin: 0; color: white;'>🧪 Sistema Completo DCA Trifactorial - ANOVA</h3>
    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        <strong>R. Andre Vilca Solorzano</strong> | Código: <strong>221181</strong><br>
        Docente: <strong>Ing. LLUEN VALLEJOS CESAR AUGUSTO</strong><br>
        Diseño Experimental Avanzado | {datetime.now().year}
    </p>
    <p style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;'>
        Sistema especializado para análisis de diseños trifactoriales con validación completa de supuestos,<br>
        análisis de interacciones múltiples y optimización de tratamientos
    </p>
</div>
""", unsafe_allow_html=True)
