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

# ========================================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ========================================================================================
st.set_page_config(
    page_title="Sistema DCA Bifactorial - ANOVA",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================================
# ESTILOS CSS PERSONALIZADOS
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
    <h1 style="color: white; margin: 0;">🧪 Sistema de Análisis DCA Bifactorial - ANOVA</h1>
    <p style="color: rgba(255,255,255,0.95); margin: 0.5rem 0 0 0;">
        Diseño Completamente al Azar con Dos Factores (A y B) | Química Analítica
    </p>
    <div style="background: rgba(255,255,255,0.2); padding: 0.75rem; border-radius: 8px; margin-top: 1rem; display: inline-block;">
        <strong>👨‍🎓 Estudiante:</strong> R. Andre Vilca Solorzano | <strong>📋 Código:</strong> 221181
    </div>
    <div style="background: rgba(255,255,255,0.15); padding: 0.5rem; border-radius: 6px; margin-top: 0.5rem; display: inline-block;">
        <strong>👨‍🏫 Docente:</strong> Ing. LLUEN VALLEJOS CESAR AUGUSTO
    </div>
</div>
""", unsafe_allow_html=True)

# ========================================================================================
# CONFIGURACIÓN Y DATOS
# ========================================================================================

EXAMPLE_DATASETS = {
    "Dataset 1 - Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Dataset 2 - Tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
}

MODELS = {
    "Modelo 1": {
        "name": "Balanceado con submuestreo balanceado",
        "description": "a=3, b=3, n=10, m=3",
        "parametros": "3 niveles Factor A × 3 niveles Factor B × 10 repeticiones × 3 submuestras = 270 observaciones",
        "estructura": "Diseño completamente balanceado donde cada combinación de factores tiene el mismo número de repeticiones y submuestras"
    },
    "Modelo 2": {
        "name": "No balanceado",
        "description": "a=3, b=3, n variable",
        "parametros": "3 × 3 tratamientos con diferente número de repeticiones (n₁₁, n₁₂, ..., n₃₃)",
        "estructura": "El número de repeticiones varía por celda debido a pérdidas experimentales o restricciones prácticas"
    },
    "Modelo 3": {
        "name": "Balanceado con submuestreo balanceado",
        "description": "a=2, b=2, n=9, m=3",
        "parametros": "2 niveles × 2 niveles × 9 repeticiones × 3 submuestras = 108 observaciones",
        "estructura": "Diseño factorial 2×2 completamente balanceado con submuestreo uniforme"
    },
    "Modelo 4": {
        "name": "Balanceado con submuestreo no balanceado",
        "description": "a=2, b=2, n=2, m variable",
        "parametros": "Repeticiones constantes pero submuestras variables (m₁₁, m₁₂, m₂₁, m₂₂)",
        "estructura": "Unidades experimentales balanceadas pero número de submuestras varía por limitaciones técnicas"
    },
    "Modelo 5": {
        "name": "No balanceado con submuestreo balanceado",
        "description": "a=2, b=2, n variable, m=6",
        "parametros": "Repeticiones variables (n₁, n₂) pero submuestras constantes (m=6)",
        "estructura": "Número de unidades experimentales varía pero cada una tiene el mismo número de submuestras"
    },
    "Modelo 6": {
        "name": "No balanceado con submuestreo no balanceado",
        "description": "a=3, b=3, n y m variables",
        "parametros": "Estructura completamente irregular: nᵢⱼ y mᵢⱼₖ variables por celda",
        "estructura": "Máxima flexibilidad: tanto repeticiones como submuestras varían según disponibilidad experimental"
    }
}

def generate_example_data(model_num):
    """Genera datos realistas de química analítica"""
    np.random.seed(42 + model_num)
    data = []
    
    if model_num == 1:
        temperaturas = ['25°C', '50°C', '75°C']
        concentraciones = ['0.1M', '0.5M', '1.0M']
        
        for i, temp in enumerate(temperaturas):
            for j, conc in enumerate(concentraciones):
                efecto_temp = i * 35
                efecto_conc = j * 28
                interaccion = (i * j) * 6
                
                for rep in range(10):
                    for sub in range(3):
                        base = 185 + efecto_temp + efecto_conc + interaccion
                        error_exp = np.random.normal(0, 8)
                        error_sub = np.random.normal(0, 3)
                        valor = base + error_exp + error_sub
                        
                        data.append({
                            'Temperatura': temp,
                            'Concentracion': conc,
                            'Repeticion': rep + 1,
                            'Submuestra': sub + 1,
                            'Rendimiento': round(max(0, valor), 2),
                            'Tratamiento': f'{temp}-{conc}',
                            'FactorA': temp,
                            'FactorB': conc
                        })
    
    elif model_num == 2:
        ph_levels = ['pH3', 'pH7', 'pH11']
        solventes = ['Etanol', 'Metanol', 'Acetona']
        reps_matrix = [[12, 15, 10], [11, 13, 14], [9, 16, 12]]
        
        for i, ph in enumerate(ph_levels):
            for j, solv in enumerate(solventes):
                efecto_ph = (i - 1) * 25
                efecto_solv = j * 18
                interaccion = abs(i - 1) * j * 4
                
                n_reps = reps_matrix[i][j]
                for rep in range(n_reps):
                    base = 210 + efecto_ph + efecto_solv + interaccion
                    valor = base + np.random.normal(0, 12)
                    
                    data.append({
                        'pH': ph,
                        'Solvente': solv,
                        'Repeticion': rep + 1,
                        'Rendimiento': round(max(0, valor), 2),
                        'Tratamiento': f'{ph}-{solv}',
                        'FactorA': ph,
                        'FactorB': solv
                    })
    
    elif model_num == 3:
        levaduras = ['Levadura_A', 'Levadura_B']
        tiempos = ['24h', '48h']
        
        for i, lev in enumerate(levaduras):
            for j, tiempo in enumerate(tiempos):
                efecto_lev = i * 32
                efecto_tiempo = j * 45
                interaccion = i * j * 8
                
                for rep in range(9):
                    for sub in range(3):
                        base = 195 + efecto_lev + efecto_tiempo + interaccion
                        valor = base + np.random.normal(0, 11) + np.random.normal(0, 4)
                        
                        data.append({
                            'Levadura': lev,
                            'Tiempo': tiempo,
                            'Repeticion': rep + 1,
                            'Submuestra': sub + 1,
                            'Rendimiento': round(min(100, max(0, valor)), 2),
                            'Tratamiento': f'{lev}-{tiempo}',
                            'FactorA': lev,
                            'FactorB': tiempo
                        })
    
    elif model_num == 4:
        agitaciones = ['100rpm', '300rpm']
        temperaturas = ['15°C', '25°C']
        submuestras_matrix = [[[8, 7], [9, 10]], [[6, 11], [8, 9]]]
        
        for i, agit in enumerate(agitaciones):
            for j, temp in enumerate(temperaturas):
                efecto_agit = i * 28
                efecto_temp = j * 22
                interaccion = i * j * 5
                
                for rep in range(2):
                    n_sub = submuestras_matrix[i][j][rep]
                    for sub in range(n_sub):
                        base = 198 + efecto_agit + efecto_temp + interaccion
                        valor = base + np.random.normal(0, 9)
                        
                        data.append({
                            'Agitacion': agit,
                            'Temperatura': temp,
                            'Repeticion': rep + 1,
                            'Submuestra': sub + 1,
                            'Rendimiento': round(min(100, max(0, valor)), 2),
                            'Tratamiento': f'{agit}-{temp}',
                            'FactorA': agit,
                            'FactorB': temp
                        })
    
    elif model_num == 5:
        carbones = ['Carbon_Activado', 'Carbon_Vegetal']
        ph_levels = ['pH4', 'pH7']
        reps_matrix = [[4, 5], [3, 6]]
        
        for i, carbon in enumerate(carbones):
            for j, ph in enumerate(ph_levels):
                efecto_carbon = i * 24
                efecto_ph = j * 31
                interaccion = i * j * 7
                
                n_reps = reps_matrix[i][j]
                for rep in range(n_reps):
                    for sub in range(6):
                        base = 203 + efecto_carbon + efecto_ph + interaccion
                        valor = base + np.random.normal(0, 10)
                        
                        data.append({
                            'Tipo_Carbon': carbon,
                            'pH': ph,
                            'Repeticion': rep + 1,
                            'Submuestra': sub + 1,
                            'Rendimiento': round(max(0, valor), 2),
                            'Tratamiento': f'{carbon}-{ph}',
                            'FactorA': carbon,
                            'FactorB': ph
                        })
    
    else:  # Modelo 6
        catalizadores = ['Cat_Pt', 'Cat_Pd', 'Cat_Ni']
        presiones = ['1atm', '5atm', '10atm']
        
        configuraciones = [
            {'cat': 'Cat_Pt', 'pres': '1atm', 'reps': 4, 'subs': [7, 6, 8, 5]},
            {'cat': 'Cat_Pt', 'pres': '5atm', 'reps': 3, 'subs': [9, 7, 10]},
            {'cat': 'Cat_Pt', 'pres': '10atm', 'reps': 5, 'subs': [6, 8, 7, 9, 6]},
            {'cat': 'Cat_Pd', 'pres': '1atm', 'reps': 3, 'subs': [8, 9, 7]},
            {'cat': 'Cat_Pd', 'pres': '5atm', 'reps': 4, 'subs': [10, 8, 9, 11]},
            {'cat': 'Cat_Pd', 'pres': '10atm', 'reps': 2, 'subs': [7, 12]},
            {'cat': 'Cat_Ni', 'pres': '1atm', 'reps': 5, 'subs': [6, 7, 8, 6, 9]},
            {'cat': 'Cat_Ni', 'pres': '5atm', 'reps': 3, 'subs': [10, 9, 11]},
            {'cat': 'Cat_Ni', 'pres': '10atm', 'reps': 4, 'subs': [8, 7, 9, 8]}
        ]
        
        for config in configuraciones:
            cat_idx = catalizadores.index(config['cat'])
            pres_idx = presiones.index(config['pres'])
            
            efecto_cat = cat_idx * 22
            efecto_pres = pres_idx * 30
            interaccion = cat_idx * pres_idx * 4
            
            for rep in range(config['reps']):
                n_sub = config['subs'][rep]
                for sub in range(n_sub):
                    base = 192 + efecto_cat + efecto_pres + interaccion
                    valor = base + np.random.normal(0, 13)
                    
                    data.append({
                        'Catalizador': config['cat'],
                        'Presion': config['pres'],
                        'Repeticion': rep + 1,
                        'Submuestra': sub + 1,
                        'Rendimiento': round(min(100, max(0, valor)), 2),
                        'Tratamiento': f"{config['cat']}-{config['pres']}",
                        'FactorA': config['cat'],
                        'FactorB': config['pres']
                    })
    
    return pd.DataFrame(data)

def calculate_anova_bifactorial(df, response_var, factor_a, factor_b):
    """Calcula ANOVA Bifactorial completo"""
    if factor_a not in df.columns or factor_b not in df.columns:
        st.error(f"⚠️ Los factores '{factor_a}' y/o '{factor_b}' no existen")
        return None
    
    grand_mean = df[response_var].mean()
    n_total = len(df)
    
    a_levels = sorted(df[factor_a].unique())
    n_a = len(a_levels)
    a_means = df.groupby(factor_a)[response_var].mean()
    a_counts = df.groupby(factor_a)[response_var].count()
    
    b_levels = sorted(df[factor_b].unique())
    n_b = len(b_levels)
    b_means = df.groupby(factor_b)[response_var].mean()
    b_counts = df.groupby(factor_b)[response_var].count()
    
    ab_means = df.groupby([factor_a, factor_b])[response_var].mean()
    ab_counts = df.groupby([factor_a, factor_b])[response_var].count()
    
    y_total = df[response_var].sum()
    CF = (y_total ** 2) / n_total
    SST = np.sum(df[response_var] ** 2) - CF
    
    SSA = 0
    for a in a_levels:
        y_a = df[df[factor_a] == a][response_var].sum()
        n_a_level = a_counts[a]
        SSA += (y_a ** 2) / n_a_level
    SSA = SSA - CF
    
    SSB = 0
    for b in b_levels:
        y_b = df[df[factor_b] == b][response_var].sum()
        n_b_level = b_counts[b]
        SSB += (y_b ** 2) / n_b_level
    SSB = SSB - CF
    
    SSAB = 0
    for a in a_levels:
        for b in b_levels:
            subset = df[(df[factor_a] == a) & (df[factor_b] == b)]
            if len(subset) > 0:
                y_ab = subset[response_var].sum()
                n_ab = len(subset)
                SSAB += (y_ab ** 2) / n_ab
    SSAB = SSAB - CF - SSA - SSB
    
    SSE = SST - SSA - SSB - SSAB
    
    df_A = n_a - 1
    df_B = n_b - 1
    df_AB = df_A * df_B
    df_E = n_total - (n_a * n_b)
    df_T = n_total - 1
    
    MSA = SSA / df_A if df_A > 0 else 0
    MSB = SSB / df_B if df_B > 0 else 0
    MSAB = SSAB / df_AB if df_AB > 0 else 0
    MSE = SSE / df_E if df_E > 0 else 0
    
    F_A = MSA / MSE if MSE > 0 else 0
    F_B = MSB / MSE if MSE > 0 else 0
    F_AB = MSAB / MSE if MSE > 0 else 0
    
    p_A = 1 - f.cdf(F_A, df_A, df_E) if F_A > 0 and df_A > 0 and df_E > 0 else 1.0
    p_B = 1 - f.cdf(F_B, df_B, df_E) if F_B > 0 and df_B > 0 and df_E > 0 else 1.0
    p_AB = 1 - f.cdf(F_AB, df_AB, df_E) if F_AB > 0 and df_AB > 0 and df_E > 0 else 1.0
    
    F_crit_A = f.ppf(0.95, df_A, df_E) if df_A > 0 and df_E > 0 else 0
    F_crit_B = f.ppf(0.95, df_B, df_E) if df_B > 0 and df_E > 0 else 0
    F_crit_AB = f.ppf(0.95, df_AB, df_E) if df_AB > 0 and df_E > 0 else 0
    
    anova_table = pd.DataFrame({
        'Fuente de Variación': [
            f'Factor A ({factor_a})', 
            f'Factor B ({factor_b})', 
            f'Interacción A×B', 
            'Error Experimental', 
            'Total'
        ],
        'GL': [df_A, df_B, df_AB, df_E, df_T],
        'SC': [SSA, SSB, SSAB, SSE, SST],
        'CM': [MSA, MSB, MSAB, MSE, np.nan],
        'F_calc': [F_A, F_B, F_AB, np.nan, np.nan],
        'F_crit': [F_crit_A, F_crit_B, F_crit_AB, np.nan, np.nan],
        'p-valor': [p_A, p_B, p_AB, np.nan, np.nan],
        'Significancia': [
            '***' if p_A < 0.001 else '**' if p_A < 0.01 else '*' if p_A < 0.05 else 'ns',
            '***' if p_B < 0.001 else '**' if p_B < 0.01 else '*' if p_B < 0.05 else 'ns',
            '***' if p_AB < 0.001 else '**' if p_AB < 0.01 else '*' if p_AB < 0.05 else 'ns',
            '', ''
        ]
    })
    
    CV = (np.sqrt(MSE) / grand_mean) * 100 if grand_mean > 0 else 0
    R_squared = (SSA + SSB + SSAB) / SST if SST > 0 else 0
    model_effectiveness = "Excelente" if R_squared > 0.90 else "Muy Bueno" if R_squared > 0.75 else "Bueno" if R_squared > 0.60 else "Regular" if R_squared > 0.40 else "Pobre"
    
    return {
        'anova_table': anova_table, 'grand_mean': grand_mean, 'CV': CV,
        'R_squared': R_squared, 'model_effectiveness': model_effectiveness, 'MSE': MSE,
        'a_means': a_means, 'b_means': b_means, 'ab_means': ab_means,
        'a_counts': a_counts, 'b_counts': b_counts, 'ab_counts': ab_counts,
        'significant_A': F_A > F_crit_A, 'significant_B': F_B > F_crit_B,
        'significant_AB': F_AB > F_crit_AB, 'F_A': F_A, 'F_B': F_B, 'F_AB': F_AB,
        'p_A': p_A, 'p_B': p_B, 'p_AB': p_AB, 'CF': CF,
        'SSA': SSA, 'SSB': SSB, 'SSAB': SSAB, 'SSE': SSE, 'SST': SST,
        'n_total': n_total, 'n_a': n_a, 'n_b': n_b
    }

def check_anova_assumptions(df, response_var, factor_a, factor_b):
    """Verifica supuestos del ANOVA"""
    results = {}
    
    if len(df) < 5000:
        stat_norm, p_norm = stats.shapiro(df[response_var])
        results['normality'] = {
            'test': 'Shapiro-Wilk', 'statistic': stat_norm, 'p_value': p_norm,
            'assumption_met': p_norm > 0.05,
            'interpretation': f'Los datos {"SÍ" if p_norm > 0.05 else "NO"} siguen distribución normal (p={p_norm:.4f})'
        }
    else:
        results['normality'] = {
            'test': 'Kolmogorov-Smirnov', 'assumption_met': None,
            'interpretation': 'Muestra muy grande, considerar TLC'
        }
    
    df_temp = df.copy()
    df_temp['_group_'] = df_temp[factor_a].astype(str) + '_' + df_temp[factor_b].astype(str)
    groups = [df_temp[df_temp['_group_'] == g][response_var].values for g in df_temp['_group_'].unique()]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) > 1 and all(len(g) > 1 for g in groups):
        stat_lev, p_lev = stats.levene(*groups)
        results['homoscedasticity'] = {
            'test': 'Levene', 'statistic': stat_lev, 'p_value': p_lev,
            'assumption_met': p_lev > 0.05,
            'interpretation': f'Las varianzas {"SÍ" if p_lev > 0.05 else "NO"} son homogéneas (p={p_lev:.4f})'
        }
    else:
        results['homoscedasticity'] = {
            'test': 'Levene', 'assumption_met': None,
            'interpretation': 'No se pudo realizar'
        }
    
    results['independence'] = {
        'test': 'Visual',
        'interpretation': 'Verificar orden de recolección'
    }
    
    return results

def tukey_hsd_bifactorial(df, response_var, factor_a, factor_b, alpha=0.05):
    """Prueba de Tukey HSD - CORREGIDO"""
    df_temp = df.copy()
    df_temp['_treatment_'] = df_temp[factor_a].astype(str) + ' × ' + df_temp[factor_b].astype(str)
    
    # CORRECCIÓN: groupby sin as_index=False, luego reset_index
    treatment_means = df_temp.groupby('_treatment_')[response_var].agg(['mean', 'count', 'std'])
    treatment_means = treatment_means.reset_index()
    treatment_means.columns = ['Tratamiento', 'Media', 'n', 'Std']
    
    treatments = treatment_means['Tratamiento'].tolist()
    k = len(treatments)
    
    anova_results = calculate_anova_bifactorial(df, response_var, factor_a, factor_b)
    if anova_results is None:
        return None
    
    MSE = anova_results['MSE']
    n_harmonic = k / sum(1/treatment_means['n'])
    q_values = {2: 2.92, 3: 3.53, 4: 3.93, 5: 4.21, 6: 4.43, 7: 4.60, 8: 4.74, 9: 4.86, 10: 4.96, 12: 5.13, 15: 5.31}
    q = q_values.get(k, 3.53 + (k-3)*0.15)
    HSD = q * np.sqrt(MSE / n_harmonic) if n_harmonic > 0 and MSE > 0 else 0
    
    comparisons = []
    for i in range(len(treatments)):
        for j in range(i+1, len(treatments)):
            t1, t2 = treatments[i], treatments[j]
            mean1 = treatment_means.loc[treatment_means['Tratamiento'] == t1, 'Media'].values[0]
            mean2 = treatment_means.loc[treatment_means['Tratamiento'] == t2, 'Media'].values[0]
            diff = abs(mean1 - mean2)
            comparisons.append({
                'Comparación': f'{t1} vs {t2}',
                'Media 1': round(mean1, 3),
                'Media 2': round(mean2, 3),
                'Diferencia': round(diff, 3),
                'HSD': round(HSD, 3),
                'Significativo': '✓ Sí' if diff > HSD else '✗ No',
                'p-ajustado': '< 0.05' if diff > HSD else '≥ 0.05'
            })
    
    return {
        'HSD': HSD, 'q_value': q, 'k': k, 'MSE': MSE,
        'n_harmonic': n_harmonic,
        'comparisons': pd.DataFrame(comparisons),
        'treatment_means': treatment_means
    }


def best_treatment_with_ci(df, response_var, factor_a, factor_b, confidence=0.95):
    """Identifica el mejor tratamiento con IC - CORREGIDO"""
    df_temp = df.copy()
    df_temp['_treatment_'] = df_temp[factor_a].astype(str) + ' × ' + df_temp[factor_b].astype(str)
    
    # CORRECCIÓN: groupby sin as_index=False, luego reset_index
    treatment_stats = df_temp.groupby('_treatment_')[response_var].agg(['mean', 'std', 'count'])
    treatment_stats = treatment_stats.reset_index()
    treatment_stats.columns = ['Tratamiento', 'Media', 'Std', 'n']
    
    best_idx = treatment_stats['Media'].idxmax()
    best_treatment = treatment_stats.loc[best_idx, 'Tratamiento']
    mean = treatment_stats.loc[best_idx, 'Media']
    std = treatment_stats.loc[best_idx, 'Std']
    n = treatment_stats.loc[best_idx, 'n']
    
    se = std / np.sqrt(n)
    t_crit = t.ppf((1 + confidence) / 2, n - 1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    
    factor_a_val = df_temp[df_temp['_treatment_'] == best_treatment][factor_a].iloc[0]
    factor_b_val = df_temp[df_temp['_treatment_'] == best_treatment][factor_b].iloc[0]
    
    return {
        'treatment': best_treatment, 'factor_a_value': factor_a_val, 'factor_b_value': factor_b_val,
        'mean': mean, 'std': std, 'n': n, 'se': se,
        'ci_lower': ci_lower, 'ci_upper': ci_upper, 'confidence': confidence,
        'cv_individual': (std / mean * 100) if mean > 0 else 0
    }

# ========================================================================================
# SIDEBAR
# ========================================================================================

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("---")
    selected_model = st.selectbox("📊 Modelo DCA", options=list(MODELS.keys()))
    
    with st.expander(f"ℹ️ Detalles del {selected_model}", expanded=True):
        st.markdown(f"""
        **{MODELS[selected_model]['name']}**
        
        📐 **Parámetros:** {MODELS[selected_model]['description']}
        
        📊 **Configuración:** {MODELS[selected_model]['parametros']}
        
        📝 **Estructura:** {MODELS[selected_model]['estructura']}
        """)
    
    st.markdown("---")
    st.header("📁 Datos")
    data_source = st.radio("Fuente:", ["Datos de Ejemplo", "Subir Archivo", "URL"])
    
    df = None
    
    if data_source == "Datos de Ejemplo":
        model_num = int(selected_model.split()[1])
        df = generate_example_data(model_num)
        st.success(f"✅ {len(df)} observaciones")
        st.info(f"📊 {df.groupby(['FactorA', 'FactorB']).size().shape[0]} tratamientos")
    
    elif data_source == "Subir Archivo":
        uploaded_file = st.file_uploader("CSV o Excel", type=['csv', 'xlsx', 'xls'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"✅ {len(df)} filas")
            except Exception as e:
                st.error(f"❌ {e}")
    
    else:
        url_option = st.selectbox("Dataset:", ["Personalizado"] + list(EXAMPLE_DATASETS.keys()))
        if url_option == "Personalizado":
            custom_url = st.text_input("URL CSV:")
            if custom_url and st.button("🔗 Cargar"):
                try:
                    response = requests.get(custom_url, timeout=10)
                    df = pd.read_csv(StringIO(response.text))
                    st.success(f"✅ {len(df)} filas")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            if st.button(f"📥 Cargar {url_option}"):
                try:
                    response = requests.get(EXAMPLE_DATASETS[url_option], timeout=10)
                    df = pd.read_csv(StringIO(response.text))
                    st.success(f"✅ {len(df)} filas")
                except Exception as e:
                    st.error(f"❌ {e}")
    
    if df is not None:
        st.markdown("---")
        st.header("🔧 Variables")
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        factor_a_col = st.selectbox("Factor A:", columns,
            index=columns.index('FactorA') if 'FactorA' in columns else 0)
        factor_b_col = st.selectbox("Factor B:", columns,
            index=columns.index('FactorB') if 'FactorB' in columns else min(1, len(columns)-1))
        response_col = st.selectbox("Respuesta:", numeric_cols if numeric_cols else columns,
            index=numeric_cols.index('Rendimiento') if 'Rendimiento' in numeric_cols else 0)
        alpha = st.slider("Nivel α:", 0.01, 0.10, 0.05, 0.01)
        
        st.markdown("---")
        st.metric("📊 Observaciones", len(df))
        st.metric("🔬 Niveles A", df[factor_a_col].nunique())
        st.metric("🧫 Niveles B", df[factor_b_col].nunique())

# ========================================================================================
# CONTENIDO PRINCIPAL
# ========================================================================================

if df is not None and 'factor_a_col' in locals():
    
    # ==================== CONTEXTO CORREGIDO ====================
    st.markdown('<div class="section-title">📋 1. CONTEXTO DEL EXPERIMENTO</div>', unsafe_allow_html=True)
    
    n_combinations = df.groupby([factor_a_col, factor_b_col]).size().shape[0]
    n_a = df[factor_a_col].nunique()
    n_b = df[factor_b_col].nunique()
    
    st.markdown(f"""
    <div class="section-box">
        <h3>🎯 Objetivo Principal</h3>
        <p><strong>Determinar si los factores {factor_a_col} y {factor_b_col}, así como su interacción, 
        tienen un efecto significativo sobre {response_col}</strong></p>
        
        <h4 style="margin-top: 1.5rem;">📊 Estructura del Diseño:</h4>
        <ul>
            <li><strong>Factor A ({factor_a_col}):</strong> {n_a} niveles → {', '.join(map(str, sorted(df[factor_a_col].unique())))}</li>
            <li><strong>Factor B ({factor_b_col}):</strong> {n_b} niveles → {', '.join(map(str, sorted(df[factor_b_col].unique())))}</li>
            <li><strong>Combinaciones A×B:</strong> {n_combinations} tratamientos experimentales</li>
            <li><strong>Total observaciones:</strong> {len(df)}</li>
        </ul>
        
        <h4 style="margin-top: 1.5rem;">🔬 Características del {MODELS[selected_model]['name']}:</h4>
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <p style="margin: 0.25rem 0;"><strong>📐 Tipo:</strong> {MODELS[selected_model]['name']}</p>
            <p style="margin: 0.25rem 0;"><strong>📊 Configuración:</strong> {MODELS[selected_model]['description']}</p>
            <p style="margin: 0.25rem 0;"><strong>📏 Parámetros:</strong> {MODELS[selected_model]['parametros']}</p>
            <p style="margin: 0.25rem 0;"><strong>📈 Estructura:</strong> {MODELS[selected_model]['estructura']}</p>
            <p style="margin: 0.25rem 0;"><strong>🔢 Promedio rep/tratamiento:</strong> {len(df) / n_combinations:.1f}</p>
        </div>
        
        <h4 style="margin-top: 1.5rem;">❓ Hipótesis a Probar:</h4>
        <ol>
            <li><strong>H₀_A:</strong> El Factor A ({factor_a_col}) NO tiene efecto significativo sobre {response_col}</li>
            <li><strong>H₀_B:</strong> El Factor B ({factor_b_col}) NO tiene efecto significativo sobre {response_col}</li>
            <li><strong>H₀_AB:</strong> NO existe interacción significativa entre los factores A y B</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== TEORÍA CON LATEX ====================
    st.markdown('<div class="section-title">📚 2. MODELO MATEMÁTICO</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="formula-box"><h3 style="color: #74b9ff; margin-top: 0;">📐 Modelo Estadístico:</h3></div>', unsafe_allow_html=True)
    st.latex(r"y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}")
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <strong>Donde:</strong><br>
        • <code>y<sub>ijk</sub></code> = Observación k del nivel i de A y nivel j de B<br>
        • <code>μ</code> = Media general poblacional<br>
        • <code>α<sub>i</sub></code> = Efecto del nivel i del factor A<br>
        • <code>β<sub>j</sub></code> = Efecto del nivel j del factor B<br>
        • <code>(αβ)<sub>ij</sub></code> = Efecto de interacción<br>
        • <code>ε<sub>ijk</sub></code> = Error aleatorio ~ N(0, σ²)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="formula-box"><h3 style="color: #74b9ff; margin: 1.5rem 0 1rem 0;">🧮 Descomposición de Sumas de Cuadrados:</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Factor de Corrección:**")
        st.latex(r"CF = \frac{(\sum y)^2}{N}")
        
        st.markdown("**2. SC Total:**")
        st.latex(r"SC_T = \sum y^2 - CF")
        
        st.markdown("**3. SC Factor A:**")
        st.latex(r"SC_A = \frac{\sum y_{i..}^2}{bn} - CF")
    
    with col2:
        st.markdown("**4. SC Factor B:**")
        st.latex(r"SC_B = \frac{\sum y_{.j.}^2}{an} - CF")
        
        st.markdown("**5. SC Interacción:**")
        st.latex(r"SC_{AB} = \frac{\sum y_{ij.}^2}{n} - CF - SC_A - SC_B")
        
        st.markdown("**6. SC Error:**")
        st.latex(r"SC_E = SC_T - SC_A - SC_B - SC_{AB}")
    
    st.markdown('<div class="formula-box"><h3 style="color: #74b9ff; margin: 1.5rem 0 1rem 0;">📊 Estadísticos F:</h3></div>', unsafe_allow_html=True)
    st.latex(r"F_A = \frac{CM_A}{CM_E} \quad F_B = \frac{CM_B}{CM_E} \quad F_{AB} = \frac{CM_{AB}}{CM_E}")
    
    # ==================== EXPLORACIÓN ====================
    st.markdown('<div class="section-title">📊 3. EXPLORACIÓN DE DATOS</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 Observaciones", len(df))
    with col2:
        st.metric(f"🔬 Niveles A", n_a)
    with col3:
        st.metric(f"🧫 Niveles B", n_b)
    with col4:
        st.metric("🎯 Tratamientos", n_combinations)
    with col5:
        st.metric("📊 Media", f"{df[response_col].mean():.2f}")
    
    with st.expander("📋 Ver Datos Completos"):
        st.dataframe(df, use_container_width=True, height=400)
    
    st.subheader(f"📊 Estadísticas por {factor_a_col}")
    st.dataframe(df.groupby(factor_a_col)[response_col].describe().round(3), use_container_width=True)
    
    st.subheader(f"📊 Estadísticas por {factor_b_col}")
    st.dataframe(df.groupby(factor_b_col)[response_col].describe().round(3), use_container_width=True)
    
    st.subheader(f"📊 Estadísticas por {factor_a_col} × {factor_b_col}")
    st.dataframe(df.groupby([factor_a_col, factor_b_col])[response_col].describe().round(3), use_container_width=True)
    
    # ==================== ANOVA ====================
    st.markdown('<div class="section-title">📈 4. ANÁLISIS DE VARIANZA</div>', unsafe_allow_html=True)
    
    anova_results = calculate_anova_bifactorial(df, response_col, factor_a_col, factor_b_col)
    
    if anova_results is not None:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Media", f"{anova_results['grand_mean']:.2f}")
        with col2:
            st.metric("F_A", f"{anova_results['F_A']:.2f}")
        with col3:
            st.metric("F_B", f"{anova_results['F_B']:.2f}")
        with col4:
            st.metric("F_AB", f"{anova_results['F_AB']:.2f}")
        with col5:
            st.metric("CV%", f"{anova_results['CV']:.2f}%")
        with col6:
            st.metric("R²", f"{anova_results['R_squared']:.4f}")
        
        st.subheader("📋 Tabla ANOVA")
        anova_display = anova_results['anova_table'].copy()
        for col in ['SC', 'CM', 'F_calc', 'F_crit', 'p-valor']:
            anova_display[col] = anova_display[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) and x != '' else "-")
        st.dataframe(anova_display, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
            <strong>Leyenda:</strong>
            <code>***</code> p<0.001 | <code>**</code> p<0.01 | <code>*</code> p<0.05 | <code>ns</code> No significativo
        </div>
        """, unsafe_allow_html=True)
        
        # CÁLCULOS DETALLADOS
        st.subheader("🧮 Cálculos Detallados")
        
        st.markdown('<div class="formula-box"><h4 style="color: #74b9ff;">Paso 1: Factor de Corrección</h4></div>', unsafe_allow_html=True)
        st.latex(r"CF = \frac{(" + f"{df[response_col].sum():.2f}" + r")^2}{" + f"{anova_results['n_total']}" + r"} = " + f"{anova_results['CF']:.4f}")
        
        st.markdown('<div class="formula-box"><h4 style="color: #74b9ff;">Paso 2: Sumas de Cuadrados</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"SC_T = " + f"{anova_results['SST']:.4f}")
            st.latex(r"SC_A = " + f"{anova_results['SSA']:.4f}")
            st.latex(r"SC_B = " + f"{anova_results['SSB']:.4f}")
        with col2:
            st.latex(r"SC_{AB} = " + f"{anova_results['SSAB']:.4f}")
            st.latex(r"SC_E = " + f"{anova_results['SSE']:.4f}")
            st.markdown(f"**Verificación:** {anova_results['SST']:.4f} = {anova_results['SSA']:.4f} + {anova_results['SSB']:.4f} + {anova_results['SSAB']:.4f} + {anova_results['SSE']:.4f}")
        
        st.markdown('<div class="formula-box"><h4 style="color: #74b9ff;">Paso 3: Estadísticos F</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.latex(r"F_A = " + f"{anova_results['F_A']:.4f}")
            st.markdown(f"p-valor = {anova_results['p_A']:.6f}")
        with col2:
            st.latex(r"F_B = " + f"{anova_results['F_B']:.4f}")
            st.markdown(f"p-valor = {anova_results['p_B']:.6f}")
        with col3:
            st.latex(r"F_{AB} = " + f"{anova_results['F_AB']:.4f}")
            st.markdown(f"p-valor = {anova_results['p_AB']:.6f}")
        
        st.markdown('<div class="formula-box"><h4 style="color: #74b9ff;">Paso 4: Calidad del Modelo</h4></div>', unsafe_allow_html=True)
        
        st.latex(r"CV = \frac{\sqrt{CM_E}}{\bar{y}} \times 100 = " + f"{anova_results['CV']:.2f}\\%")
        st.latex(r"R^2 = \frac{SC_A + SC_B + SC_{AB}}{SC_T} = " + f"{anova_results['R_squared']:.4f}")
        
        st.markdown(f"""
        <div style='background: #e8f5e9; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
            <strong>CV:</strong> {anova_results['CV']:.2f}% - 
            {"✓ Excelente" if anova_results['CV'] < 10 else "✓ Muy Bueno" if anova_results['CV'] < 20 else "✓ Aceptable" if anova_results['CV'] < 30 else "⚠ Regular"}
            <br><strong>Modelo:</strong> {anova_results['model_effectiveness']} (R² = {anova_results['R_squared']*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
        
        # ==================== SUPUESTOS ====================
        st.markdown('<div class="section-title">✅ 5. VALIDACIÓN DE SUPUESTOS</div>', unsafe_allow_html=True)
        
        assumptions = check_anova_assumptions(df, response_col, factor_a_col, factor_b_col)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 1️⃣ Normalidad")
            norm = assumptions['normality']
            if 'p_value' in norm:
                st.metric("p-valor", f"{norm['p_value']:.4f}")
                if norm['assumption_met']:
                    st.success("✅ " + norm['interpretation'])
                else:
                    st.error("❌ " + norm['interpretation'])
        
        with col2:
            st.markdown("### 2️⃣ Homocedasticidad")
            if 'homoscedasticity' in assumptions and 'p_value' in assumptions['homoscedasticity']:
                homo = assumptions['homoscedasticity']
                st.metric("p-valor", f"{homo['p_value']:.4f}")
                if homo['assumption_met']:
                    st.success("✅ " + homo['interpretation'])
                else:
                    st.error("❌ " + homo['interpretation'])
        
        with col3:
            st.markdown("### 3️⃣ Independencia")
            st.info(assumptions['independence']['interpretation'])
        
        # ==================== INTERPRETACIONES ====================
        st.markdown('<div class="section-title">💡 6. INTERPRETACIONES</div>', unsafe_allow_html=True)
        
        st.subheader(f"🔬 Factor A: {factor_a_col}")
        if anova_results['significant_A']:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ EFECTO SIGNIFICATIVO</h4>
                <p><strong>Decisión:</strong> Se rechaza H₀_A (α = {alpha})</p>
                <p>F = {anova_results['F_A']:.4f}, p = {anova_results['p_A']:.6f} < {alpha}</p>
                <p><strong>Conclusión:</strong> {factor_a_col} afecta significativamente a {response_col}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ NO HAY EFECTO SIGNIFICATIVO</h4>
                <p>F = {anova_results['F_A']:.4f}, p = {anova_results['p_A']:.6f} ≥ {alpha}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader(f"🧫 Factor B: {factor_b_col}")
        if anova_results['significant_B']:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ EFECTO SIGNIFICATIVO</h4>
                <p>F = {anova_results['F_B']:.4f}, p = {anova_results['p_B']:.6f} < {alpha}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ NO HAY EFECTO SIGNIFICATIVO</h4>
                <p>F = {anova_results['F_B']:.4f}, p = {anova_results['p_B']:.6f} ≥ {alpha}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("🔄 Interacción A×B")
        if anova_results['significant_AB']:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ INTERACCIÓN SIGNIFICATIVA</h4>
                <p>F = {anova_results['F_AB']:.4f}, p = {anova_results['p_AB']:.6f} < {alpha}</p>
                <p><strong>Implicación:</strong> El efecto de un factor DEPENDE del otro</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ Sin interacción (F = {anova_results['F_AB']:.4f}, p = {anova_results['p_AB']:.6f})")
        
        # ==================== VISUALIZACIONES CORREGIDAS ====================
        st.markdown('<div class="section-title">📉 7. VISUALIZACIONES</div>', unsafe_allow_html=True)
        
        st.subheader("📈 Gráfico de Interacción A × B")
        
        st.markdown("""
        <div class="assumptions-box">
            <strong>🎯 Interpretación:</strong>
            <ul>
                <li><strong>Líneas paralelas:</strong> NO hay interacción</li>
                <li><strong>Líneas que se cruzan:</strong> SÍ hay interacción</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # CORREGIDO: evitar conflictos con reset_index
        df_int = df[[factor_a_col, factor_b_col, response_col]].copy()
        interaction_data = df_int.groupby([factor_a_col, factor_b_col], as_index=False)[response_col].mean()
        
        fig_int = go.Figure()
        
        for b_level in sorted(df[factor_b_col].unique()):
            subset = interaction_data[interaction_data[factor_b_col] == b_level]
            fig_int.add_trace(go.Scatter(
                x=subset[factor_a_col],
                y=subset[response_col],
                mode='lines+markers',
                name=f'{factor_b_col}: {b_level}',
                line=dict(width=3),
                marker=dict(size=12, symbol='diamond')
            ))
        
        fig_int.update_layout(
            title=f'Interacción {factor_a_col} × {factor_b_col}',
            xaxis_title=factor_a_col,
            yaxis_title=f'Media de {response_col}',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_int, use_container_width=True)
        
        # Heatmap CORREGIDO
        st.subheader("🔥 Mapa de Calor")
        
        df_piv = df[[factor_a_col, factor_b_col, response_col]].copy()
        pivot_table = df_piv.pivot_table(
            values=response_col,
            index=factor_b_col,
            columns=factor_a_col,
            aggfunc='mean'
        )
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlGn',
            text=np.round(pivot_table.values, 2),
            texttemplate='%{text}',
            textfont={"size": 14, "color": "black"},
            colorbar=dict(title=response_col)
        ))
        
        fig_heat.update_layout(
            title=f'{response_col} por {factor_a_col} × {factor_b_col}',
            xaxis_title=factor_a_col,
            yaxis_title=factor_b_col,
            height=450,
            template='plotly_white'
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.subheader("📊 Efectos Principales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_a = df[[factor_a_col, response_col]].copy()
            main_a = df_a.groupby(factor_a_col)[response_col].agg(['mean', 'std', 'count'])
            main_a = main_a.reset_index()
            main_a['se'] = main_a['std'] / np.sqrt(main_a['count'])
            
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(
                x=main_a[factor_a_col],
                y=main_a['mean'],
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                marker=dict(size=12),
                error_y=dict(type='data', array=main_a['se'], visible=True)
            ))
            
            fig_a.update_layout(
                title=f'Efecto Principal: {factor_a_col}',
                xaxis_title=factor_a_col,
                yaxis_title=f'Media de {response_col}',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_a, use_container_width=True)
            
            if anova_results['significant_A']:
                st.success("✓ Efecto significativo")
            else:
                st.info("○ Sin efecto significativo")
        
        with col2:
            df_b = df[[factor_b_col, response_col]].copy()
            main_b = df_b.groupby(factor_b_col)[response_col].agg(['mean', 'std', 'count'])
            
            main_b = main_b.reset_index()
            main_b['se'] = main_b['std'] / np.sqrt(main_b['count'])
            
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(
                x=main_b[factor_b_col],
                y=main_b['mean'],
                mode='lines+markers',
                line=dict(color='#764ba2', width=3),
                marker=dict(size=12),
                error_y=dict(type='data', array=main_b['se'], visible=True)
            ))
            
            fig_b.update_layout(
                title=f'Efecto Principal: {factor_b_col}',
                xaxis_title=factor_b_col,
                yaxis_title=f'Media de {response_col}',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_b, use_container_width=True)
            
            if anova_results['significant_B']:
                st.success("✓ Efecto significativo")
            else:
                st.info("○ Sin efecto significativo")
        
        # Box Plots CORREGIDO
        st.subheader("📦 Box Plots por Tratamiento")
        
        df_box = df.copy()
        df_box['_trat_box_'] = df_box[factor_a_col].astype(str) + ' × ' + df_box[factor_b_col].astype(str)
        
        fig_box = px.box(
            df_box,
            x='_trat_box_',
            y=response_col,
            color=factor_b_col,
            title=f'Distribución de {response_col} por Tratamiento'
        )
        
        fig_box.update_layout(height=500, template='plotly_white', xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # ==================== MEJOR TRATAMIENTO ====================
        st.markdown('<div class="section-title">🏆 8. MEJOR TRATAMIENTO</div>', unsafe_allow_html=True)
        
        best = best_treatment_with_ci(df, response_col, factor_a_col, factor_b_col, confidence=0.95)
        
        # Ranking CORREGIDO
        df_rank = df.copy()
        df_rank['_trat_rank_'] = df_rank[factor_a_col].astype(str) + ' × ' + df_rank[factor_b_col].astype(str)
        ranking = df_rank.groupby('_trat_rank_')[response_col].agg(['mean', 'std', 'count'])
        ranking = ranking.reset_index()
        ranking.columns = ['Tratamiento', 'Media', 'Std', 'n']
        ranking = ranking.sort_values('Media', ascending=False).reset_index(drop=True)
        ranking['Ranking'] = range(1, len(ranking) + 1)
        ranking['CV%'] = (ranking['Std'] / ranking['Media'] * 100).round(2)
        
        second_best_mean = ranking.iloc[1]['Media'] if len(ranking) > 1 else 0
        difference_percent = ((best['mean'] - second_best_mean) / second_best_mean * 100) if second_best_mean > 0 else 0
        
        st.markdown(f"""
        <div class="best-treatment-box">
            <h2 style="margin: 0; text-align: center;">🥇 MEJOR TRATAMIENTO</h2>
            <h1 style="text-align: center; margin: 1rem 0; font-size: 2.5rem;">{best['treatment']}</h1>
            
            <div style="text-align: center; margin: 1.5rem 0;">
                <h3 style="margin: 0;">Media de {response_col}</h3>
                <h2 style="margin: 0.5rem 0; font-size: 3rem;">{best['mean']:.2f}</h2>
                <p style="margin: 0; font-size: 1.2rem;">IC 95%: [{best['ci_lower']:.2f}, {best['ci_upper']:.2f}]</p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 2rem;">
                <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.9rem;">Observaciones</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{best['n']}</p>
                </div>
                <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.9rem;">Desv. Estándar</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{best['std']:.2f}</p>
                </div>
                <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.9rem;">CV Individual</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{best['cv_individual']:.2f}%</p>
                </div>
                <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.9rem;">Superioridad</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{difference_percent:.1f}%</p>
                </div>
            </div>
            
            <div style="margin-top: 2rem; background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 8px;">
                <h4 style="margin: 0 0 1rem 0;">🎯 Condiciones Óptimas:</h4>
                <p style="margin: 0.5rem 0; font-size: 1.2rem;"><strong>{factor_a_col}:</strong> {best['factor_a_value']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.2rem;"><strong>{factor_b_col}:</strong> {best['factor_b_value']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Fórmula IC
        st.markdown('<div class="formula-box"><h4 style="color: #74b9ff;">Intervalo de Confianza (95%):</h4></div>', unsafe_allow_html=True)
        st.latex(r"IC_{95\%} = \bar{y} \pm t_{\alpha/2,n-1} \times \frac{s}{\sqrt{n}}")
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>Cálculo:</strong><br>
            • Media (ȳ) = {best['mean']:.4f}<br>
            • SE = s/√n = {best['std']:.4f}/√{best['n']} = {best['se']:.4f}<br>
            • t<sub>crítico</sub> = {t.ppf((1 + 0.95) / 2, best['n'] - 1):.4f}<br>
            • <strong>IC = [{best['ci_lower']:.4f}, {best['ci_upper']:.4f}]</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Ranking
        st.subheader("📊 Ranking Completo")
        
        ranking_display = ranking.copy()
        ranking_display['Media'] = ranking_display['Media'].round(3)
        ranking_display['Std'] = ranking_display['Std'].round(3)
        
        st.dataframe(
            ranking_display[['Ranking', 'Tratamiento', 'Media', 'Std', 'CV%', 'n']],
            use_container_width=True,
            hide_index=True
        )
        
        # Análisis detallado
        st.subheader("📌 Análisis Detallado")
        
        cv_interpretation = "excelente" if best['cv_individual'] < 10 else "buena" if best['cv_individual'] < 20 else "aceptable" if best['cv_individual'] < 30 else "regular"
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; border-left: 4px solid #11998e;">
            <h4 style="color: #11998e; margin-top: 0;">Justificación Estadística</h4>
            
            <h5>1. Superioridad:</h5>
            <ul>
                <li><strong>Media:</strong> {best['mean']:.2f} unidades</li>
                <li><strong>Ventaja:</strong> {difference_percent:.1f}% sobre 2° lugar</li>
                <li><strong>IC 95%:</strong> [{best['ci_lower']:.2f}, {best['ci_upper']:.2f}]</li>
                <li><strong>Error estándar:</strong> ±{best['se']:.3f}</li>
            </ul>
            
            <h5>2. Confiabilidad:</h5>
            <ul>
                <li><strong>CV:</strong> {best['cv_individual']:.2f}% - Precisión {cv_interpretation}</li>
                <li><strong>n:</strong> {best['n']} observaciones</li>
                <li><strong>Desv. Std:</strong> ±{best['std']:.2f}</li>
            </ul>
            
            <h5>3. Condiciones Óptimas:</h5>
            <ul>
                <li><strong>{factor_a_col} = {best['factor_a_value']}</strong></li>
                <li><strong>{factor_b_col} = {best['factor_b_value']}</strong></li>
            </ul>
            
            <h5>4. Aplicación Práctica:</h5>
            <ul>
                <li>✓ Optimización de procesos</li>
                <li>✓ Estandarización (SOP)</li>
                <li>✓ Control de calidad</li>
                <li>✓ Escalamiento industrial</li>
            </ul>
            
            <h5>5. Recomendaciones:</h5>
            <ol>
                <li>Realizar experimentos confirmatorios (n ≥ 5)</li>
                <li>Pruebas piloto antes de producción</li>
                <li>Implementar gráficos de control</li>
                <li>Documentar SOP con estas condiciones</li>
                <li>Análisis de costo-beneficio</li>
                <li>Estudiar robustez del proceso</li>
            </ol>
            
            <h5>6. Sobre la Interacción:</h5>
            <p>{"⚠️ <strong>CRÍTICO:</strong> Interacción significativa detectada. Mantener AMBOS factores en niveles óptimos simultáneamente." if anova_results['significant_AB'] else "✓ Sin interacción significativa. Los factores pueden optimizarse independientemente."}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ==================== RESUMEN EJECUTIVO ====================
        st.markdown('<div class="section-title">📝 9. RESUMEN EJECUTIVO</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="section-box" style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);">
            <h3 style="color: #667eea;">📊 Resumen del Análisis</h3>
            
            <h4>🔬 Diseño:</h4>
            <ul>
                <li><strong>Modelo:</strong> {MODELS[selected_model]['name']}</li>
                <li><strong>Configuración:</strong> {MODELS[selected_model]['parametros']}</li>
                <li><strong>Observaciones:</strong> {anova_results['n_total']}</li>
                <li><strong>Factores:</strong> {factor_a_col} ({anova_results['n_a']} niveles) × {factor_b_col} ({anova_results['n_b']} niveles)</li>
            </ul>
            
            <h4>📈 Resultados ANOVA:</h4>
            <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
                <thead>
                    <tr style="background: #667eea; color: white;">
                        <th style="padding: 0.75rem;">Efecto</th>
                        <th style="padding: 0.75rem; text-align: center;">F</th>
                        <th style="padding: 0.75rem; text-align: center;">p-valor</th>
                        <th style="padding: 0.75rem; text-align: center;">Resultado</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 0.75rem;"><strong>Factor A</strong></td>
                        <td style="padding: 0.75rem; text-align: center;">{anova_results['F_A']:.4f}</td>
                        <td style="padding: 0.75rem; text-align: center;">{anova_results['p_A']:.6f}</td>
                        <td style="padding: 0.75rem; text-align: center;">{"✅ Sig" if anova_results['significant_A'] else "❌ NS"}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.75rem;"><strong>Factor B</strong></td>
                        <td style="padding: 0.75rem; text-align: center;">{anova_results['F_B']:.4f}</td>
                        <td style="padding: 0.75rem; text-align: center;">{anova_results['p_B']:.6f}</td>
                        <td style="padding: 0.75rem; text-align: center;">{"✅ Sig" if anova_results['significant_B'] else "❌ NS"}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 0.75rem;"><strong>Interacción A×B</strong></td>
                        <td style="padding: 0.75rem; text-align: center;">{anova_results['F_AB']:.4f}</td>
                        <td style="padding: 0.75rem; text-align: center;">{anova_results['p_AB']:.6f}</td>
                        <td style="padding: 0.75rem; text-align: center;">{"⚠️ Sig" if anova_results['significant_AB'] else "✅ NS"}</td>
                    </tr>
                </tbody>
            </table>
            
            <h4>🎯 Calidad:</h4>
            <ul>
                <li><strong>CV:</strong> {anova_results['CV']:.2f}% - {"✓ Excelente" if anova_results['CV'] < 10 else "✓ Muy Bueno" if anova_results['CV'] < 20 else "✓ Aceptable" if anova_results['CV'] < 30 else "⚠ Regular"}</li>
                <li><strong>R²:</strong> {anova_results['R_squared']:.4f} ({anova_results['R_squared']*100:.1f}%)</li>
                <li><strong>Modelo:</strong> {anova_results['model_effectiveness']}</li>
            </ul>
            
            <h4>🏆 Tratamiento Óptimo:</h4>
            <ul>
                <li><strong>Combinación:</strong> {best['treatment']}</li>
                <li><strong>Condiciones:</strong> {factor_a_col} = {best['factor_a_value']}, {factor_b_col} = {best['factor_b_value']}</li>
                <li><strong>Media:</strong> {best['mean']:.2f} ± {best['se']:.3f}</li>
                <li><strong>IC 95%:</strong> [{best['ci_lower']:.2f}, {best['ci_upper']:.2f}]</li>
                <li><strong>Superioridad:</strong> {difference_percent:.1f}% sobre 2° lugar</li>
            </ul>
            
            <h4>💡 Conclusiones:</h4>
            <ol>
                <li>{"✅ Factor A SÍ tiene efecto significativo (p=" + f"{anova_results['p_A']:.6f}" + ")" if anova_results['significant_A'] else "○ Factor A NO tiene efecto significativo"}</li>
                <li>{"✅ Factor B SÍ tiene efecto significativo (p=" + f"{anova_results['p_B']:.6f}" + ")" if anova_results['significant_B'] else "○ Factor B NO tiene efecto significativo"}</li>
                <li>{"⚠️ EXISTE interacción significativa - Efectos dependientes" if anova_results['significant_AB'] else "✅ Sin interacción - Efectos independientes"}</li>
                <li>El modelo explica el {anova_results['R_squared']*100:.1f}% de la variabilidad</li>
                <li>Tratamiento <strong>{best['treatment']}</strong> maximiza {response_col}</li>
            </ol>
            
            <h4>📋 Recomendaciones:</h4>
            <ul>
                <li>✅ Implementar condiciones óptimas identificadas</li>
                <li>✅ Experimentos confirmatorios (≥ 5)</li>
                <li>✅ Establecer SOP con especificaciones</li>
                <li>✅ Gráficos de control para monitoreo</li>
                <li>✅ Documentar rango de tolerancia</li>
                <li>✅ Capacitar personal en condiciones óptimas</li>
                {"<li>⚠️ Mantener AMBOS factores en niveles óptimos (interacción)</li>" if anova_results['significant_AB'] else "<li>✓ Factores pueden controlarse independientemente</li>"}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # ==================== EXPORTAR ====================
        st.markdown('<div class="section-title">📥 10. EXPORTAR RESULTADOS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Descargar Datos (CSV)",
                data=csv,
                file_name=f"datos_DCA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Datos', index=False)
                anova_results['anova_table'].to_excel(writer, sheet_name='ANOVA', index=False)
                anova_results['a_means'].to_frame(name='Media').to_excel(writer, sheet_name=f'Medias {factor_a_col}')
                anova_results['b_means'].to_frame(name='Media').to_excel(writer, sheet_name=f'Medias {factor_b_col}')
                anova_results['ab_means'].to_frame(name='Media').to_excel(writer, sheet_name='Medias AB')
                
                best_df = pd.DataFrame([{
                    'Tratamiento': best['treatment'],
                    factor_a_col: best['factor_a_value'],
                    factor_b_col: best['factor_b_value'],
                    'Media': best['mean'],
                    'Std': best['std'],
                    'n': best['n'],
                    'IC_inf': best['ci_lower'],
                    'IC_sup': best['ci_upper'],
                    'CV%': best['cv_individual']
                }])
                best_df.to_excel(writer, sheet_name='Mejor', index=False)
                
                ranking.to_excel(writer, sheet_name='Ranking', index=False)
                
                summary = pd.DataFrame([{
                    'Analisis': 'ANOVA Bifactorial',
                    'Modelo': selected_model,
                    'Fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'N': anova_results['n_total'],
                    'Factor_A': factor_a_col,
                    'Niveles_A': anova_results['n_a'],
                    'Factor_B': factor_b_col,
                    'Niveles_B': anova_results['n_b'],
                    'Respuesta': response_col,
                    'Media_general': anova_results['grand_mean'],
                    'CV': anova_results['CV'],
                    'R2': anova_results['R_squared'],
                    'Modelo_efectividad': anova_results['model_effectiveness'],
                    'F_A': anova_results['F_A'],
                    'p_A': anova_results['p_A'],
                    'Sig_A': 'Sí' if anova_results['significant_A'] else 'No',
                    'F_B': anova_results['F_B'],
                    'p_B': anova_results['p_B'],
                    'Sig_B': 'Sí' if anova_results['significant_B'] else 'No',
                    'F_AB': anova_results['F_AB'],
                    'p_AB': anova_results['p_AB'],
                    'Interaccion': 'Sí' if anova_results['significant_AB'] else 'No',
                    'Mejor': best['treatment'],
                    'Media_mejor': best['mean'],
                    'Alpha': alpha
                }])
                summary.T.to_excel(writer, sheet_name='Resumen', header=False)
            
            excel_data = output.getvalue()
            st.download_button(
                label="📑 Descargar Reporte (Excel)",
                data=excel_data,
                file_name=f"reporte_ANOVA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.success("✅ Archivos listos. El Excel incluye todos los análisis completos.")

else:
    st.info("👈 Selecciona datos en el panel lateral")
    
    st.markdown("""
    <div class="assumptions-box">
        <h3>📖 Guía de Uso Rápida</h3>
        <ol>
            <li><strong>Selecciona un modelo</strong> (1-6)</li>
            <li><strong>Carga datos:</strong> Ejemplo, archivo o URL</li>
            <li><strong>Configura variables:</strong> Factor A, Factor B, Respuesta</li>
            <li><strong>Obtén:</strong>
                <ul>
                    <li>✓ ANOVA bifactorial completo</li>
                    <li>✓ Fórmulas matemáticas LaTeX</li>
                    <li>✓ Validación de supuestos</li>
                    <li>✓ Gráfico de interacción</li>
                    <li>✓ Mejor tratamiento con IC</li>
                    <li>✓ Visualizaciones interactivas</li>
                    <li>✓ Resumen ejecutivo</li>
                    <li>✓ Exportación Excel completa</li>
                </ul>
            </li>
        </ol>
        
        <h4 style="margin-top: 1.5rem;">📊 Formato de Datos:</h4>
        <p>Tu archivo debe tener al menos 3 columnas:</p>
        <ul>
            <li><strong>Factor A:</strong> Primera variable independiente</li>
            <li><strong>Factor B:</strong> Segunda variable independiente</li>
            <li><strong>Respuesta:</strong> Variable dependiente numérica</li>
        </ul>
        
        <h4 style="margin-top: 1.5rem;">💡 Ejemplo:</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: #667eea; color: white;">
                    <th style="padding: 0.5rem; border: 1px solid #ddd;">Temperatura</th>
                    <th style="padding: 0.5rem; border: 1px solid #ddd;">Concentracion</th>
                    <th style="padding: 0.5rem; border: 1px solid #ddd;">Rendimiento</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">25°C</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">0.1M</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">185.5</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">50°C</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">0.5M</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">245.2</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ========================================================================================
# FOOTER
# ========================================================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <h3 style='margin: 0; color: white;'>🧪 Sistema Completo DCA Bifactorial - ANOVA</h3>
    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        <strong>R. Andre Vilca Solorzano</strong> | Código: <strong>221181</strong><br>
        Docente: <strong>Ing. LLUEN VALLEJOS CESAR AUGUSTO</strong><br>
        Diseño Experimental | Universidad Nacional del Altiplano | {datetime.now().year}
    </p>
    
</div>
""", unsafe_allow_html=True)
                                  
