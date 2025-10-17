# 🧪 Sistema de Análisis DCA Bifactorial - ANOVA

Sistema profesional de análisis estadístico para Diseños Completamente al Azar (DCA) con dos factores, desarrollado con Streamlit.

## 👨‍🎓 Información del Estudiante

- **Nombre:** R. Andre Vilca Solorzano
- **Código:** 221181
- **Curso:** Diseño Experimental - Química Analítica

## 🚀 Características

### ✨ Modelos Soportados

1. **Modelo 1:** Balanceado con submuestreo balanceado
2. **Modelo 2:** No balanceado
3. **Modelo 3:** Balanceado con submuestreo balanceado (2 factores)
4. **Modelo 4:** Balanceado con submuestreo no balanceado
5. **Modelo 5:** No balanceado con submuestreo balanceado
6. **Modelo 6:** No balanceado con submuestreo no balanceado

### 📊 Análisis Estadístico

- ✅ **ANOVA Completo:** Tabla ANOVA con SC, CM, F, p-valor
- ✅ **Prueba de Tukey (HSD):** Comparaciones múltiples
- ✅ **Prueba de Duncan (MRT):** Grupos homogéneos
- ✅ **Estadísticas Descriptivas:** Medias, varianzas, CV
- ✅ **Interpretaciones Automáticas:** Decisiones estadísticas

### 📉 Visualizaciones Avanzadas

- 🌐 **Superficie 3D Interactiva:** Interacción de factores
- 🔥 **Heatmap Animado:** Matriz de rendimientos
- 📍 **Gráfico de Contorno:** Isocurvas de rendimiento
- 📦 **Box Plots:** Distribución por tratamiento
- 🎻 **Violin Plots:** Densidad de distribución
- 🎯 **Radar Chart:** Comparación multivariada
- 📈 **Gráficos Interactivos:** Zoom, pan, hover

### 📁 Fuentes de Datos

- 📊 Datos de ejemplo integrados (química: concentración × pH)
- 📤 Carga de archivos CSV y Excel
- 🔗 Importación desde URLs (Kaggle, GitHub, etc.)

## 🛠️ Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación en Windows

1. **Abrir PowerShell o CMD como Administrador**

2. **Navegar a la carpeta del proyecto:**
```bash
cd C:\Users\andre\Downloads\DASH
```

3. **Crear entorno virtual:**
```bash
python -m venv venv
```

4. **Activar el entorno virtual:**
```bash
# En PowerShell
.\venv\Scripts\Activate.ps1

# En CMD
.\venv\Scripts\activate.bat
```

5. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## ▶️ Ejecución

### Método 1: Desde el entorno virtual

```bash
# Activar entorno virtual (si no está activo)
.\venv\Scripts\Activate.ps1

# Ejecutar la aplicación
streamlit run app.py
```

### Método 2: Directamente

```bash
python -m streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Guía de Uso

### 1. Seleccionar Modelo

En el panel lateral, selecciona uno de los 6 modelos de diseño experimental disponibles.

### 2. Cargar Datos

Elige una de las tres opciones:

- **Datos de Ejemplo:** Datos generados automáticamente para el modelo seleccionado
- **Subir Archivo:** Carga tu propio CSV o Excel
- **Cargar desde URL:** Importa datos desde una URL pública

#### Formato de Datos Requerido

Tu archivo debe contener al menos estas columnas:

```
Tratamiento, Rendimiento
T1, 210.5
T1, 215.3
T2, 230.1
...
```

Para diseños bifactoriales, incluir:

```
Concentracion, pH, Rendimiento
0.1M, pH3, 210.5
0.1M, pH3, 215.3
0.5M, pH7, 230.1
...
```

### 3. Configurar Variables

En el panel lateral, selecciona:

- **Variable de Tratamiento:** Factor a analizar
- **Variable de Respuesta:** Variable dependiente
- **Nivel de significancia (α):** Por defecto 0.05

### 4. Explorar Resultados

Navega por las pestañas:

- **📊 Datos:** Vista previa y estadísticas descriptivas
- **📈 ANOVA:** Tabla ANOVA completa con interpretaciones
- **🔬 Comparaciones Múltiples:** Pruebas de Tukey y Duncan
- **📉 Visualizaciones Avanzadas:** Gráficos interactivos 3D
- **📝 Reporte:** Resumen ejecutivo y descarga de resultados

## 🔗 Datasets de Ejemplo

El sistema incluye links a datasets reales:

1. **Chemical Reactions:** https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
2. **Lab Measurements:** https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
3. **pH Studies:** https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv

## 🧮 Fórmulas Implementadas

### Modelo Estadístico

```
y_ij = μ + α_i + e_ij
```

### Descomposición de Varianza

```
SCT = SCA + SCE
```

### Estadístico F

```
F = MSA / MSE
```

### Prueba de Tukey (HSD)

```
HSD = q * √(MSE / n)
```

### Prueba de Duncan (LSR)

```
LSR = r * √(MSE / n)
```

## 📦 Estructura del Proyecto

```
DASH/
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias
├── README.md          # Este archivo
├── .gitignore         # Archivos ignorados por Git
├── venv/              # Entorno virtual (generado)
└── data/              # Datos de ejemplo (opcional)
    └── ejemplo.csv
```

## 🐛 Solución de Problemas

### Error: "streamlit command not found"

```bash
# Asegúrate de que el entorno virtual esté activado
.\venv\Scripts\Activate.ps1

# O instala streamlit globalmente
pip install streamlit
```

### Error: "ModuleNotFoundError"

```bash
# Reinstala las dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: "Permission denied"

```bash
# Ejecuta PowerShell como Administrador
# O cambia la política de ejecución
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Puerto 8501 ocupado

```bash
# Usa un puerto diferente
streamlit run app.py --server.port 8502
```

## 🎨 Personalización

### Cambiar tema de colores

En `app.py`, modifica la sección de CSS:

```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #tu-color-1, #tu-color-2);
    }
</style>
""", unsafe_allow_html=True)
```

### Agregar nuevos modelos

En la sección `MODELS` de `app.py`, agrega:

```python
"Modelo 7": {
    "name": "Tu nuevo modelo",
    "description": "Descripción del modelo",
    "balanced": True,
    "has_subsampling": False
}
```

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos para el curso de Diseño Experimental.

## 👤 Autor

**R. Andre Vilca Solorzano**
- Código: 221181
- Universidad: [Tu Universidad]
- Email: [Tu email]

## 🙏 Agradecimientos

- Profesor del curso de Diseño Experimental
- Comunidad de Streamlit
- Datasets de ejemplo de Kaggle y GitHub

## 📚 Referencias

1. Montgomery, D. C. (2017). *Design and Analysis of Experiments*. 9th Edition.
2. Steel, R. G. D., & Torrie, J. H. (1980). *Principles and Procedures of Statistics*.
3. Documentación oficial de Streamlit: https://docs.streamlit.io
4. Documentación de SciPy: https://docs.scipy.org

---

## 🚀 Versiones

### v1.0.0 (2025-01-16)
- ✅ Implementación inicial
- ✅ 6 modelos de diseño experimental
- ✅ Visualizaciones avanzadas con Plotly
- ✅ Pruebas de comparación múltiple
- ✅ Sistema de carga de datos múltiple

---

**¡Gracias por usar el Sistema de Análisis DCA Bifactorial!** 🧪📊

Si encuentras algún problema o tienes sugerencias, no dudes en reportarlo.