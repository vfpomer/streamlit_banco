import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import traceback
import os
import pyodbc
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
import streamlit.components.v1 as components
import datetime
import logging
import shutil
import tempfile
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Panel Banco: Morosidad y Predicción",
    page_icon="🏦",
    layout="wide"
)


# ---- CSS personalizado para multiselect azul ----
st.markdown("""
    <style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    /* Checkbox seleccionado en azul */
    .stCheckbox > div:first-child > label > div[data-testid="stTick"] svg {
        color: #2563eb !important;
    }
    /* Para Streamlit >= 1.32, el check puede necesitar este selector extra */
    input[type="checkbox"]:checked + div:after {
        border-color: #2563eb !important;
        background: #2563eb !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Panel de Análisis Bancario: Morosidad y Predicción")
st.markdown("""
Este panel permite explorar los datos de usuarios, créditos y morosidad del banco, así como visualizar predicciones y modelos de riesgo.
""")

# ----------- Carga de datos -----------


@st.cache_data(ttl=3600)
def load_banco_data():
    import pyodbc
    import pandas as pd


    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        usuarios = pd.read_csv(os.path.join(base_dir, 'usuarios.csv'))
        creditos = pd.read_csv(os.path.join(base_dir, 'creditos.csv'))
        activos = pd.read_csv(os.path.join(base_dir, 'activo.csv'))
        monedas = pd.read_csv(os.path.join(base_dir, 'moneda.csv'))
        cuentas = pd.read_csv(os.path.join(base_dir, 'cuenta.csv'))
        
        return usuarios, creditos, activos, monedas, cuentas

    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.text(traceback.format_exc())
        return None, None, None


# ----------- Llamada a la función -----------
usuarios, creditos, activos, monedas, cuentas = load_banco_data()

if usuarios is None:
    st.stop()


# ----------- SIDEBAR: Filtros de búsqueda -----------
with st.sidebar:
    st.image("img/icono.jpg", use_container_width=True)  # Usa use_container_width en vez de use_column_width
    st.header("🔎 Filtros de búsqueda")

    # Filtros para activos financieros
    st.subheader("Filtrar Activos Financieros")
    mostrar_activos = st.checkbox("Mostrar activos financieros", value=True)
    tipo_activo_options = activos['tipo_activo'].unique().tolist() if not activos.empty and 'tipo_activo' in activos.columns else []
    tipo_activo_filter = st.multiselect(
        "Tipo de activo",
        options=tipo_activo_options,
        default=tipo_activo_options
    )

    # Filtros para monedas digitales
    st.subheader("Filtrar Monedas Digitales")
    mostrar_monedas = st.checkbox("Mostrar monedas digitales", value=True)
    tipo_moneda_options = monedas['tipo_moneda'].unique().tolist() if not monedas.empty and 'tipo_moneda' in monedas.columns else []
    tipo_moneda_filter = st.multiselect(
        "Tipo de moneda",
        options=tipo_moneda_options,
        default=tipo_moneda_options
    )

    # Filtros para cuentas bancarias
    st.subheader("Filtrar Cuentas Bancarias")
    mostrar_cuentas = st.checkbox("Mostrar cuentas bancarias", value=True)
    tipo_cuenta_options = cuentas['tipo_cuenta'].unique().tolist() if not cuentas.empty and 'tipo_cuenta' in cuentas.columns else []
    tipo_cuenta_filter = st.multiselect(
        "Tipo de cuenta",
        options=tipo_cuenta_options,
        default=tipo_cuenta_options
    )

# ----------- Aplicar filtros a activos financieros -----------
activos_filtrados = activos.copy()
if mostrar_activos:
    if tipo_activo_filter and 'tipo_activo' in activos_filtrados.columns:
        activos_filtrados = activos_filtrados[activos_filtrados['tipo_activo'].isin(tipo_activo_filter)]
else:
    activos_filtrados = activos_filtrados.iloc[0:0]  # DataFrame vacío si no se quiere mostrar

# ----------- Aplicar filtros a monedas digitales -----------
monedas_filtradas = monedas.copy()
if mostrar_monedas:
    if tipo_moneda_filter and 'tipo_moneda' in monedas_filtradas.columns:
        monedas_filtradas = monedas_filtradas[monedas_filtradas['tipo_moneda'].isin(tipo_moneda_filter)]
else:
    monedas_filtradas = monedas_filtradas.iloc[0:0]

# ----------- Aplicar filtros a cuentas bancarias -----------
cuentas_filtradas = cuentas.copy()
if mostrar_cuentas:
    if tipo_cuenta_filter and 'tipo_cuenta' in cuentas_filtradas.columns:
        cuentas_filtradas = cuentas_filtradas[cuentas_filtradas['tipo_cuenta'].isin(tipo_cuenta_filter)]
else:
    cuentas_filtradas = cuentas_filtradas.iloc[0:0]

# ----------- Funciones para el Sistema RAG -----------
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)

# Clase RAGSystem con conocimiento general
class SimpleRAGSystem:
    def __init__(self, pdf_paths=None):
        self.pdf_paths = pdf_paths or []
        self.documents = []

    def load_documents(self):
        if self.pdf_paths:
            # Cargar PDFs simulados
            self.documents = [f"Documento {i}: Contenido financiero" for i in range(len(self.pdf_paths))]
        else:
            # Conocimiento general embebido (puede crecer)
            self.documents = [
                "La morosidad se refiere al incumplimiento de pagos. Está influenciada por ingresos, edad, historial y estado civil.",
                "Las personas menores de 40 años suelen tener mayor riesgo de morosidad debido a menor estabilidad laboral o ingresos más bajos.",
                "El estado civil también influye: personas separadas o divorciadas pueden enfrentar mayores gastos y menor capacidad de pago.",
                "Los modelos de scoring crediticio consideran edad, ingresos, historial de pagos, empleo y estado civil.",
                "Una alta morosidad afecta negativamente la rentabilidad bancaria y obliga a provisionar pérdidas.",
                "El análisis predictivo ayuda a detectar clientes con alto riesgo crediticio antes del impago.",
                "El historial crediticio y la estabilidad financiera son claves para acceder a créditos con mejores condiciones."
            ]
        return True

    def create_vectorstore(self):
        # Simulación: nada que hacer aún
        return True

    def setup_chain(self):
        # Simulación: nada que hacer aún
        return True

    def rewrite_question(self, question):
        if len(question) < 10:
            return f"¿Podrías explicar más sobre: {question}?"
        return question

    def answer_question(self, question):
        question_lower = question.lower()

        # Respuestas específicas para preguntas comunes complejas
        if "menor de 40" in question_lower or "joven" in question_lower:
            return "Ser menor de 40 años puede aumentar el riesgo de morosidad debido a una posible menor estabilidad laboral o menores ingresos."

        if "separado" in question_lower or "divorciado" in question_lower:
            return "Estar separado o divorciado puede influir en el riesgo crediticio, ya que puede implicar mayores gastos u obligaciones financieras."

        if "estado civil" in question_lower and "morosidad" in question_lower:
            return "El estado civil es una variable considerada en modelos de scoring. Personas separadas o con cargas familiares pueden ser evaluadas con mayor riesgo."

        if "edad" in question_lower and "morosidad" in question_lower:
            return "La edad puede influir en el riesgo de morosidad. Los modelos crediticios suelen considerar que los perfiles más jóvenes pueden tener mayor riesgo."

        # Búsqueda por temas generales
        topics = {
            "morosidad": self.documents[0],
            "creditos": self.documents[6],
            "riesgo": self.documents[5],
            "banco": self.documents[4],
            "prediccion": self.documents[5],
            "scoring": self.documents[3],
            "analisis": self.documents[5],
            "edad": self.documents[1],
            "estado civil": self.documents[2],
            "historial": self.documents[6]
        }

        for key, response in topics.items():
            if key in question_lower:
                return response

        # Respuesta por defecto
        return f"No encontré una respuesta directa para tu consulta, pero te puedo decir que los factores como edad, estado civil, ingresos e historial crediticio son determinantes en la evaluación del riesgo financiero."



# ----------- Tabs principales -----------
tabs = st.tabs([
    "📊 Estadística Bancaria",
    "📈 Predicción Morosidad 6 meses",
    "🤖 Modelo de Morosidad",
	"🧠 Asistente RAG",
    "📝 Conclusiones"
])


# ----------- Tab 1: EDA Banco -----------
with tabs[0]:
    st.header("Análisis Exploratorio de Datos Bancarios")

    # Filtros en una sola fila
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
    with col1:
        dni_input = st.text_input("DNI")
    with col2:
        nombre_input = st.text_input("Nombre")
    with col3:
        moroso_filter = st.selectbox("Moroso", options=["Todos", "Sí", "No"])
    with col4:
        estado_opciones = st.selectbox("Estado Crédito",options=["Todos", "pendiente", "pagado", "vencido"])
    # --- Filtrado usuarios ---
    usuarios_filtrados = usuarios.copy()
    if dni_input:
        usuarios_filtrados = usuarios_filtrados[usuarios_filtrados['dni'].astype(str).str.contains(dni_input, case=False, na=False)]
    if nombre_input:
        usuarios_filtrados = usuarios_filtrados[usuarios_filtrados['nombre'].astype(str).str.contains(nombre_input, case=False, na=False)]
    if moroso_filter == "Sí":
        usuarios_filtrados = usuarios_filtrados[usuarios_filtrados['es_moroso'] == 1]
    elif moroso_filter == "No":
        usuarios_filtrados = usuarios_filtrados[usuarios_filtrados['es_moroso'] == 0]

    # --- Limpieza y unificación de tipos y espacios ---
    usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str).str.strip()
    creditos['usuario_id'] = creditos['usuario_id'].astype(str).str.strip()

    # --- Comprobación de intersección ---
    # --- Comprobación de intersección ---
    interseccion = set(creditos['usuario_id']).intersection(set(usuarios_filtrados['id']))

    if len(interseccion) == 0:
        # st.error("No hay IDs comunes entre usuarios filtrados y créditos, no se mostrarán créditos.")
        creditos_filtrados = creditos.iloc[0:0]  # DataFrame vacío
    else:
        # --- Filtrado créditos ---
        creditos_filtrados = creditos[creditos['usuario_id'].isin(usuarios_filtrados['id'])]
        # Solo filtra por estado si "Todos" no está seleccionado y hay algún estado seleccionado
        if estado_opciones != "Todos":
            creditos_filtrados = creditos_filtrados[creditos_filtrados['estado'] == estado_credito]

    # st.write(f"Créditos filtrados: {creditos_filtrados.shape}")
    # st.write("Ejemplo usuario_id en créditos filtrados:", creditos_filtrados['usuario_id'].head(10).tolist())

    # --- Visualización o mensajes ---
    if creditos_filtrados.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados en créditos.")
    #else:
        # Aquí va tu código para mostrar gráficos y análisis con creditos_filtrados
        #st.success(f"Mostrando {creditos_filtrados.shape[0]} créditos filtrados.")

        # Ejemplo simple:
        #st.dataframe(creditos_filtrados.head(5))

    eda_tabs = st.tabs([
        "👤 Usuarios",
        "💳 Créditos",
        "💰 Activos Financieros",
        "🏦 Cuentas Bancarias",
        "🪙 Monedas Digitales"
    ])

    # --- Usuarios ---
    with eda_tabs[0]:
        if usuarios_filtrados.empty:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")
        else:
            st.subheader("Mapa interactivo de usuarios, activos, monedas y créditos por provincia")
            # Calcula la ruta al archivo HTML del mapa de forma robusta
            mapa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'mapa_usuarios_provincias.html'))

            if os.path.exists(mapa_path):
                with open(mapa_path, "r", encoding="utf-8") as f:
                    html_mapa = f.read()
                components.html(html_mapa, height=600)
            else:
                st.warning("No se encontró el mapa interactivo. Por favor, genera el archivo 'mapa_usuarios_provincias.html' en la carpeta docs.")

            st.subheader("Usuarios por Provincia")
            if 'provincia' in usuarios_filtrados.columns:
                provincia_counts = usuarios_filtrados['provincia'].value_counts().reset_index()
                provincia_counts.columns = ['provincia', 'usuarios']
                fig = px.bar(provincia_counts, x='provincia', y='usuarios', title="Cantidad de usuarios por provincia")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columna 'provincia' en usuarios.")

            st.subheader("Distribución de Morosidad")
            if 'es_moroso' in usuarios_filtrados.columns:
                moroso_counts = usuarios_filtrados['es_moroso'].value_counts().rename({0: "No moroso", 1: "Moroso"})
                fig = px.pie(values=moroso_counts.values, names=moroso_counts.index, title="Distribución de morosidad")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columna 'es_moroso' en usuarios.")

            st.subheader("Usuarios por Estado Civil")
            if 'estado_civil' in usuarios_filtrados.columns:
                estado_civil_counts = usuarios_filtrados['estado_civil'].value_counts().reset_index()
                estado_civil_counts.columns = ['estado_civil', 'usuarios']
                fig = px.bar(estado_civil_counts, x='estado_civil', y='usuarios', title="Usuarios por estado civil")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columna 'estado_civil' en usuarios.")

            st.subheader("Usuarios por Profesión")
            if 'profesion' in usuarios_filtrados.columns:
                profesion_counts = usuarios_filtrados['profesion'].value_counts().head(20).reset_index()
                profesion_counts.columns = ['profesion', 'usuarios']
                fig = px.bar(profesion_counts, x='profesion', y='usuarios', title="Top 20 profesiones de usuarios")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columna 'profesion' en usuarios.")

            st.subheader("Distribución de Edad de Usuarios")
            if 'fecha_nacimiento' in usuarios_filtrados.columns:
                hoy = pd.Timestamp(datetime.date.today())
                usuarios_filtrados['edad'] = pd.to_datetime(usuarios_filtrados['fecha_nacimiento'], errors='coerce').apply(
                    lambda x: hoy.year - x.year - ((hoy.month, hoy.day) < (x.month, x.day)) if pd.notnull(x) else np.nan
                )
                # Define los grupos de edad
                bins = [18, 25, 35, 45, 55, 65, 75, 100]
                labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
                usuarios_filtrados['grupo_edad'] = pd.cut(usuarios_filtrados['edad'], bins=bins, labels=labels, right=False)
                edad_counts = usuarios_filtrados['grupo_edad'].value_counts().sort_index()
                fig = px.bar(
                    x=edad_counts.index.astype(str),
                    y=edad_counts.values,
                    labels={'x': 'Grupo de Edad', 'y': 'Cantidad de Usuarios'},
                    title="Distribución de usuarios por grupo de edad"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columna 'fecha_nacimiento' en usuarios.")

            st.subheader("Distribución de Salario de Usuarios")
            if 'salario' in usuarios_filtrados.columns:
                fig = px.histogram(
                    usuarios_filtrados,
                    x='salario',
                    nbins=20,  # Menos bins para mayor claridad
                    title="Distribución de salario de usuarios"
                )
                fig.update_layout(
                    bargap=0.2,  # Espacio entre barras
                    width=900,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columna 'salario' en usuarios.")

            # Mostrar gráficos de morosos solo si el filtro es_moroso NO está en "No"
            if moroso_filter != "No":
                st.subheader("Usuarios Morosos por Provincia")
                if 'provincia' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
                    morosos_prov = usuarios_filtrados[usuarios_filtrados['es_moroso'] == 1]['provincia'].value_counts().reset_index()
                    morosos_prov.columns = ['provincia', 'morosos']
                    fig = px.bar(morosos_prov, x='provincia', y='morosos', title="Morosos por provincia")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para morosos por provincia.")

                st.subheader("Usuarios Morosos por Estado Civil")
                if 'estado_civil' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
                    morosos_ec = usuarios_filtrados[usuarios_filtrados['es_moroso'] == 1]['estado_civil'].value_counts().reset_index()
                    morosos_ec.columns = ['estado_civil', 'morosos']
                    fig = px.bar(morosos_ec, x='estado_civil', y='morosos', title="Morosos por estado civil")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para morosos por estado civil.")

                st.subheader("Usuarios Morosos por Profesión")
                if 'profesion' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
                    morosos_prof = usuarios_filtrados[usuarios_filtrados['es_moroso'] == 1]['profesion'].value_counts().head(20).reset_index()
                    morosos_prof.columns = ['profesion', 'morosos']
                    fig = px.bar(morosos_prof, x='profesion', y='morosos', title="Top 20 profesiones de usuarios morosos")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para morosos por profesión.")

           

            # --- Análisis financiero avanzado de usuarios ---

            # --- Análisis financiero avanzado de usuarios ---

            # 1. Correlación financiera entre variables
            st.subheader("Correlación entre Variables Financieras de Usuarios")
            try:
                # Preparamos los datos agregados por usuario
                monto_total_creditos = creditos_filtrados.groupby('usuario_id')['monto'].sum().reset_index(name='monto_total_creditos')
                activos_por_usuario = activos_filtrados.groupby('usuario_id')['monto'].sum().reset_index(name='valor_total_activos') if not activos_filtrados.empty else pd.DataFrame(columns=['usuario_id','valor_total_activos'])
                monedas_por_usuario = monedas_filtradas.groupby('usuario_id')['valor_actual'].sum().reset_index(name='total_cripto') if not monedas_filtradas.empty else pd.DataFrame(columns=['usuario_id','total_cripto'])
                cuentas_por_usuario = cuentas_filtradas.groupby('usuario_id').size().reset_index(name='cantidad_cuentas') if not cuentas_filtradas.empty else pd.DataFrame(columns=['usuario_id','cantidad_cuentas'])

                # Forzar usuario_id a string en todos los DataFrames
                df_usuarios_corr = usuarios_filtrados.rename(columns={'id': 'usuario_id'}).copy()
                df_usuarios_corr['usuario_id'] = df_usuarios_corr['usuario_id'].astype(str)
                monto_total_creditos['usuario_id'] = monto_total_creditos['usuario_id'].astype(str)
                activos_por_usuario['usuario_id'] = activos_por_usuario['usuario_id'].astype(str)
                monedas_por_usuario['usuario_id'] = monedas_por_usuario['usuario_id'].astype(str)
                cuentas_por_usuario['usuario_id'] = cuentas_por_usuario['usuario_id'].astype(str)

                df_merged = (
                    df_usuarios_corr
                    .merge(monto_total_creditos, on='usuario_id', how='left')
                    .merge(activos_por_usuario, on='usuario_id', how='left')
                    .merge(monedas_por_usuario, on='usuario_id', how='left')
                    .merge(cuentas_por_usuario, on='usuario_id', how='left')
                )

                corr_vars = ['monto_total_creditos', 'valor_total_activos', 'total_cripto', 'cantidad_cuentas']
                corr = df_merged[corr_vars].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Correlación entre Variables Financieras")
                st.pyplot(fig)
            except Exception as e:
                st.info(f"No se pudo calcular la correlación financiera: {e}")

            # 3. Clustering de usuarios por riesgo (KMeans)
            st.subheader("Distribución de Usuarios por Clúster de Riesgo")
            try:
                from sklearn.preprocessing import LabelEncoder
                from sklearn.cluster import KMeans

                df_cluster = usuarios_filtrados.copy()
                cat_vars = ['provincia', 'estado_civil', 'profesion', 'nacionalidad']
                encoders = {}
                for col in cat_vars:
                    df_cluster[col] = df_cluster[col].fillna("Desconocido")
                    le = LabelEncoder()
                    df_cluster[col] = le.fit_transform(df_cluster[col].astype(str))
                    encoders[col] = le  # Guardamos el encoder para decodificar luego

                # --- Normalizar salario y otras columnas numéricas con coma decimal ---
                for col in ['salario', 'edad']:
                    if col in df_cluster.columns:
                        df_cluster[col] = (
                            df_cluster[col]
                            .astype(str)
                            .str.replace(',', '.', regex=False)
                            .replace('', np.nan)
                            .astype(float)
                        )

                features = ['edad', 'salario', 'provincia', 'estado_civil', 'profesion']
                X = df_cluster[features].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                df_cluster['cluster_riesgo'] = kmeans.fit_predict(X_scaled)

                fig, ax = plt.subplots(figsize=(7, 4))
                sns.countplot(x='cluster_riesgo', data=df_cluster, palette='tab10', ax=ax)
                ax.set_title("Distribución de Usuarios por Clúster de Riesgo")
                st.pyplot(fig)

                # Perfil de clusters (tabla)
                st.markdown("**Perfil promedio de cada clúster:**")
                perfil = df_cluster.groupby('cluster_riesgo')[features].mean().round(2).reset_index()

                # Decodificar columnas categóricas
                for col in ['provincia', 'estado_civil', 'profesion']:
                    if col in perfil.columns:
                        perfil[col] = perfil[col].round().astype(int)
                        perfil[col] = encoders[col].inverse_transform(perfil[col])

                st.dataframe(perfil)
            except Exception as e:
                st.info(f"No se pudo calcular el clustering: {e}")


            # 4. Usuarios por nacionalidad (Top 10)
            st.subheader("Top 10 Nacionalidades con más Usuarios")
            try:
                usuarios_nacionalidad = usuarios_filtrados.groupby('nacionalidad').size().reset_index(name='cantidad_usuarios')
                top_nacionalidades = usuarios_nacionalidad.sort_values('cantidad_usuarios', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=top_nacionalidades, x='nacionalidad', y='cantidad_usuarios', palette='viridis', ax=ax)
                ax.set_title("Top 10 Nacionalidades con más Usuarios")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"No se pudo mostrar usuarios por nacionalidad: {e}")

            # 5. Distribución de antigüedad de clientes
            st.subheader("Distribución de Antigüedad de Clientes (en Años)")
            try:
                if 'antiguedad_cliente' in usuarios_filtrados.columns:
                    usuarios_filtrados['antiguedad_cliente'] = pd.to_datetime(usuarios_filtrados['antiguedad_cliente'], errors='coerce')
                    usuarios_filtrados['antiguedad_anos'] = ((pd.Timestamp('today') - usuarios_filtrados['antiguedad_cliente']).dt.days / 365).round(2)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(usuarios_filtrados['antiguedad_anos'].dropna(), bins=30, kde=True, color='steelblue', ax=ax)
                    ax.set_title('Distribución de Antigüedad de Clientes (en Años)')
                    ax.set_xlabel('Antigüedad (años)')
                    ax.set_ylabel('Frecuencia')
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"No se pudo mostrar la antigüedad de clientes: {e}")

            # 6. Edad media de usuarios con créditos por provincia
            st.subheader("Edad Media de Usuarios con Créditos por Provincia")
            try:
                if 'edad' in usuarios_filtrados.columns and 'provincia' in usuarios_filtrados.columns:
                    edad_creditos_provincia = creditos_filtrados.merge(
                        usuarios_filtrados[['id', 'edad', 'provincia']],
                        left_on='usuario_id', right_on='id', how='left'
                    )
                    edad_media_por_provincia = edad_creditos_provincia.groupby('provincia')['edad'].mean().reset_index()
                    edad_media_por_provincia = edad_media_por_provincia.sort_values('edad', ascending=False)
                    fig, ax = plt.subplots(figsize=(16, 8))
                    sns.barplot(data=edad_media_por_provincia, x='provincia', y='edad', palette='viridis', ax=ax)
                    ax.set_title("Edad Media de Usuarios con Créditos por Provincia", fontsize=16)
                    ax.set_xlabel("Provincia", fontsize=12)
                    ax.set_ylabel("Edad Media", fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    for p in ax.patches:
                        ax.annotate(f'{p.get_height():.1f}',
                                    (p.get_x() + p.get_width() / 2, p.get_height()),
                                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"No se pudo calcular la edad media por provincia: {e}")

            # 7. Edad media de usuarios con créditos por nacionalidad (Top 30)
            st.subheader("Edad Media de Usuarios con Créditos por Nacionalidad (Top 30)")
            try:
                if 'edad' in usuarios_filtrados.columns and 'nacionalidad' in usuarios_filtrados.columns:
                    temp = creditos_filtrados.merge(
                        usuarios_filtrados[['id', 'edad', 'nacionalidad']],
                        left_on='usuario_id', right_on='id', how='left'
                    )
                    edad_media_nacionalidad = temp.groupby('nacionalidad')['edad'].mean().reset_index()
                    top_nacionalidades = edad_media_nacionalidad.sort_values('edad', ascending=False).head(30)
                    fig, ax = plt.subplots(figsize=(16, 8))
                    sns.barplot(data=top_nacionalidades, x='nacionalidad', y='edad', palette='viridis', ax=ax)
                    ax.set_title("Edad Media de Usuarios con Créditos por Nacionalidad (Top 30)", fontsize=16)
                    ax.set_xlabel("Nacionalidad", fontsize=12)
                    ax.set_ylabel("Edad Media", fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    for p in ax.patches:
                        height = p.get_height()
                        ax.annotate(f'{height:.1f}',
                                    (p.get_x() + p.get_width() / 2, height),
                                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3), textcoords='offset points')
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"No se pudo calcular la edad media por nacionalidad: {e}")

            # 8. Estadísticas básicas de créditos por usuario filtrado
            st.subheader("Estadísticas Básicas de Créditos de Usuarios Filtrados")
            try:
                if not creditos_filtrados.empty:
                    creditos_usuario = creditos_filtrados.groupby('usuario_id')['monto'].sum().reset_index(name='monto_total_creditos')
                    st.write(f"**Crédito máximo:** {creditos_usuario['monto_total_creditos'].max():,.2f} €.")
                    st.write(f"**Crédito mínimo:** {creditos_usuario['monto_total_creditos'].min():,.2f} €.")
                    st.write(f"**Crédito promedio:** {creditos_usuario['monto_total_creditos'].mean():,.2f} €.")
                    st.write(f"**Total de créditos:** {creditos_usuario['monto_total_creditos'].sum():,.2f} €.")
            except Exception as e:
                st.info(f"No se pudo calcular las estadísticas básicas de créditos: {e}")

            st.markdown("### Tabla de usuarios filtrados")
            st.dataframe(usuarios_filtrados, use_container_width=True)

    # Pestaña Créditos
    with eda_tabs[1]:
        if creditos_filtrados.empty:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")
        else:
            st.subheader("Análisis de Créditos")

            # Top 30 créditos individuales más elevados
            st.markdown("**Top 30 Créditos Individuales Más Elevados**")
            if not creditos_filtrados.empty:
                # Unimos con usuarios para mostrar nombre y apellido si existen
                if 'nombre' in usuarios_filtrados.columns and 'apellido' in usuarios_filtrados.columns:
                    top_creditos = creditos_filtrados.merge(
                        usuarios_filtrados[['id', 'nombre', 'apellido']],
                        left_on='usuario_id', right_on='id', how='left'
                    )
                    top_creditos = top_creditos.sort_values('monto', ascending=False).head(30)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(
                        data=top_creditos,
                        x='monto',
                        y=top_creditos['nombre'] + " " + top_creditos['apellido'] + " (" + top_creditos['usuario_id'].astype(str) + ")",
                        palette='Blues_r',
                        ax=ax
                    )
                    ax.set_title('Top 30 Créditos Individuales Más Elevados')
                    ax.set_xlabel('Monto del Crédito')
                    ax.set_ylabel('Usuario (ID)')
                    for i, v in enumerate(top_creditos['monto']):
                        ax.text(v, i, f"{v:,.2f}", color='black', va='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No hay columnas 'nombre' y 'apellido' en usuarios para mostrar el detalle.")
            else:
                st.info("No hay créditos para mostrar.")

            # Créditos por Estado
            st.markdown("**Cantidad de Créditos por Estado**")
            creditos_por_estado = creditos_filtrados['estado'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = creditos_por_estado.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
            for i, valor in enumerate(creditos_por_estado):
                ax.text(i, valor + 0.5, str(valor), ha='center', va='bottom', fontsize=10)
            ax.set_title('Cantidad de Créditos por Estado')
            ax.set_xlabel('Estado del Crédito')
            ax.set_ylabel('Número de Créditos')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            creditos_por_usuario = creditos_filtrados.groupby('usuario_id').size().reset_index(name='creditos')

            # --- Distribución de créditos por usuario (histograma) ---
            st.markdown("**Distribución del Número de Créditos por Usuario**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(creditos_por_usuario['creditos'], bins=30, kde=True, ax=ax)
            ax.set_title("Distribución del Número de Créditos por Usuario")
            ax.set_xlabel("Cantidad de Créditos")
            ax.set_ylabel("Frecuencia")
            plt.tight_layout()
            st.pyplot(fig)

            # --- Boxplot de créditos por usuario ---
            st.markdown("**Resumen Estadístico de Créditos por Usuario**")
            fig, ax = plt.subplots(figsize=(10, 2))
            sns.boxplot(x=creditos_por_usuario['creditos'], color='skyblue', ax=ax)
            media = creditos_por_usuario['creditos'].mean()
            mediana = creditos_por_usuario['creditos'].median()
            p90 = creditos_por_usuario['creditos'].quantile(0.90)
            p95 = creditos_por_usuario['creditos'].quantile(0.95)
            p99 = creditos_por_usuario['creditos'].quantile(0.99)
            for valor, etiqueta, color in zip([media, mediana, p90, p95, p99], ['Media', 'Mediana', 'P90', 'P95', 'P99'], ['red', 'green', 'orange', 'purple', 'black']):
                ax.axvline(valor, color=color, linestyle='--', label=f'{etiqueta}: {valor:.1f}')
            ax.set_title("Resumen Estadístico de Créditos por Usuario")
            ax.set_xlabel("Cantidad de Créditos")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # Barplot: Usuarios por cantidad de créditos
            st.markdown("**Usuarios por Cantidad de Créditos**")
            creditos_por_usuario = creditos_filtrados.groupby('usuario_id').size().reset_index(name='creditos')
            usuarios_por_cantidad = creditos_por_usuario['creditos'].value_counts().sort_index().reset_index()
            usuarios_por_cantidad.columns = ['cantidad_creditos', 'num_usuarios']

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=usuarios_por_cantidad, x='cantidad_creditos', y='num_usuarios', color='skyblue', ax=ax)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9)
            ax.set_title('Usuarios por Cantidad de Créditos')
            ax.set_xlabel('Cantidad de Créditos')
            ax.set_ylabel('Número de Usuarios')
            plt.tight_layout()
            st.pyplot(fig)

            # --- Histograma del monto total de créditos por usuario ---
            st.markdown("**Distribución del Monto Total de Créditos por Usuario**")
            monto_creditos = creditos_filtrados.groupby('usuario_id')['monto'].sum().reset_index(name='monto_total_creditos')
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(monto_creditos['monto_total_creditos'], bins=30, kde=True, ax=ax)
            ax.set_title("Distribución del Monto Total de Créditos por Usuario")
            ax.set_xlabel("Monto Total de Créditos")
            ax.set_ylabel("Frecuencia")
            plt.tight_layout()
            st.pyplot(fig)

            # --- Histograma logarítmico del monto total de créditos por usuario ---
            st.markdown("**Distribución Log del Monto Total de Créditos por Usuario**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(np.log1p(monto_creditos['monto_total_creditos']), bins=30, kde=True, color='skyblue', ax=ax)
            ax.set_title("Distribución Log del Monto Total de Créditos por Usuario")
            ax.set_xlabel("Log(Monto Total + 1)")
            ax.set_ylabel("Frecuencia")
            plt.tight_layout()
            st.pyplot(fig)

            # --- Promedio del monto de crédito por tipo de empleo (Top 20) ---
            if 'profesion' in usuarios_filtrados.columns and 'salario' in usuarios_filtrados.columns:
                st.markdown("**Top 20: Promedio de Monto de Crédito por Tipo de Empleo**")
                creditos_filtrados['usuario_id'] = creditos_filtrados['usuario_id'].astype(str)
                usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str)
                creditos_empleo = creditos_filtrados.merge(usuarios_filtrados, left_on='usuario_id', right_on='id')
                promedio_credito = creditos_empleo.groupby('profesion')['monto'].mean().reset_index()
                top_20_profesiones = promedio_credito.sort_values(by='monto', ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(15, 10))
                barplot = sns.barplot(data=top_20_profesiones, x='profesion', y='monto', palette='Set2', ax=ax)
                for p in barplot.patches:
                    height = p.get_height()
                    barplot.text(p.get_x() + p.get_width() / 2., height + 0.5, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                ax.set_title("Top 20: Promedio de Monto de Crédito por Tipo de Empleo")
                plt.xticks(rotation=45, ha='right')
                ax.set_ylabel("Promedio de Monto")
                ax.set_xlabel("Cargo de Empleo")
                plt.tight_layout()
                st.pyplot(fig)

            # --- Créditos por Provincia ---
            if 'provincia' in usuarios_filtrados.columns:
                st.markdown("**Cantidad de Créditos por Provincia**")
                creditos_provincia = creditos_filtrados.merge(
                    usuarios_filtrados[['id', 'provincia']], left_on='usuario_id', right_on='id'
                )
                creditos_provincia = creditos_provincia.groupby('provincia').size().reset_index(name='cantidad_creditos')
                fig, ax = plt.subplots(figsize=(16, 6))
                sns.barplot(data=creditos_provincia.sort_values('cantidad_creditos', ascending=False), x='provincia', y='cantidad_creditos', palette='crest', ax=ax)
                plt.xticks(rotation=45, ha='right')
                ax.set_title("Cantidad de Créditos por Provincia")
                ax.set_ylabel("N° Créditos")
                ax.set_xlabel("Provincia")
                plt.tight_layout()
                st.pyplot(fig)

            # --- Créditos por Nacionalidad ---
            if 'nacionalidad' in usuarios_filtrados.columns:
                st.markdown("**Top 20 Nacionalidades con más Créditos**")
                creditos_nacionalidad = creditos_filtrados.merge(
                    usuarios_filtrados[['id', 'nacionalidad']], left_on='usuario_id', right_on='id'
                )
                creditos_nacionalidad = creditos_nacionalidad.groupby('nacionalidad').size().reset_index(name='cantidad_creditos')
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.barplot(data=creditos_nacionalidad.sort_values('cantidad_creditos', ascending=False).head(20), x='nacionalidad', y='cantidad_creditos', palette='viridis', ax=ax)
                plt.xticks(rotation=45, ha='right')
                ax.set_title("Top 20 Nacionalidades con más Créditos")
                plt.tight_layout()
                st.pyplot(fig)

            # --- ¿Los morosos tienen más créditos? ---
            if 'es_moroso' in usuarios_filtrados.columns:
                st.markdown("**Cantidad de Créditos según si el Usuario es Moroso**")
                usuarios_mora = usuarios_filtrados[['id', 'es_moroso']].rename(columns={'id': 'usuario_id'})
                creditos_mora = creditos_filtrados.merge(usuarios_mora, on='usuario_id')
                creditos_mora = creditos_mora.groupby('es_moroso').size().reset_index(name='cantidad_creditos')
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.barplot(data=creditos_mora, x='es_moroso', y='cantidad_creditos', palette='flare', ax=ax)
                ax.set_title("Cantidad de Créditos según si el Usuario es Moroso")
                ax.set_xlabel("Moroso (0 = No, 1 = Sí)")
                ax.set_ylabel("Cantidad de Créditos")
                plt.tight_layout()
                st.pyplot(fig)

            # --- Usuarios con más de 3 créditos (Top 10) ---
            if 'nombre' in usuarios_filtrados.columns and 'apellido' in usuarios_filtrados.columns:
                st.markdown("**Top 10 Usuarios con Más de 3 Créditos**")
                creditos_por_usuario = creditos_filtrados.groupby('usuario_id').size().reset_index(name='cantidad_creditos')
                usuarios_mas_creditos = usuarios_filtrados[usuarios_filtrados['id'].isin(creditos_por_usuario[creditos_por_usuario['cantidad_creditos'] > 3]['usuario_id'])]
                # Unimos para traer la cantidad de créditos de cada usuario
                top_usuarios = usuarios_mas_creditos.merge(
                    creditos_por_usuario, left_on='id', right_on='usuario_id'
                )[['id', 'nombre', 'apellido', 'cantidad_creditos']]
                top_usuarios = top_usuarios.sort_values(by='cantidad_creditos', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=top_usuarios, x='cantidad_creditos', y='apellido', palette='viridis', ax=ax)
                ax.set_title("Top 10 Usuarios con Más Créditos")
                ax.set_xlabel("Cantidad de Créditos")
                ax.set_ylabel("Apellido")
                plt.tight_layout()
                st.pyplot(fig)

            # --- Distribución de la duración de los créditos ---
            if 'fecha_inicio' in creditos_filtrados.columns and 'fecha_fin' in creditos_filtrados.columns:
                st.markdown("**Distribución de la Duración de los Créditos (días)**")
                df_temp = creditos_filtrados.copy()
                df_temp['fecha_inicio'] = pd.to_datetime(df_temp['fecha_inicio'])
                df_temp['fecha_fin'] = pd.to_datetime(df_temp['fecha_fin'])
                df_temp['duracion_credito'] = (df_temp['fecha_fin'] - df_temp['fecha_inicio']).dt.days
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.histplot(df_temp['duracion_credito'], bins=30, kde=True, color='teal', ax=ax)
                ax.set_title("Distribución de la Duración de los Créditos (días)")
                ax.set_xlabel("Duración (días)")
                ax.set_ylabel("Cantidad de Créditos")
                plt.tight_layout()
                st.pyplot(fig)

            # --- Mapa de calor de créditos por provincia y estado ---
            if 'provincia' in usuarios_filtrados.columns and 'estado' in creditos_filtrados.columns:
                st.markdown("**Mapa de Calor de Créditos por Provincia y Estado**")
                creditos_provincia_estado = creditos_filtrados.merge(
                    usuarios_filtrados[['id', 'provincia']], left_on='usuario_id', right_on='id', how='left'
                )
                pivot = creditos_provincia_estado.groupby(['provincia', 'estado']).size().unstack().fillna(0)
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(pivot, annot=True, fmt='g', cmap='viridis', cbar_kws={'label': 'Cantidad de Créditos'}, ax=ax)
                ax.set_title("Mapa de Calor de Créditos por Provincia y Estado")
                ax.set_xlabel("Estado del Crédito")
                ax.set_ylabel("Provincia")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # --- Mapa de calor de créditos por nacionalidad y estado ---
            if 'nacionalidad' in usuarios_filtrados.columns and 'estado' in creditos_filtrados.columns:
                st.markdown("**Mapa de Calor de Créditos por Nacionalidad y Estado**")
                creditos_nacionalidad_estado = creditos_filtrados.merge(
                    usuarios_filtrados[['id', 'nacionalidad']], left_on='usuario_id', right_on='id', how='left'
                )
                pivot = creditos_nacionalidad_estado.groupby(['nacionalidad', 'estado']).size().unstack().fillna(0)
                fig, ax = plt.subplots(figsize=(12, 20))
                sns.heatmap(pivot, annot=True, fmt='g', cmap='viridis', cbar_kws={'label': 'Cantidad de Créditos'}, ax=ax)
                ax.set_title("Mapa de Calor de Créditos por Nacionalidad y Estado")
                ax.set_xlabel("Estado del Crédito")
                ax.set_ylabel("Nacionalidad")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

    # --- Activos Financieros ---
    with eda_tabs[2]:
        if mostrar_activos:
            if activos_filtrados.empty:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")
            else:
                st.subheader("Activos Financieros por Usuario (Top 30)")
                if 'usuario_id' in activos_filtrados.columns and 'monto' in activos_filtrados.columns:
                    activos_sum = activos_filtrados.groupby('usuario_id')['monto'].sum().sort_values(ascending=False).head(30).reset_index()
                    activos_sum['usuario_id'] = activos_sum['usuario_id'].astype(str)  # <-- Solución clave
                    activos_sum.columns = ['usuario_id', 'monto_total']
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(
                        data=activos_sum,
                        y='usuario_id',
                        x='monto_total',
                        palette='Blues_r',
                        ax=ax
                    )
                    ax.set_title("Top 30 usuarios por monto total en activos", fontsize=16)
                    ax.set_xlabel("Monto total", fontsize=14)
                    ax.set_ylabel("Usuario ID", fontsize=14)
                    for i, v in enumerate(activos_sum['monto_total']):
                        ax.text(v, i, f"{v:,.2f}", color='black', va='center', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No hay columnas necesarias en activos financieros.")

                # --- Análisis avanzado de activos financieros ---

                # 1. Tipos de activos financieros por nacionalidad (Top 30)
                if not activos_filtrados.empty and not usuarios_filtrados.empty:
                    st.subheader("Cantidad de Activos Financieros por Nacionalidad y Tipo de Activo (Top 30)")
                    try:
                        activos_filtrados['usuario_id'] = activos_filtrados['usuario_id'].astype(str)
                        usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str)
                        activos_nacionalidad = activos_filtrados.merge(
                            usuarios_filtrados[['id', 'nacionalidad']], left_on='usuario_id', right_on='id', how='left'
                        )
                        activos_nacionalidad = activos_nacionalidad.groupby(['nacionalidad', 'tipo_activo']).size().reset_index(name='cantidad_activos')
                        total_activos_por_nacionalidad = activos_nacionalidad.groupby('nacionalidad')['cantidad_activos'].sum().reset_index()
                        top30_nacionalidades = total_activos_por_nacionalidad.sort_values('cantidad_activos', ascending=False).head(30)['nacionalidad']
                        activos_nacionalidad_top30 = activos_nacionalidad[activos_nacionalidad['nacionalidad'].isin(top30_nacionalidades)]

                        fig, ax = plt.subplots(figsize=(16, 8))
                        sns.barplot(data=activos_nacionalidad_top30, x='nacionalidad', y='cantidad_activos', hue='tipo_activo', palette='viridis', ax=ax)
                        ax.set_title("Cantidad de Activos Financieros por Nacionalidad y Tipo de Activo (Top 30)", fontsize=16)
                        ax.set_xlabel("Nacionalidad", fontsize=14)
                        ax.set_ylabel("Cantidad de Activos", fontsize=14)
                        plt.xticks(rotation=45, ha='right', fontsize=12)
                        plt.yticks(fontsize=12)
                        for p in ax.patches:
                            height = p.get_height()
                            if height > 0:
                                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                                            ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), textcoords='offset points')
                        plt.legend(title='Tipo de Activo', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar activos por nacionalidad: {e}")

                # 2. Activos financieros por rango de edad y tipo
                if not activos_filtrados.empty and not usuarios_filtrados.empty and 'edad' in usuarios_filtrados.columns:
                    st.subheader("Cantidad de Activos Financieros por Rango de Edad y Tipo de Activo")
                    try:
                        activos_filtrados['usuario_id'] = activos_filtrados['usuario_id'].astype(str)
                        usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str)
                        activos_edad = activos_filtrados.merge(
                            usuarios_filtrados[['id', 'edad']], left_on='usuario_id', right_on='id', how='left'
                        )
                        activos_edad['rango_edad'] = pd.cut(activos_edad['edad'], bins=bins, labels=labels, right=True)
                        activos_edad_agg = activos_edad.groupby(['rango_edad', 'tipo_activo']).size().reset_index(name='cantidad_activos')

                        fig, ax = plt.subplots(figsize=(14, 7))
                        sns.barplot(data=activos_edad_agg, x='rango_edad', y='cantidad_activos', hue='tipo_activo', palette='viridis', ax=ax)
                        ax.set_title("Cantidad de Activos Financieros por Rango de Edad y Tipo de Activo", fontsize=16)
                        ax.set_xlabel("Rango de Edad", fontsize=14)
                        ax.set_ylabel("Cantidad de Activos", fontsize=14)
                        plt.xticks(rotation=0, ha='center', fontsize=12)
                        plt.yticks(fontsize=12)
                        plt.legend(title='Tipo de Activo', fontsize=12, title_fontsize=13)
                        for p in ax.patches:
                            height = p.get_height()
                            if height > 0:
                                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                                            ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3), textcoords='offset points')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar activos por rango de edad: {e}")

               # 3. Mapa de calor de activos financieros por provincia y tipo
                if not activos_filtrados.empty and not usuarios_filtrados.empty:
                    st.subheader("Mapa de Calor de Activos Financieros por Provincia y Tipo")
                    try:
                        activos_filtrados['usuario_id'] = activos_filtrados['usuario_id'].astype(str)
                        usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str)
                        activos_provincia_tipo = activos_filtrados.merge(
                            usuarios_filtrados[['id', 'provincia']], left_on='usuario_id', right_on='id', how='left'
                        )
                        pivot = activos_provincia_tipo.groupby(['provincia', 'tipo_activo']).size().unstack().fillna(0)
                        fig, ax = plt.subplots(figsize=(12, 20))
                        sns.heatmap(pivot, annot=True, fmt='g', cmap='viridis', cbar_kws={'label': 'Cantidad de Activos'}, ax=ax)
                        ax.set_title("Mapa de Calor de Activos Financieros por Provincia y Tipo")
                        ax.set_xlabel("Tipo de Activo")
                        ax.set_ylabel("Provincia")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el mapa de calor de activos: {e}")

                st.markdown("### Tabla de activos financieros filtrados")
                st.dataframe(activos_filtrados, use_container_width=True)

    # --- Cuentas Bancarias ---
    with eda_tabs[3]:
        if mostrar_cuentas:
            if cuentas_filtradas.empty:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")
            else:
                st.subheader("Cuentas Bancarias")
                if 'usuario_id' in cuentas_filtradas.columns:
                    cuentas_sum = cuentas_filtradas['usuario_id'].value_counts().head(30).reset_index()
                    cuentas_sum.columns = ['usuario_id', 'num_cuentas']
                    cuentas_sum['usuario_id'] = cuentas_sum['usuario_id'].astype(str)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(
                        data=cuentas_sum,
                        y='usuario_id',
                        x='num_cuentas',
                        palette='crest',
                        ax=ax
                    )
                    ax.set_title("Top 30 usuarios por número de cuentas bancarias", fontsize=16)
                    ax.set_xlabel("Nº cuentas", fontsize=14)
                    ax.set_ylabel("Usuario ID", fontsize=14)
                    for i, v in enumerate(cuentas_sum['num_cuentas']):
                        ax.text(v, i, f"{v}", color='black', va='center', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No hay columna 'usuario_id' en cuentas bancarias.")
                
                # --- Análisis avanzado de cuentas bancarias ---

                # 1. Histograma de cantidad de cuentas bancarias por usuario
                if not cuentas_filtradas.empty:
                    st.subheader("Cantidad de Cuentas Bancarias por Usuario")
                    try:
                        cuentas_por_usuario = cuentas_filtradas.groupby('usuario_id').size().reset_index(name='cantidad_cuentas')
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(cuentas_por_usuario['cantidad_cuentas'], bins=10, ax=ax, color='skyblue')
                        ax.set_title("Cantidad de Cuentas Bancarias por Usuario")
                        ax.set_xlabel("Cantidad de Cuentas")
                        ax.set_ylabel("Frecuencia")
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el histograma de cuentas por usuario: {e}")

                # 2. Cantidad total por tipo de cuenta (combinado usuarios y cuentas)
                if not cuentas_filtradas.empty and not usuarios_filtrados.empty and 'tipo_cuenta' in cuentas_filtradas.columns and 'tipo_cuenta' in usuarios_filtrados.columns:
                    st.subheader("Cantidad Total por Tipo de Cuenta (Combinado)")
                    try:
                        cuentas_cuenta = cuentas_filtradas[['usuario_id', 'tipo_cuenta']].copy()
                        cuentas_usuario = usuarios_filtrados[['id', 'tipo_cuenta']].copy()
                        cuentas_usuario = cuentas_usuario.rename(columns={'id': 'usuario_id'})
                        cuentas_cuenta['origen'] = 'cuentas'
                        cuentas_usuario['origen'] = 'usuarios'
                        df_tipos_cuenta = pd.concat([cuentas_cuenta, cuentas_usuario], ignore_index=True)
                        cantidad_cuentas = df_tipos_cuenta['tipo_cuenta'].value_counts().reset_index()
                        cantidad_cuentas.columns = ['tipo_cuenta', 'cantidad']
                        fig, ax = plt.subplots(figsize=(8, 5))
                        barplot = sns.barplot(data=cantidad_cuentas, x='tipo_cuenta', y='cantidad', palette='pastel', ax=ax)
                        for p in barplot.patches:
                            height = p.get_height()
                            barplot.text(p.get_x() + p.get_width() / 2., height + 0.3, f'{int(height)}', ha='center', va='bottom', fontsize=10)
                        ax.set_title('Cantidad Total por Tipo de Cuenta (Combinado)')
                        ax.set_xlabel('Tipo de Cuenta')
                        ax.set_ylabel('Cantidad')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el gráfico combinado de tipo de cuenta: {e}")

                # 3. Cantidad de cuentas bancarias por provincia y tipo de cuenta
                if not cuentas_filtradas.empty and not usuarios_filtrados.empty and 'provincia' in usuarios_filtrados.columns:
                    st.subheader("Cantidad de Cuentas Bancarias por Provincia y Tipo de Cuenta")
                    try:
                        cuentas_filtradas['usuario_id'] = cuentas_filtradas['usuario_id'].astype(str)
                        usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str)
                        cuentas_provincia = cuentas_filtradas.merge(
                            usuarios_filtrados[['id', 'provincia']], left_on='usuario_id', right_on='id', how='left'
                        )
                        cuentas_provincia_agg = cuentas_provincia.groupby(['provincia', 'tipo_cuenta']).size().reset_index(name='cantidad_cuentas')
                        fig, ax = plt.subplots(figsize=(14, 7))
                        sns.barplot(data=cuentas_provincia_agg, x='provincia', y='cantidad_cuentas', hue='tipo_cuenta', palette='viridis', ax=ax)
                        ax.set_title("Cantidad de Cuentas Bancarias por Provincia y Tipo de Cuenta")
                        ax.set_xlabel("Provincia")
                        ax.set_ylabel("Cantidad de Cuentas")
                        plt.xticks(rotation=45, ha='right')
                        for p in ax.patches:
                            height = p.get_height()
                            if height > 0:
                                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                                            ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
                        plt.legend(title='Tipo de Cuenta')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el gráfico por provincia y tipo de cuenta: {e}")

                # 4. Cantidad de cuentas bancarias por nacionalidad y tipo de cuenta (Top 30)
                if not cuentas_filtradas.empty and not usuarios_filtrados.empty and 'nacionalidad' in usuarios_filtrados.columns:
                    st.subheader("Cantidad de Cuentas Bancarias por Nacionalidad y Tipo de Cuenta (Top 30)")
                    try:
                        cuentas_filtradas['usuario_id'] = cuentas_filtradas['usuario_id'].astype(str)
                        usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str)
                        cuentas_nacionalidad = cuentas_filtradas.merge(
                            usuarios_filtrados[['id', 'nacionalidad']], left_on='usuario_id', right_on='id', how='left'
                        )
                        cuentas_nacionalidad = cuentas_nacionalidad.groupby(['nacionalidad', 'tipo_cuenta']).size().reset_index(name='cantidad_cuentas')
                        total_cuentas_por_nacionalidad = cuentas_nacionalidad.groupby('nacionalidad')['cantidad_cuentas'].sum().reset_index()
                        top30_nacionalidades = total_cuentas_por_nacionalidad.sort_values('cantidad_cuentas', ascending=False).head(30)['nacionalidad']
                        cuentas_nacionalidad_top30 = cuentas_nacionalidad[cuentas_nacionalidad['nacionalidad'].isin(top30_nacionalidades)]
                        fig, ax = plt.subplots(figsize=(16, 8))
                        sns.barplot(data=cuentas_nacionalidad_top30, x='nacionalidad', y='cantidad_cuentas', hue='tipo_cuenta', palette='Set2', ax=ax)
                        ax.set_title("Cantidad de Cuentas Bancarias por Nacionalidad y Tipo de Cuenta (Top 30)", fontsize=16)
                        ax.set_xlabel("Nacionalidad", fontsize=14)
                        ax.set_ylabel("Cantidad de Cuentas", fontsize=14)
                        plt.xticks(rotation=45, ha='right', fontsize=12)
                        plt.yticks(fontsize=12)
                        for p in ax.patches:
                            height = p.get_height()
                            if height > 0:
                                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                                            ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), textcoords='offset points')
                        plt.legend(title='Tipo de Cuenta', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el gráfico por nacionalidad y tipo de cuenta: {e}")

                # 5. Stacked barplot de cuentas bancarias por provincia y tipo de cuenta
                if not cuentas_filtradas.empty and not usuarios_filtrados.empty and 'provincia' in usuarios_filtrados.columns:
                    st.subheader("Cuentas Bancarias por Provincia y Tipo de Cuenta (Stacked)")
                    try:
                        cuentas_filtradas['usuario_id'] = cuentas_filtradas['usuario_id'].astype(str)
                        usuarios_filtrados['id'] = usuarios_filtrados['id'].astype(str)
                        cuentas_prov = cuentas_filtradas.merge(
                            usuarios_filtrados[['id', 'provincia']], left_on='usuario_id', right_on='id', how='left'
                        )
                        cuentas_bancarias_provincia = cuentas_prov.groupby(['provincia', 'tipo_cuenta']).size().unstack(fill_value=0)
                        cuentas_bancarias_provincia.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
                        plt.title("Cantidad de Cuentas Bancarias por Provincia y Tipo de Cuenta")
                        plt.xlabel("Provincia")
                        plt.ylabel("Cantidad de Cuentas")
                        plt.xticks(rotation=45, ha='right', fontsize=11)
                        plt.legend(title='Tipo de Cuenta')
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.info(f"No se pudo mostrar el stacked barplot de cuentas por provincia: {e}")

                st.markdown("### Tabla de cuentas bancarias filtradas")
                st.dataframe(cuentas_filtradas, use_container_width=True)

    # --- Monedas Digitales ---
    with eda_tabs[4]:
        if mostrar_monedas:
            if monedas_filtradas.empty:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")
            else:
                st.subheader("Monedas Digitales")
                if 'usuario_id' in monedas_filtradas.columns:
                    monedas_sum = monedas_filtradas['usuario_id'].value_counts().head(30).reset_index()
                    monedas_sum.columns = ['usuario_id', 'num_monedas']
                    monedas_sum['usuario_id'] = monedas_sum['usuario_id'].astype(str)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(
                        data=monedas_sum,
                        y='usuario_id',
                        x='num_monedas',
                        palette='viridis',
                        ax=ax
                    )
                    ax.set_title("Top 30 usuarios por número de monedas digitales", fontsize=16)
                    ax.set_xlabel("Nº monedas", fontsize=14)
                    ax.set_ylabel("Usuario ID", fontsize=14)
                    for i, v in enumerate(monedas_sum['num_monedas']):
                        ax.text(v, i, f"{v}", color='black', va='center', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No hay columna 'usuario_id' en monedas digitales.")
                
                # --- Análisis avanzado de monedas digitales ---

                # 1. Cantidad de monedas digitales por tipo
                if not monedas_filtradas.empty:
                    st.subheader("Cantidad de Monedas Digitales por Tipo")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.countplot(
                            data=monedas_filtradas,
                            x='tipo_moneda',
                            order=monedas_filtradas['tipo_moneda'].value_counts().index,
                            palette='viridis',
                            ax=ax
                        )
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height + 0.5),
                                        ha='center', va='bottom', fontsize=9)
                        ax.set_title("Cantidad de Monedas Digitales por Tipo")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el gráfico de monedas por tipo: {e}")

                # 2. Top 20 usuarios con más monedas digitales
                if not monedas_filtradas.empty:
                    st.subheader("Top 20: Cantidad de Monedas Digitales por Usuario")
                    try:
                        monedas_por_usuario = monedas_filtradas.groupby('usuario_id').size().reset_index(name='cantidad_monedas')
                        top_20_monedas = monedas_por_usuario.sort_values(by='cantidad_monedas', ascending=False).head(20)
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(data=top_20_monedas, x='usuario_id', y='cantidad_monedas', palette='viridis', ax=ax)
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height + 0.5),
                                        ha='center', va='bottom', fontsize=9)
                        ax.set_title("Top 20: Cantidad de Monedas Digitales por Usuario")
                        ax.set_xlabel("ID de Usuario")
                        ax.set_ylabel("Cantidad de Monedas")
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el top 20 de monedas por usuario: {e}")

                # 3. Distribución de valor total en monedas digitales
                '''if not monedas_filtradas.empty:
                    st.subheader("Distribución de Valor Total en Monedas Digitales")
                    try:
                        valor_cripto = monedas_filtradas.groupby('usuario_id')['valor_actual'].sum().reset_index(name='total_cripto')
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.histplot(valor_cripto['total_cripto'], bins=30, kde=True, ax=ax)
                        ax.set_title("Distribución de Valor Total en Monedas Digitales")
                        ax.set_xlabel("Valor Total (€)")
                        ax.set_ylabel("Frecuencia")
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar la distribución de valor total: {e}")'''

                # 4. Top 30 usuarios con más monedas digitales
                if not monedas_filtradas.empty:
                    st.subheader("Top 30 Usuarios con Más Monedas Digitales")
                    try:
                        monedas_por_usuario = monedas_filtradas.groupby('usuario_id').size().reset_index(name='cantidad_monedas')
                        top30_monedas = monedas_por_usuario.sort_values(by='cantidad_monedas', ascending=False).head(30)
                        fig, ax = plt.subplots(figsize=(14, 7))
                        sns.barplot(data=top30_monedas, x='usuario_id', y='cantidad_monedas', palette='viridis', ax=ax)
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                                        ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
                        ax.set_title("Top 30 Usuarios con Mayor Cantidad de Monedas Digitales")
                        ax.set_xlabel("ID de Usuario")
                        ax.set_ylabel("Cantidad de Monedas")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"No se pudo mostrar el top 30 de monedas por usuario: {e}")

                st.markdown("### Tabla de monedas digitales filtradas")
                st.dataframe(monedas_filtradas, use_container_width=True)

# ----------- Tab 2: Predicción Morosidad 6 meses -----------
with tabs[1]:
    st.header("Predicción de Morosos y Créditos (Próximos 6 meses)")

    st.markdown("##### Predicción de morosos nuevos y créditos nuevos por mes (Julio–Diciembre 2025)")

    # Datos de ejemplo (puedes reemplazar por los calculados en tu notebook)
    df_pred = pd.DataFrame({
        "Fecha": pd.date_range("2025-07-31", periods=6, freq="M"),
        "Morosos esperados": [16, 16, 16, 15, 15, 15],
        "Créditos esperados": [137, 137, 137, 137, 138, 138],
    })
    df_pred["Tasa de morosidad (%)"] = (df_pred["Morosos esperados"] / df_pred["Créditos esperados"] * 100).round(2)

    st.dataframe(df_pred, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred["Fecha"], y=df_pred["Morosos esperados"], mode='lines+markers', name="Morosos"))
    fig.add_trace(go.Scatter(x=df_pred["Fecha"], y=df_pred["Créditos esperados"], mode='lines+markers', name="Créditos"))
    fig.update_layout(title="Predicción de morosos y créditos nuevos", xaxis_title="Fecha", yaxis_title="Cantidad")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(df_pred, x="Fecha", y="Tasa de morosidad (%)", markers=True, title="Tasa de morosidad esperada (%)")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Conclusiones clave:**
    - La tasa de morosidad esperada se mantiene estable entre 10.8% y 11.8%.
    - No se esperan picos preocupantes a corto plazo.
    - La predicción sugiere un comportamiento controlado del riesgo crediticio.
    """)

# ----------- Tab 3: Modelo de Morosidad -----------
with tabs[2]:
    st.header("Modelo de Predicción de Morosidad")

    st.markdown("""
    Se entrenó un modelo Random Forest para predecir la variable `es_moroso` usando características como edad, salario, provincia, estado civil, profesión, cantidad de créditos y valor de activos.
    """)

    st.markdown("##### Ejemplo de métricas del modelo (puedes reemplazar por tus resultados reales):")
    st.code("""
Accuracy promedio (CV): 0.89
Mejor F1 Score (CV): 0.90

Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     70560
           1       0.99      0.99      0.99    102850

    accuracy                           0.99    173410
   macro avg       0.99      0.99      0.99    173410
weighted avg       0.99      0.99      0.99    173410
    """, language="text")

    st.markdown("##### Importancia de variables (ejemplo):")
    st.bar_chart(pd.Series({
        "salario": 0.25,
        "edad": 0.18,
        "cantidad_creditos": 0.16,
        "valor_activos": 0.15,
        "provincia": 0.10,
        "estado_civil": 0.09,
        "profesion": 0.07
    }))

    # --- ¿Los morosos tienen más créditos? ---
    if not usuarios_filtrados.empty and not creditos_filtrados.empty and 'es_moroso' in usuarios_filtrados.columns:
        st.subheader("¿Los morosos tienen más créditos?")
        try:
            usuarios_mora = usuarios_filtrados[['id', 'es_moroso']].rename(columns={'id': 'usuario_id'})
            creditos_mora = creditos_filtrados.merge(usuarios_mora, on='usuario_id')
            creditos_mora = creditos_mora.groupby('es_moroso').size().reset_index(name='cantidad_creditos')
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(data=creditos_mora, x='es_moroso', y='cantidad_creditos', palette='flare', ax=ax)
            ax.set_title("Cantidad de Créditos según si el Usuario es Moroso")
            ax.set_xlabel("Moroso (0 = No, 1 = Sí)")
            ax.set_ylabel("Cantidad de Créditos")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"No se pudo mostrar el gráfico de morosos y créditos: {e}")

    # --- Morosidad según Edad ---
    if not usuarios_filtrados.empty and 'edad' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
        st.subheader("Tasa de Morosidad por Rango de Edad")
        try:
            bins = [18, 25, 35, 45, 55, 65, 100]
            labels = ['18–25', '26–35', '36–45', '46–55', '56–65', '65+']
            usuarios_filtrados['rango_edad'] = pd.cut(usuarios_filtrados['edad'], bins=bins, labels=labels)
            mora_por_edad = usuarios_filtrados.groupby('rango_edad')['es_moroso'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=mora_por_edad, x='rango_edad', y='es_moroso', palette='magma', ax=ax)
            ax.set_title("Tasa de Morosidad por Rango de Edad")
            ax.set_ylabel("Proporción de Morosos")
            ax.set_xlabel("Rango de Edad")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"No se pudo mostrar la morosidad por edad: {e}")

    # --- Morosidad según Provincia ---
    if not usuarios_filtrados.empty and 'provincia' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
        st.subheader("Tasa de Morosidad por Provincia")
        try:
            mora_provincia = usuarios_filtrados.groupby('provincia')['es_moroso'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(18, 6))
            sns.barplot(data=mora_provincia.sort_values('es_moroso', ascending=False), x='provincia', y='es_moroso', palette='rocket', ax=ax)
            plt.xticks(rotation=45, ha='right')
            ax.set_title("Tasa de Morosidad por Provincia")
            ax.set_ylabel("Proporción de Morosos")
            ax.set_xlabel("Provincia")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"No se pudo mostrar la morosidad por provincia: {e}")

    # --- Morosidad según Nacionalidad (Top 20) ---
    if not usuarios_filtrados.empty and 'nacionalidad' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
        st.subheader("Tasa de Morosidad por Nacionalidad (Top 20)")
        try:
            mora_nacionalidad = usuarios_filtrados.groupby('nacionalidad')['es_moroso'].mean().reset_index()
            top_nacionalidad = mora_nacionalidad.sort_values('es_moroso', ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(18, 6))
            sns.barplot(data=top_nacionalidad, x='nacionalidad', y='es_moroso', palette='flare', ax=ax)
            plt.xticks(rotation=45, ha='right')
            ax.set_title("Tasa de Morosidad por Nacionalidad (Top 20)")
            ax.set_ylabel("Proporción de Morosos")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"No se pudo mostrar la morosidad por nacionalidad: {e}")

    # --- Morosidad según Estado Civil ---
    if not usuarios_filtrados.empty and 'estado_civil' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
        st.subheader("Tasa de Morosidad por Estado Civil")
        try:
            mora_estado_civil = usuarios_filtrados.groupby('estado_civil')['es_moroso'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(18, 6))
            sns.barplot(data=mora_estado_civil.sort_values('es_moroso', ascending=False), x='estado_civil', y='es_moroso', palette='coolwarm', ax=ax)
            plt.xticks(rotation=45, ha='right')
            ax.set_title("Tasa de Morosidad por Estado Civil")
            ax.set_ylabel("Proporción de Morosos")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"No se pudo mostrar la morosidad por estado civil: {e}")

    # --- Morosidad según salario ---
    if not usuarios_filtrados.empty and 'salario' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
        st.subheader("Distribución de salario según Morosidad")
        try:
            df_usuarios_clean = usuarios_filtrados.dropna(subset=['salario', 'es_moroso'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df_usuarios_clean, x='es_moroso', y='salario', palette='Set3', ax=ax)
            ax.set_title("Distribución de salario según Morosidad")
            ax.set_xlabel("")
            ax.set_ylabel("Salario")
            ax.set_xticklabels(['No Moroso', 'Moroso'])
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"No se pudo mostrar la distribución de salario según morosidad: {e}")

    # --- Morosidad por Tipo de Empleo (Top 20 burbuja) ---
    if not usuarios_filtrados.empty and 'profesion' in usuarios_filtrados.columns and 'es_moroso' in usuarios_filtrados.columns:
        st.subheader("Top 20 Tasa de Morosidad por Tipo de Profesión")
        try:
            df_usuarios_clean = usuarios_filtrados.dropna(subset=['profesion', 'es_moroso'])
            mora_empleo = df_usuarios_clean.groupby('profesion').agg(
                tasa_morosidad=('es_moroso', 'mean'),
                cantidad_usuarios=('profesion', 'size')
            ).reset_index()
            top20_mora = mora_empleo.sort_values('tasa_morosidad', ascending=False).head(20)
            fig = px.scatter(
                top20_mora,
                x='profesion',
                y='tasa_morosidad',
                size='cantidad_usuarios',
                color='tasa_morosidad',
                color_continuous_scale='Viridis',
                hover_name='profesion',
                hover_data={
                    'tasa_morosidad': ':.2f',
                    'cantidad_usuarios': True,
                    'profesion': False
                },
                title='Top 20 Tasa de Morosidad por Tipo de Profesión',
                labels={'tasa_morosidad': 'Tasa de Morosidad', 'profesion': 'Profesión'}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                yaxis_range=[0, 1],
                xaxis={'categoryorder':'total descending'},
                height=600,
                margin=dict(t=60, b=150)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"No se pudo mostrar la morosidad por profesión: {e}")

    st.markdown("""
    **Notas:**
    - El modelo muestra alta precisión y generaliza bien.
    - Las variables más influyentes son salario, edad y cantidad de créditos.
    - Se recomienda monitorear el modelo y actualizarlo con nuevos datos.
    """)

# ----------- Tab 4: Asistente RAG -----------
with tabs[3]:
    st.header("🧠 Asistente Económico-Financiero Inteligente")
    st.markdown("""
    Sube documentos PDF y haz preguntas sobre economía, inversión, créditos o morosidad. 
    El sistema analizará tus archivos y responderá de forma precisa.
    """)
    
    # Inicializar el estado de la sesión para el historial
    if 'rag_history' not in st.session_state:
        st.session_state.rag_history = []
    
    # Layout en columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input para la pregunta
        user_question = st.text_area(
            "Tu Pregunta", 
            placeholder="Ejemplo: ¿Qué impacto tiene la morosidad en la rentabilidad bancaria?",
            height=100
        )
    
    with col2:
        # Subida de archivos PDF
        uploaded_files = st.file_uploader(
            "Sube documentos PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="Puedes subir múltiples archivos PDF para análisis"
        )
        
        # Mostrar archivos subidos
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} archivo(s) subido(s)")
            for file in uploaded_files:
                st.write(f"📄 {file.name}")
    
    # Botones de acción
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        ask_button = st.button("🔍 Preguntar", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_history = st.button("🗑️ Limpiar Historial", use_container_width=True)
    
    # Limpiar historial si se solicita
    if clear_history:
        st.session_state.rag_history = []
        st.success("Historial limpiado")
    
    # Procesar pregunta
    # Procesar pregunta
    if ask_button:
        if not user_question.strip():
            st.warning("⚠️ Por favor, escribe una pregunta.")
        else:
            with st.spinner("🔄 Analizando documentos y generando respuesta..."):
                try:
                    # Inicializar el sistema RAG según si hay o no archivos
                    if uploaded_files:
                        rag_system = create_rag_from_files(uploaded_files)
                    else:
                        # Crear un sistema simple con conocimiento general
                        rag_system = SimpleRAGSystem()
                        rag_system.load_documents()  # Carga conocimientos generales como edad, morosidad, estado civil
                        rag_system.create_vectorstore()
                        rag_system.setup_chain()

                    if rag_system:
                        # Reformular la pregunta si es necesario
                        refined_question = rag_system.rewrite_question(user_question)

                        # Obtener respuesta del sistema
                        answer = rag_system.answer_question(refined_question)

                        # Construcción de la respuesta completa
                        if refined_question != user_question:
                            full_answer = f"**Tu pregunta fue reformulada para mayor claridad:**\n> {refined_question}\n\n---\n\n{answer}"
                        else:
                            full_answer = answer

                        # Agregar al historial de conversación
                        st.session_state.rag_history.append({
                            "question": user_question,
                            "answer": full_answer,
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "has_files": bool(uploaded_files)
                        })

                        st.success("✅ Respuesta generada correctamente")
                    else:
                        st.error("❌ Error al inicializar el sistema RAG")

                except Exception as e:
                    st.error(f"❌ Error al procesar la consulta: {str(e)}")

    
    # Mostrar historial de conversación
    if st.session_state.rag_history:
        st.markdown("---")
        st.subheader("💬 Historial de Conversación")
        
        # Mostrar cada entrada del historial (más reciente primero)
        for i, entry in enumerate(reversed(st.session_state.rag_history)):
            file_indicator = "📄" if entry.get('has_files', False) else "💭"
            with st.expander(f"{file_indicator} {entry['timestamp']} - {entry['question'][:50]}{'...' if len(entry['question']) > 50 else ''}", expanded=(i == 0)):
                st.markdown(f"**Pregunta:** {entry['question']}")
                st.markdown("**Respuesta:**")
                st.markdown(entry['answer'])
                if entry.get('has_files', False):
                    st.caption("🔍 Basado en documentos subidos")
                else:
                    st.caption("💡 Basado en conocimiento general")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre el Asistente RAG"):
        st.markdown("""
        **¿Cómo funciona el Asistente RAG?**
        
        1. **Subida de documentos (Opcional)**: Carga archivos PDF con información financiera, económica o bancaria.
        
        2. **Procesamiento inteligente**: El sistema analiza y vectoriza el contenido de tus documentos.
        
        3. **Búsqueda semántica**: Encuentra la información más relevante para responder tu pregunta.
        
        4. **Respuesta contextual**: Genera una respuesta basada en el contenido específico de tus documentos o conocimiento general.
        
        **Tipos de consultas recomendadas:**
        - Análisis de riesgo crediticio
        - Indicadores de morosidad  
        - Estrategias de inversión
        - Evaluación financiera
        - Políticas bancarias
        - Regulaciones económicas
        - Modelos predictivos
        - Gestión de carteras
        
        **📌 Funciona sin archivos**: Puedes hacer preguntas generales sobre temas bancarios y financieros sin necesidad de subir documentos.
        
        **Nota**: Esta es una versión de demostración. En producción, integrarías un modelo de lenguaje más avanzado como GPT-4, Claude, o Llama.
        """)

# ----------- Tab 5: Conclusiones -----------
with tabs[4]:
    st.header("Conclusiones Generales del Análisis Bancario")

    st.markdown("""
- La tasa de morosidad esperada ronda el 11% para los próximos 6 meses.
- El modelo de predicción de morosidad es robusto y útil para la gestión de riesgo.
- Se recomienda refinar los modelos de scoring, automatizar alertas tempranas y considerar modelos avanzados de predicción temporal.
- La entidad debe provisionar entre **75.000 € y 80.000 € mensuales** por pérdidas esperadas de nuevos créditos morosos.
- Es fundamental monitorear periódicamente el desempeño del modelo y actualizarlo con nuevos datos.
    """)


# ----------- Descargable -----------
with st.expander("Descargar datos de usuarios filtrados"):
    st.dataframe(usuarios_filtrados, use_container_width=True)
    csv = usuarios_filtrados.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Descargar usuarios filtrados (CSV)",
        data=csv,
        file_name="usuarios_filtrados.csv",
        mime="text/csv",
    )