# Prototipo Multiagente - Proyectos Inmobiliarios (Colombia)

Aplicación en **Streamlit** para evaluar la **pre-factibilidad** y monitorear el **ciclo de vida de obras inmobiliarias** en Colombia. Combina reglas de dominio, modelado financiero y LLMs (Mistral por defecto) para acelerar decisiones C-level usando datos dummy de referencia.

Esta es la url del proyecto en streamlit cloud https://proyectos-inmobiliarios.streamlit.app/
---

## 1. Descripcion del proyecto

El prototipo integra tres grandes bloques:

- **Pre-factibilidad**: evaluacion normativa, financiera y de riesgo de un proyecto inmobiliario.
- **Monitor de obra**: seguimiento de hitos versus una linea base, generacion de alertas y estados de ciclo de vida.
- **Chat del proyecto**: asistente conversacional con contexto completo de pre-factibilidad y obra.

Ademas incluye dos tabs de documentacion interna:

- **Info del Proyecto**: que se construyo, como, por que Mistral, frameworks y relaciones de archivos.
- **Arquitectura**: diagramas Graphviz de relaciones entre archivos y flujo macro de datos.

Todos los datos son **dummy** y tienen fines demostrativos. No deben usarse como normativa, financieros o de mercado reales.

---

## 2. Funcionalidades principales

### 2.1 Pre-factibilidad

- Entrada de variables clave: ciudad, uso de suelo, area del lote, pisos, unidades, tamano promedio y costo del lote.
- Evaluacion **normativa** (pisos maximos, FAR, ocupacion, altura) con reglas dummy.
- Evaluacion **financiera**: VAN (NPV), TIR anual, margen de utilidad, ingresos y costos.
- Reporte ejecutivo generado con LLM (Mistral).
- Asesor de diseno preliminar.
- Checklist de viabilidad generado con LLM (Mistral).
- "Hazlo factible": ajuste heuristico de unidades y pisos para cumplir normativa y mejorar margen.
- Recomendacion de mix de unidades.
- Analisis de sensibilidad (±10%) y multivariable.
- Simulacion Monte Carlo.
- Comparacion y guardado de escenarios.
- Estructuracion de deuda con cronograma de pagos.
- Impacto de impuestos y costos de transaccion.
- Exportacion a PDF y Excel.

### 2.2 Monitor de obra

- Carga de cronograma base y eventos en obra.
- Computo de avance por hito y estado de semaforo.
- Estado del ciclo de vida del proyecto.
- Generacion de linea base sugerida.
- Reporte de alertas generado con LLM (Mistral).
- Exportacion de hitos a CSV.

### 2.3 Chat del proyecto

- Contexto enriquecido con datos de pre-factibilidad y monitor de obra.
- Roles (C-level, Tecnico, Legal, Marketing, etc.).
- Tono (Ejecutivo, Casual, Formal).
- Chain-of-thought y seleccion de idioma.
- Descarga y limpieza de historial.

---

## 3. Arquitectura

```text
Usuario
  |
  v
Streamlit (app.py)  ---->  UI, session_state, downloads
  |
  +--> Agentes (src/agents/)
  |      +-- prefactibility.py  (evaluacion, reporte, diseno, checklist)
  |      +-- construction.py    (monitor + alertas)
  |      +-- chat.py            (asistente conversacional)
  |
  +--> Dominio (src/domain/)
  |      +-- finance.py         (VAN, TIR, flujo de caja, deuda, impuestos)
  |      +-- normative.py       (reglas normativas)
  |      +-- construction_monitor.py (avance y riesgo)
  |
  +--> Datos (src/data/)
  |      +-- loaders.py         (carga y cache de CSV)
  |      +-- generate.py        (generacion de datos dummy)
  |
  +--> LLM (src/llm/)
         +-- client.py          (MultiProviderLLM sobre openai SDK)
         +-- providers.py       (configs de Mistral, OpenAI, etc.)
```

No se usa **LangChain** ni base de datos vectorial. Los LLMs se invocan directamente a traves de un wrapper propio para mantener el control de prompts y reducir dependencias. El RAG actual se implementa inyectando texto plano (CSVs, resultados y reportes) en el contexto del chat.

---

## 4. Estructura de archivos

```
.
├── app.py                          # Aplicacion Streamlit principal
├── requirements.txt                # Dependencias
├── data/
│   ├── normative_rules.csv         # Reglas normativas dummy
│   ├── market_assumptions.csv      # Supuestos de mercado dummy
│   ├── baseline_schedule.csv       # Cronograma base dummy
│   └── site_events.csv             # Eventos de obra dummy
├── src/
│   ├── agents/
│   │   ├── chat.py                 # Agente conversacional
│   │   ├── construction.py         # Agente de monitor de obra
│   │   └── prefactibility.py       # Agente de pre-factibilidad
│   ├── data/
│   │   ├── generate.py             # Generacion de datos dummy
│   │   └── loaders.py              # Carga de datos
│   ├── domain/
│   │   ├── construction_monitor.py # Logica de avance de obra
│   │   ├── finance.py              # Modelado financiero
│   │   └── normative.py            # Logica normativa
│   └── llm/
│       ├── client.py               # Cliente multi-proveedor
│       └── providers.py            # Configuracion de proveedores
└── scripts/                        # Scripts de generacion de datos
```

---

## 5. Tecnologias y librerias

- **Streamlit**: UI rapida y session state.
- **pandas / numpy / scipy**: manipulacion y calculos numericos.
- **plotly**: visualizaciones interactivas.
- **openai SDK**: llamadas a modelos compatibles con OpenAI (Mistral usa el mismo formato).
- **fpdf2**: generacion de reportes PDF.
- **openpyxl**: exportacion a Excel.
- **Graphviz**: diagramas de arquitectura (renderizados con `st.graphviz_chart`).

---

## 6. Configuracion y ejecucion local

### 6.1 Requisitos

- Python 3.10+
- pip

### 6.2 Instalacion

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 6.3 Variables de entorno

Crea un archivo `.streamlit/secrets.toml` (local) o variables de entorno con las API keys:

```toml
MISTRAL_API_KEY = "tu_api_key_de_mistral"
OPENAI_API_KEY = "tu_api_key_de_openai"  # opcional
```

Actualmente el proyecto fuerza el uso de **Mistral** para reporte ejecutivo, checklist, asesor de diseno y alertas porque es la unica API key configurada por defecto. Si se agregan otras keys, `MultiProviderLLM` las usara automaticamente.

### 6.4 Ejecutar localmente

```bash
streamlit run app.py
```

---

## 7. Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub.
2. Conecta el repositorio en Streamlit Cloud.
3. Configura `MISTRAL_API_KEY` en **Settings > Secrets**.
4. Haz clic en **Reboot** cada vez que subas cambios al repositorio.

---

## 8. Limitaciones conocidas

- Los datos son **dummy** y no representan normativa, mercado ni costos reales.
- No hay persistencia de base de datos: todo vive en `st.session_state`.
- No hay autenticacion ni multiusuario.
- No se usan embeddings ni vector DB: el RAG es por inyeccion de texto plano.
- No se integran fuentes externas de normativa o comparables reales.
- Los modelos de prediccion de retrasos y carga de MS Project/Primavera no estan implementados.

---

## 9. Roadmap sugerido

1. **RAG y vectorizacion**: ChromaDB/FAISS + embeddings para normativa, comparables y documentos.
2. **Agentes con herramientas**: LangChain o LlamaIndex para orquestar agentes con memoria y herramientas.
3. **Datos reales**: conectores a planeacion urbana, catastro y bases de comparables.
4. **Persistencia**: base de datos (SQLite/PostgreSQL) y autenticacion.
5. **Monitor avanzado**: carga de cronogramas MS Project/Primavera y prediccion de retrasos con ML.
6. **Testing**: pytest, CI/CD y despliegue automatizado.

---

## 10. Autor

Prototipo desarrollado como demostracion de una aplicacion multiagente para proyectos inmobiliarios en Colombia.
