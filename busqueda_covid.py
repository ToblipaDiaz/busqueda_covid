# app_consultas_covid.py
# Streamlit app: Consultas COVID (CKAN datos.gob.cl)
# - Limpieza robusta de columnas (sin acentos / símbolos, snake_case)
# - Reintentos HTTP con backoff
# - Detección/parseo de fechas
# - Gráficos seguros (aplanan groupby y limitan Top N)
# - Descarga CSV y (opcional) Parquet

import io
import re
import unicodedata
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------
# Configuración general
# ----------------------------
APP_NAME = "Consultas COVID"
CKAN_ROOT = "https://datos.gob.cl/api/3/action"
st.set_page_config(page_title=APP_NAME, layout="wide")

# Sesión HTTP con reintentos (tolerante a 429/5xx)
SESSION = requests.Session()
RETRIES = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
)
SESSION.mount("https://", HTTPAdapter(max_retries=RETRIES))
DEFAULT_TIMEOUT = (5, 30)  # (connect, read)


# ----------------------------
# Utilidades
# ----------------------------
def _clean_columns(cols):
    """Normaliza nombres: sin acentos/símbolos, snake_case en minúsculas."""
    def fix(x):
        s = str(x)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^\w]+", "_", s)   # todo lo que no es [A-Za-z0-9_] -> _
        s = re.sub(r"_+", "_", s)       # colapsa ___ a _
        return s.strip("_").lower()
    return [fix(c) for c in cols]


def auto_date_col(df: pd.DataFrame):
    """Detecta/convierte una columna de fecha si es posible y la retorna."""
    hints = ("fecha", "date", "datetime", "timestamp", "fch")
    # ya en datetime
    for c in df.columns:
        if any(h in c for h in hints) and pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # intenta parsear object
    for c in df.columns:
        if any(h in c for h in hints) and df[c].dtype == "object":
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > len(parsed) * 0.5:
                df[c] = parsed
                return c
    return None


# ----------------------------
# Acceso a CKAN
# ----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def search_resources(query: str = "covid", rows: int = 50) -> pd.DataFrame:
    """Busca datasets y retorna recursos con DataStore activo."""
    try:
        r = SESSION.get(
            f"{CKAN_ROOT}/package_search",
            params={"q": query, "rows": rows},
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Error HTTP al buscar recursos: {e}")
    j = r.json()
    results = j.get("result", {}).get("results", [])
    items = []
    for pkg in results:
        org = (pkg.get("organization") or {}).get("title") or (pkg.get("organization") or {}).get("name")
        for res in pkg.get("resources", []):
            if res.get("datastore_active") is True:
                items.append(
                    {
                        "dataset_title": pkg.get("title"),
                        "resource_name": res.get("name"),
                        "resource_id": res.get("id"),
                        "format": res.get("format"),
                        "org": org,
                        "dataset_id": pkg.get("id"),
                    }
                )
    df = pd.DataFrame(items)
    if not df.empty:
        df.columns = _clean_columns(df.columns)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_datastore(resource_id: str, limit: int = 5000) -> pd.DataFrame:
    """Descarga registros desde el DataStore para un resource_id."""
    try:
        r = SESSION.get(
            f"{CKAN_ROOT}/datastore_search",
            params={"resource_id": resource_id, "limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Error HTTP al consultar DataStore: {e}")
    j = r.json()
    if not j.get("success"):
        raise RuntimeError("La API de CKAN respondió success=False")
    recs = j.get("result", {}).get("records", [])
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    # Limpieza de columnas
    df.columns = _clean_columns(df.columns)

    # Detección/parseo de fechas
    _ = auto_date_col(df)  # puede convertir in-place

    # Convertir números de texto (coma o punto decimal)
    for c in df.columns:
        if df[c].dtype == "object":
            ser = df[c].astype(str)
            if ser.str.contains(r"\d", regex=True).any():
                df[c] = pd.to_numeric(ser.str.replace(",", ".", regex=False), errors="ignore")
    return df


# ----------------------------
# UI principal
# ----------------------------
def main():
    st.title(APP_NAME)
    st.caption("Explora recursos COVID en datos.gob.cl (CKAN) y consulta el DataStore")

    # --- Búsqueda ---
    with st.sidebar:
        st.header("Búsqueda")
        q = st.text_input("Palabra clave", value="covid")
        rows = st.slider("Conjuntos a revisar", 10, 100, 50, step=10)
        st.write("Pulsa el botón para ejecutar la búsqueda (cache 30 min).")
        do_search = st.button("Buscar recursos")

    if "searched" not in st.session_state:
        st.session_state["searched"] = False
    if do_search:
        st.session_state["searched"] = True

    if not st.session_state["searched"]:
        st.info("Ingresa una palabra clave y pulsa **Buscar recursos**.")
        st.stop()

    with st.spinner("Buscando recursos en datos.gob.cl…"):
        results = search_resources(query=q, rows=rows)

    if results.empty:
        st.warning("No se encontraron recursos con DataStore activo.")
        st.stop()

    st.subheader("Recursos encontrados (DataStore activo)")
    st.dataframe(results, use_container_width=True, height=260)

    # Selección de recurso
    key_map = {
        f'{r["dataset_title"]} — {r["resource_name"]} ({r["format"]}) [{r["org"]}]': r["resource_id"]
        for _, r in results.iterrows()
    }
    choice = st.selectbox("Selecciona un recurso para cargar", options=list(key_map.keys()))
    resource_id = key_map[choice]

    with st.sidebar:
        st.header("Carga de datos")
        limit = st.slider("Filas a descargar (limit)", 500, 10000, 5000, step=500)

    with st.spinner("Descargando y preparando datos…"):
        try:
            df = fetch_datastore(resource_id, limit=limit)
        except Exception as e:
            st.error(f"⚠️ No fue posible obtener datos del recurso seleccionado.\n\n**Detalle:** {e}")
            st.stop()

    if df.empty:
        st.warning("No se recibieron registros para este recurso con los parámetros dados.")
        st.stop()

    st.markdown(f"**resource_id seleccionado:** `{resource_id}`")
    st.dataframe(df.head(200), use_container_width=True, height=320)

    # Métricas
    date_col = auto_date_col(df)  # por si cambió con head()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", len(df))
    c2.metric("Columnas", df.shape[1])
    c3.metric("Columna de fecha", date_col or "—")

    all_cols = list(df.columns)
    numeric_candidates = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

    # --- Panel de análisis ---
    with st.sidebar:
        st.header("Análisis")
        num_cols = st.multiselect("Columnas numéricas", numeric_candidates)
        cat_cols = st.multiselect("Columnas categóricas", [c for c in all_cols if df[c].dtype == "object"])

    # Filtro por fecha
    if date_col and df[date_col].dropna().size > 0:
        valid_dates = df[date_col].dropna()
        with st.sidebar:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            rango = st.date_input(
                "Rango de fechas",
                (min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )
        mask = (df[date_col] >= pd.to_datetime(rango[0])) & (
            df[date_col] <= pd.to_datetime(rango[1]) + pd.Timedelta(days=1)
        )
        df = df[mask]

    # Promedios por grupo + chart seguro
    if num_cols and cat_cols:
        st.subheader("Promedios por grupo")
        valid_nums = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not valid_nums:
            st.warning("No se seleccionaron columnas numéricas válidas.")
        else:
            agg = df.groupby(cat_cols, dropna=False)[valid_nums].mean(numeric_only=True)
            agg_display = agg.reset_index()
            agg_display.columns = [str(c) for c in agg_display.columns]
            st.dataframe(agg_display, use_container_width=True)

            metric = st.selectbox("Métrica para gráfico de barras", valid_nums, index=0)
            top_n = st.slider("Top N categorías para el gráfico", 5, 50, 20, step=5)
            plot_df = agg_display.sort_values(metric, ascending=False).head(top_n)
            index_cols = cat_cols if len(cat_cols) > 1 else cat_cols[0]
            chart_df = plot_df.set_index(index_cols)[metric]
            st.bar_chart(chart_df)

    elif num_cols:
        st.subheader("Estadísticas descriptivas")
        st.write(df[num_cols].describe().T)
    else:
        st.info("Selecciona columnas numéricas y/o categóricas para analizar.")

    # Serie temporal (si procede)
    if date_col and numeric_candidates:
        st.subheader("Serie temporal")
        metric_ts = st.selectbox("Métrica (serie temporal)", numeric_candidates, index=0)
        ts = df[[date_col, metric_ts]].dropna().set_index(date_col).sort_index()
        if not ts.empty:
            st.line_chart(ts[metric_ts])

    # Descargas
    st.download_button(
        "Descargar CSV (datos filtrados)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"consultas_covid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # Parquet (opcional)
    try:
        parquet_buf = io.BytesIO()
        df.to_parquet(parquet_buf, index=False)
        st.download_button(
            "Descargar Parquet (compacto)",
            data=parquet_buf.getvalue(),
            file_name=f"consultas_covid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
            mime="application/octet-stream",
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
