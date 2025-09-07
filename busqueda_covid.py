
# app_covid_search_v2.py  (versión parcheada)
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
import unicodedata, re

APP_NAME = "Consultas COVID"
st.set_page_config(page_title=APP_NAME, layout="wide")
CKAN_ROOT = "https://datos.gob.cl/api/3/action"

def _clean_columns(cols):
    """Normalize column names: remove accents, symbols; snake_case lowercase."""
    def fix(x):
        s = str(x)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^\w]+", "_", s)       # replace non-word chars by underscore
        s = re.sub(r"_+", "_", s)           # collapse multiple underscores
        return s.strip("_").lower()
    return [fix(c) for c in cols]

@st.cache_data(ttl=1800, show_spinner=False)
def search_resources(query: str = "covid", rows: int = 50) -> pd.DataFrame:
    resp = requests.get(f"{CKAN_ROOT}/package_search", params={"q": query, "rows": rows}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("result", {}).get("results", [])
    items = []
    for pkg in results:
        org = (pkg.get("organization") or {}).get("title") or (pkg.get("organization") or {}).get("name")
        for res in pkg.get("resources", []):
            if res.get("datastore_active") is True:
                items.append({
                    "dataset_title": pkg.get("title"),
                    "resource_name": res.get("name"),
                    "resource_id": res.get("id"),
                    "format": res.get("format"),
                    "org": org,
                    "dataset_id": pkg.get("id"),
                })
    df = pd.DataFrame(items)
    if not df.empty:
        df.columns = _clean_columns(df.columns)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_datastore(resource_id: str, limit: int = 5000) -> pd.DataFrame:
    params = {"resource_id": resource_id, "limit": limit}
    resp = requests.get(f"{CKAN_ROOT}/datastore_search", params=params, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    if not j.get("success"):
        raise RuntimeError("Respuesta CKAN indica success=False")
    recs = j.get("result", {}).get("records", [])
    df = pd.DataFrame(recs)

    # Clean columns
    df.columns = _clean_columns(df.columns)

    # Parse dates
    for c in df.columns:
        if "fecha" in c and df[c].dtype == "object":
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Convert numeric-like strings (including commas) to numbers when posible
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="ignore")
            except Exception:
                pass
    return df

def auto_date_col(df: pd.DataFrame):
    for c in df.columns:
        if "fecha" in c and pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def main():
    st.title(APP_NAME)
    st.caption("Explora recursos COVID en datos.gob.cl (CKAN) y consulta el DataStore")

    with st.sidebar:
        st.header("Búsqueda")
        q = st.text_input("Palabra clave", value="covid")
        rows = st.slider("Conjuntos a revisar", 10, 100, 50, step=10)
        st.write("Pulsa el botón para ejecutar la búsqueda (cache 30 min).")
        do_search = st.button("Buscar recursos")

    if 'searched' not in st.session_state:
        st.session_state['searched'] = True
    if do_search:
        st.session_state['searched'] = True

    results = search_resources(query=q, rows=rows) if st.session_state['searched'] else pd.DataFrame()
    if results.empty:
        st.warning("No se encontraron recursos con DataStore activo.")
        return

    st.subheader("Recursos encontrados (DataStore activo)")
    st.dataframe(results, use_container_width=True, height=250)

    key_map = {f'{r["dataset_title"]} — {r["resource_name"]} ({r["format"]}) [{r["org"]}]': r["resource_id"]
               for _, r in results.iterrows()}
    choice = st.selectbox("Selecciona un recurso para cargar", options=list(key_map.keys()))
    resource_id = key_map[choice]

    with st.sidebar:
        st.header("Carga de datos")
        limit = st.slider("Filas a descargar (limit)", 500, 10000, 5000, step=500)

    df = fetch_datastore(resource_id, limit=limit)
    if df.empty:
        st.warning("No hay datos para este recurso.")
        return

    st.markdown(f"**resource_id seleccionado:** `{resource_id}`")
    st.dataframe(df.head(200), use_container_width=True, height=300)

    date_col = auto_date_col(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", len(df)); c2.metric("Columnas", df.shape[1]); c3.metric("Columna de fecha", date_col or "—")

    all_cols = list(df.columns)
    numeric_candidates = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

    with st.sidebar:
        st.header("Análisis")
        num_cols = st.multiselect("Columnas numéricas", numeric_candidates)
        cat_cols = st.multiselect("Columnas categóricas", [c for c in all_cols if df[c].dtype == "object"])

    # Filtro por fecha (si existe)
    if date_col:
        with st.sidebar:
            min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
            rango = st.date_input("Rango de fechas", (min_d, max_d), min_value=min_d, max_value=max_d)
        df = df[(df[date_col] >= pd.to_datetime(rango[0])) & (df[date_col] <= pd.to_datetime(rango[1]) + pd.Timedelta(days=1))]

    # ---- Promedios por grupo (seguro) ----
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
            index_cols = cat_cols if len(cat_cols) > 1 else cat_cols[0]
            chart_df = agg_display.set_index(index_cols)[metric]
            st.bar_chart(chart_df)

    # ---- Estadísticas simples ----
    elif num_cols:
        st.subheader("Estadísticas descriptivas")
        st.write(df[num_cols].describe().T)
    else:
        st.info("Selecciona columnas numéricas y/o categóricas para analizar.")

    # ---- Serie temporal ----
    if date_col and numeric_candidates:
        st.subheader("Serie temporal")
        metric_ts = st.selectbox("Métrica (serie temporal)", numeric_candidates, index=0)
        ts = df[[date_col, metric_ts]].dropna().set_index(date_col).sort_index()
        if not ts.empty:
            st.line_chart(ts[metric_ts])

    # ---- Descarga ----
    st.download_button(
        "Descargar CSV (datos filtrados)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"consultas_covid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
