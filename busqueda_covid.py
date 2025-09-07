# busqueda_covid.py
# =========================================
# SOLEMNE II - Taller de Programación II
# # =========================================
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="COVID-19 Chile - Explorador CKAN", layout="wide")

CKAN_ROOT = "https://datos.gob.cl/api/3/action"

@st.cache_data(ttl=1800, show_spinner=False)
def search_resources(query: str = "covid", rows: int = 50) -> pd.DataFrame:
    """Busca datasets en CKAN por palabra clave y devuelve DataFrame
    de recursos cuyo DataStore esté activo (datastore_active=True)."""
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
    return pd.DataFrame(items)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_datastore(resource_id: str, limit: int = 5000) -> pd.DataFrame:
    """Descarga datos de DataStore para un resource_id dado."""
    params = {"resource_id": resource_id, "limit": limit}
    resp = requests.get(f"{CKAN_ROOT}/datastore_search", params=params, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    if not j.get("success"):
        raise RuntimeError("Respuesta CKAN indica success=False")
    recs = j.get("result", {}).get("records", [])
    df = pd.DataFrame(recs)
    # Normalización simple
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for c in df.columns:
        if "fecha" in c and df[c].dtype == "object":
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # numéricos básicos
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="ignore")
            except Exception:
                pass
    return df

def auto_date_col(df: pd.DataFrame):
    for c in df.columns:
        if "fecha" in c and pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def main():
    st.title("Explorador COVID-19 (datos.gob.cl - CKAN)")
    st.caption("Busca recursos con 'covid' en el portal y carga datos desde el DataStore")

    with st.sidebar:
        st.header("Búsqueda")
        q = st.text_input("Palabra clave", value="covid")
        rows = st.slider("Conjuntos a revisar", 10, 100, 50, step=10)
        st.write("Pulsa el botón para ejecutar la búsqueda (está cacheada 30 min).")
        do_search = st.button("Buscar recursos")

    # Ejecutamos búsqueda siempre que hay clic o primera carga
    if 'searched' not in st.session_state:
        st.session_state['searched'] = True  # primera carga
    if do_search:
        st.session_state['searched'] = True

    if st.session_state['searched']:
        try:
            results = search_resources(query=q, rows=rows)
        except Exception as e:
            st.error(f"Error al buscar recursos CKAN: {e}")
            return
    else:
        results = pd.DataFrame()

    if results.empty:
        st.warning("No se encontraron recursos con DataStore activo para esa búsqueda.")
        return

    st.subheader("Recursos encontrados (DataStore activo)")
    st.dataframe(results, use_container_width=True, height=250)

    # Selector de recurso
    key_map = {f'{r["dataset_title"]} — {r["resource_name"]} ({r["format"]}) [{r["org"]}]': r["resource_id"]
               for _, r in results.iterrows()}
    choice = st.selectbox("Selecciona un recurso para cargar", options=list(key_map.keys()))
    resource_id = key_map[choice]

    # Parámetros de descarga y carga
    with st.sidebar:
        st.header("Carga de datos")
        limit = st.slider("Filas a descargar (limit)", 500, 10000, 5000, step=500)

    try:
        df = fetch_datastore(resource_id, limit=limit)
    except Exception as e:
        st.error(f"Error al obtener datos del DataStore: {e}")
        return

    if df.empty:
        st.warning("No hay datos para este recurso o el límite es muy bajo.")
        return

    st.markdown(f"**resource_id seleccionado:** `{resource_id}`")
    st.dataframe(df.head(200), use_container_width=True, height=300)

    # Detección de fecha y métricas
    date_col = auto_date_col(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", len(df))
    c2.metric("Columnas", df.shape[1])
    c3.metric("Columna de fecha", date_col or "—")

    # Selección de columnas numéricas y cat.
    all_cols = list(df.columns)
    with st.sidebar:
        st.header("Análisis")
        num_cols = st.multiselect("Columnas numéricas", [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])])
        cat_cols = st.multiselect("Columnas categóricas", [c for c in all_cols if df[c].dtype == "object"])

    # Filtro por fecha si existe
    if date_col:
        with st.sidebar:
            min_d = df[date_col].min().date()
            max_d = df[date_col].max().date()
            rango = st.date_input("Rango de fechas", (min_d, max_d), min_value=min_d, max_value=max_d)
        df = df[(df[date_col] >= pd.to_datetime(rango[0])) & (df[date_col] <= pd.to_datetime(rango[1]) + pd.Timedelta(days=1))]

    # Análisis
    if num_cols and cat_cols:
        st.subheader("Promedios por grupo")
        agg = df.groupby(cat_cols)[num_cols].mean(numeric_only=True)
        st.dataframe(agg, use_container_width=True)
        st.bar_chart(agg[num_cols[0]])
    elif num_cols:
        st.subheader("Estadísticas descriptivas")
        st.write(df[num_cols].describe().T)
    else:
        st.info("Selecciona columnas numéricas y/o categóricas para analizar.")

    # Serie temporal
    if date_col and num_cols:
        st.subheader("Serie temporal")
        metric = st.selectbox("Métrica", num_cols)
        ts = df[[date_col, metric]].dropna().set_index(date_col).sort_index()
        if not ts.empty:
            st.line_chart(ts[metric])

    # Descargar CSV
    st.download_button(
        "Descargar CSV (datos filtrados)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"covid_ckan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
