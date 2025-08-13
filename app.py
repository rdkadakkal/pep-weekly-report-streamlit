# app.py
import io
import os
from pathlib import Path
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

st.set_page_config(page_title="Pepsico Cleaner + DQ Lookup", layout="wide")

# ===============================
# Helpers & Core Transformation
# ===============================

TEMPLATE_COLUMNS = [
    'Tenant Name', 'Shipment Mode', 'Agg Date', 'Carrier Name', 'Destination Country',
    'Drop-off Region', 'Region Pickup', 'Pickup Country', 'Tracking Method', 'Tracking Type',
    'Period Date', 'Destination Country.1', 'Final Status Reason', 'P44 Shipment ID',
    'Pickup Country.1', 'Tracked', 'Active Equipment ID', 'Attr1 Name', 'Attr1 Value',
    'Attr2 Name', 'Attr2 Value', 'Attr3 Name', 'Attr3 Value', 'Attr4 Name', 'Attr4 Value',
    'Attr5 Name', 'Attr5 Value', 'Bill of Lading', 'Destination Name', 'Dropoff Arrival Milestone',
    'Dropoff City State', 'Dropoff Departure Milestone', 'Ft Shipment Error',
    'Has Equipment ID (Yes / No)', 'Historical Equipment ID', 'IS_PING_COMPLETE',
    'P44 Carrier ID', 'Pickup Arrival Milestone', 'Pickup City State',
    'Pickup Departure Milestone', 'Pickup State', 'PICKUP_ARRIVAL_STATUS_30_MIN',
    'Pickup Name', 'Tenant ID', 'Tl Equipment ID Source', 'TOTAL_STOPS', 'TRACKING_METHOD_RCA'
]

COUNTRY_MAP = {
    "AD": "Andorra", "BA": "Bosnia and Herzegovina", "BE": "Belgium",
    "BG": "Bulgaria", "CY": "Cyprus", "DE": "Germany", "EE": "Estonia",
    "ES": "Spain", "FR": "France", "GB": "United Kingdom of Great Britain and Northern Ireland",
    "GR": "Greece", "HR": "Croatia", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RS": "Serbia", "SI": "Slovenia", "US": "United States of America"
}

EXPECTED_ATTR_MAPPING = {
    'Business Unit': 'Attr1',
    'PO': 'Attr2',
    'TMSTOPID': 'Attr3',
    'Order Type': 'Attr4',
    'GTMSLOAT': 'Attr5'
}

def dedupe_semicolon_list(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        # normalize commas to semicolons, split, strip, unique-preserve-order
        for sep in [',', ';']:
            value = value.replace(sep, ';')
        parts = [p.strip() for p in value.split(';') if p.strip()]
        unique = list(dict.fromkeys(parts))
        return ';'.join(unique)
    return value

def rearrange_attrs_row(row):
    out = {
        'Attr1 Name': '', 'Attr1 Value': '',
        'Attr2 Name': '', 'Attr2 Value': '',
        'Attr3 Name': '', 'Attr3 Value': '',
        'Attr4 Name': '', 'Attr4 Value': '',
        'Attr5 Name': '', 'Attr5 Value': ''
    }
    for i in range(1, 6):
        name = row.get(f'Attr{i} Name')
        value = row.get(f'Attr{i} Value')
        if pd.notna(name) and name in EXPECTED_ATTR_MAPPING:
            target = EXPECTED_ATTR_MAPPING[name]
            out[f'{target} Name'] = name
            out[f'{target} Value'] = value
    return pd.Series(out)

def monday_of_week(series_dt: pd.Series) -> pd.Series:
    # Robust: Monday = date - timedelta(weekday)
    return (series_dt - pd.to_timedelta(series_dt.dt.weekday, unit='D')).dt.normalize()

def _norm_bol(s):
    if pd.isna(s):
        return None
    return str(s).strip().upper()

def _find_col_ci(df, target_name):
    """Case-insensitive, space-normalized column finder."""
    canonical = " ".join(target_name.lower().split())
    for c in df.columns:
        if " ".join(str(c).lower().split()) == canonical:
            return c
    return None

def process_files(main_df: pd.DataFrame, dq_df: pd.DataFrame | None, keep_audit_col: bool = False):
    # 1) Start from template columns structure
    out = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    for col in TEMPLATE_COLUMNS:
        if col in main_df.columns:
            out[col] = main_df[col]
        else:
            out[col] = None

    # 2) Manual renames (copy from Shipment Tracking Type/Method if present)
    if 'Shipment Tracking Type' in main_df.columns:
        out['Tracking Type'] = main_df['Shipment Tracking Type']
    if 'Shipment Tracking Method' in main_df.columns:
        out['Tracking Method'] = main_df['Shipment Tracking Method']

    # 3) Agg Date from Period Date (week starting Monday)
    agg_nats = 0
    if 'Period Date' in out.columns:
        pdts = pd.to_datetime(out['Period Date'], errors='coerce')
        out['Agg Date'] = monday_of_week(pdts)
        agg_nats = int(pdts.isna().sum())

    # 4) Country code mapping
    for col in ['Destination Country', 'Pickup Country']:
        if col in out.columns:
            out[col] = out[col].map(COUNTRY_MAP).fillna(out[col])

    # 5) Attribute realignment
    attrs = out.apply(rearrange_attrs_row, axis=1)
    for col in attrs.columns:
        out[col] = attrs[col]

    # 6) De-duplicate AttrX Value lists
    for i in range(1, 6):
        c = f'Attr{i} Value'
        if c in out.columns:
            out[c] = out[c].apply(dedupe_semicolon_list)

    # 7) VLOOKUP-style update from DQ (if provided)
    updated_count = 0
    if dq_df is not None:
        bol_col = _find_col_ci(dq_df, "Bill of Lading")
        err_col = _find_col_ci(dq_df, "Tracking Error")
        if bol_col is None or err_col is None:
            raise KeyError(f"Required columns not found in DQ file. Have: {list(dq_df.columns)}")

        dq_df['__BOL_KEY__'] = dq_df[bol_col].map(_norm_bol)
        dq_df['__TRACKING_ERROR__'] = dq_df[err_col]
        dq_lookup = (
            dq_df.dropna(subset=['__BOL_KEY__'])
                 .drop_duplicates(subset=['__BOL_KEY__'])
                 .set_index('__BOL_KEY__')['__TRACKING_ERROR__']
        )

        if 'Bill of Lading' not in out.columns:
            raise KeyError("'Bill of Lading' column missing in main dataset.")
        if 'Ft Shipment Error' not in out.columns:
            raise KeyError("'Ft Shipment Error' column missing in main dataset.")

        out['__BOL_KEY__'] = out['Bill of Lading'].map(_norm_bol)

        mask_not_identified = (
            out['Ft Shipment Error'].astype(str).str.strip().str.casefold().eq('not identified')
        )
        mapped_errors = out.loc[mask_not_identified, '__BOL_KEY__'].map(dq_lookup)
        idx_to_write = mapped_errors.index[
            mapped_errors.notna() & mapped_errors.astype(str).str.len().gt(0)
        ]
        out.loc[idx_to_write, 'Ft Shipment Error'] = mapped_errors.loc[idx_to_write]
        updated_count = len(idx_to_write)

        if keep_audit_col:
            out['Tracking Error (from DQ)'] = out['__BOL_KEY__'].map(dq_lookup)

        out.drop(columns=['__BOL_KEY__'], inplace=True, errors='ignore')

    return out, {'agg_date_nats': agg_nats, 'ft_error_updates': updated_count}

def to_excel_bytes(df: pd.DataFrame, filename: str = "Pepsico0.xlsx") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read()

# ===============================
# UI
# ===============================

st.title("Pepsico Report Cleaner + DQ VLOOKUP")
st.markdown(
    "Upload the **Data Availability Trend by Selected Dimensions (4).xlsx** and (optionally) "
    "**Data Quality by Carrier (3).xlsx**. The app will reshape columns, compute **Agg Date** "
    "(Monday of the week), realign Attrs, de-duplicate lists, map country codes, and fill "
    "**Ft Shipment Error** where it’s *Not Identified* using the DQ file."
)

with st.sidebar:
    st.header("Options")
    keep_audit = st.checkbox("Keep audit column ‘Tracking Error (from DQ)’", value=False)
    show_preview = st.checkbox("Show result preview (first 200 rows)", value=True)

col1, col2 = st.columns(2)
with col1:
    main_file = st.file_uploader("Main file: Data Availability Trend by Selected Dimensions",
                                 type=["xlsx"])
with col2:
    dq_file = st.file_uploader("Vlookup file: Data Quality by Carrier", type=["xlsx"])

process = st.button("Process")

if process:
    if not main_file:
        st.error("Please upload the main file.")
        st.stop()

    try:
        main_df = pd.read_excel(main_file, dtype=str)
        # Keep original dtypes where relevant
    except Exception as e:
        st.error(f"Failed to read main file: {e}")
        st.stop()

    dq_df = None
    if dq_file is not None:
        try:
            dq_df = pd.read_excel(dq_file, dtype=str)
        except Exception as e:
            st.warning(f"Could not read DQ file—continuing without VLOOKUP. Error: {e}")
            dq_df = None

    try:
        result_df, stats = process_files(main_df, dq_df, keep_audit_col=keep_audit)

        st.success("Processing complete.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows processed", len(result_df))
        m2.metric("NaT in Period Date", stats.get('agg_date_nats', 0))
        m3.metric("Ft Shipment Error updated", stats.get('ft_error_updates', 0))

        if show_preview:
            st.dataframe(result_df.head(200))

        xls_bytes = to_excel_bytes(result_df, filename="Pepsico0.xlsx")
        st.download_button(
            label="⬇️ Download Pepsico0.xlsx",
            data=xls_bytes,
            file_name="Pepsico0.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except KeyError as ke:
        st.error(f"Missing required column: {ke}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

st.markdown("---")
st.caption("Tip: If your column headers differ slightly (spaces/casing), the DQ step is flexible; core template columns must match exactly.")
