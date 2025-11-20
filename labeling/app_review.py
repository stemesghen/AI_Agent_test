import json, glob
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------------------------
# Data setup
# ---------------------------
DATA_DIR = Path("data")
NORM_DIR = DATA_DIR / "normalized"
CLS_DIR  = DATA_DIR / "classified"
EXT_DIR  = DATA_DIR / "extracted"
LABELS_F = DATA_DIR / "labels" / "review.csv"
LABELS_F.parent.mkdir(parents=True, exist_ok=True)

# Taxonomy (synced with azure_provider.py)
INCIDENT_TYPES = [
    "grounding", "collision", "fire", "explosion", "piracy",
    "weather", "port_closure", "strike", "spill",
    "engine_failure", "canal_blockage", "security_threat"
]

# ---------------------------
# Helpers
# ---------------------------
def parse_csv(s: str):
    if not s:
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]

def current_is_done(is_incident: bool, types_list):
    """Row is 'done' if is_incident is False, or (True and at least one type)."""
    if not is_incident:
        return True
    return len(types_list) > 0

def safe_str(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

NULLISH = {"", "nan", "none", "null", "nil", "n/a", "na", "-"}

def non_empty(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s.lower() not in NULLISH else None

def coalesce(*vals) -> str:
    for v in vals:
        vv = non_empty(v)
        if vv is not None:
            return vv
    return ""

# ---------------------------
# Load data functions
# ---------------------------
@st.cache_data
def load_rows():
    rows = []
    cls_files = glob.glob(str(CLS_DIR / "*.classify.json"))
    for cf in cls_files:
        with open(cf, "r", encoding="utf-8") as f:
            cls = json.load(f)
        doc_id = cls["doc_id"]

        nf = NORM_DIR / (Path(cf).name.replace(".classify.json", ".json"))
        if not nf.exists():
            continue
        with open(nf, "r", encoding="utf-8") as f:
            norm = json.load(f)

        ef = EXT_DIR / (Path(cf).name.replace(".classify.json", ".extract.json"))
        extracted = {}
        if ef.exists():
            with open(ef, "r", encoding="utf-8") as f:
                extracted = json.load(f)

        rows.append({
            "doc_id": str(doc_id),
            "title": norm.get("title", ""),
            "url": norm.get("url", ""),
            "published_at": (norm.get("published_at", "") or "")[:10],
            "source_id": norm.get("source_id", ""),
            "is_incident_pred": bool(cls.get("is_incident", False)),
            "incident_types_pred": ",".join(cls.get("incident_types", [])),
            "vessel_pred": (extracted.get("vessel") if extracted else None),
            "imo_pred": (extracted.get("imo") if extracted else None),
            "port_pred": (extracted.get("port") if extracted else None),
            "date_pred": (extracted.get("date") if extracted else None),
            "content_text": (norm.get("content_text", "") or "")[:2000],
        })
    return pd.DataFrame(rows)

def load_labels():
    if LABELS_F.exists():
        df = pd.read_csv(LABELS_F).fillna("")
        if "doc_id" in df.columns:
            df["doc_id"] = df["doc_id"].astype(str)
        return df
    return pd.DataFrame(columns=[
        "doc_id","is_incident_true","incident_types_true",
        "vessel_true","imo_true","port_true","date_true","notes"
    ])

def upsert_label(row):
    df = load_labels()
    idx = df.index[df["doc_id"] == row["doc_id"]]
    if len(idx):
        df.loc[idx[0]] = row
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LABELS_F, index=False)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Incident Review", layout="wide")
st.title("Incident Review & Labeling")

data = load_rows()
labels = load_labels()
if len(data):
    data["doc_id"] = data["doc_id"].astype(str)

# keep current row in session
if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0

left, right = st.columns([2,3])

with left:
    st.subheader("Filters")
    only_inc = st.checkbox("Only show predicted incidents", value=True, key="only_inc")
    q = st.text_input("Search title", key="search_q")

    df = data.copy()
    if only_inc:
        df = df[df["is_incident_pred"] == True]
    if q:
        df = df[df["title"].str.contains(q, case=False, na=False)]

    st.caption(f"{len(df)} items")

    # Progress across current filtered set
    in_scope_ids = set(df["doc_id"].astype(str))
    lab = labels.copy()
    if len(lab):
        lab["doc_id"] = lab["doc_id"].astype(str)
        lab_in_scope = lab[lab["doc_id"].isin(in_scope_ids)]
    else:
        lab_in_scope = lab

    labeled_count = 0
    for _, r in lab_in_scope.iterrows():
        # consider labeled if is_incident set and if True, has ≥1 type
        is_incident_val = str(r.get("is_incident_true", "")).lower()
        if is_incident_val in ("true", "false"):
            if is_incident_val == "true":
                if len(parse_csv(r.get("incident_types_true", ""))) > 0:
                    labeled_count += 1
            else:
                labeled_count += 1
    st.caption(f"Labeled {labeled_count} / {len(df)}")
    st.progress(labeled_count / max(len(df), 1))

    # Advance behavior toggles
    auto_adv = st.checkbox("Auto-advance after save", value=True, key="auto_adv")
    only_advance_when_done = st.checkbox("Advance only when label is complete", value=True, key="adv_when_done")

    # Clamp index first
    max_idx = max(len(df) - 1, 0)
    st.session_state.row_idx = min(st.session_state.row_idx, max_idx)

    # Different key for number_input to avoid mutation conflict
    new_idx = int(st.number_input(
        "Row",
        min_value=0,
        max_value=max_idx,
        value=int(st.session_state.row_idx),
        step=1,
        key="row_spin"
    ))
    if new_idx != st.session_state.row_idx:
        st.session_state.row_idx = new_idx

with right:
    if len(df) == 0:
        st.info("No rows match your filter.")
    else:
        row = df.iloc[int(st.session_state.row_idx)].to_dict()
        st.subheader(row["title"])
        st.write(f"**Date**: {row['published_at']}  |  **Source**: {row['source_id']}")
        if row["url"]:
            st.write(f"[Open article]({row['url']})")
        with st.expander("Show content", expanded=False):
            st.write(row["content_text"])

        st.markdown("---")
        st.subheader("Predictions")
        p1, p2 = st.columns([1,3])
        p1.metric("Pred is_incident", "YES" if row["is_incident_pred"] else "NO")
        p2.write(
            f"Types: `{safe_str(row.get('incident_types_pred'))}` · "
            f"Vessel: `{safe_str(row.get('vessel_pred'))}` · "
            f"Port: `{safe_str(row.get('port_pred'))}` · "
            f"Date: `{safe_str(row.get('date_pred'))}` · "
            f"IMO: `{safe_str(row.get('imo_pred'))}`"
        )

        st.markdown("---")
        st.subheader("Your Labels (Ground Truth)")

        # pull prior saved label if exists
        prior_df = labels[labels["doc_id"] == row["doc_id"]]
        prior = prior_df.iloc[0].to_dict() if len(prior_df) else {}

        c1, c2 = st.columns(2)

        # Put all inputs inside a form so they submit atomically
        with st.form(key=f"form_{row['doc_id']}"):
            is_incident_true = c1.selectbox(
                "Is incident?",
                [True, False],
                index=0 if row["is_incident_pred"] else 1,
                key=f"is_{row['doc_id']}"
            )

            incident_types_true = c1.multiselect(
                "Incident types",
                INCIDENT_TYPES,
                default=parse_csv(row.get("incident_types_pred", "")),
                key=f"types_{row['doc_id']}"
            )

            vessel_true = c2.text_input(
                "Vessel",
                value=coalesce(prior.get("vessel_true"), row.get("vessel_pred")),
                key=f"v_{row['doc_id']}"
            )
            port_true = c2.text_input(
                "Port",
                value=coalesce(prior.get("port_true"), row.get("port_pred")),
                key=f"p_{row['doc_id']}"
            )
            date_true = c2.text_input(
                "Date (YYYY-MM-DD)",
                value=coalesce(prior.get("date_true"), row.get("date_pred"), row.get("published_at")),
                key=f"d_{row['doc_id']}"
            )
            imo_true = c2.text_input(
                "IMO (7 digits)",
                value=coalesce(prior.get("imo_true"), row.get("imo_pred")),
                key=f"imo_{row['doc_id']}"
            )
            notes = st.text_area("Notes", value=safe_str(prior.get("notes","")), key=f"n_{row['doc_id']}")

            # Row status (computed from current widget values)
            row_done = current_is_done(bool(is_incident_true), incident_types_true)
            if row_done:
                st.success("Status: Labeled")
            else:
                st.info("Status: Not complete (add incident types if 'Is incident?' is True)")

            # Optional hard check
            if is_incident_true and len(incident_types_true) == 0:
                st.warning("Please select at least one incident type when 'Is incident?' is True.")

            submitted = st.form_submit_button("Save label")

        # Handle submission AFTER the form context
        if submitted:
            upsert_label({
                "doc_id": str(row["doc_id"]),
                "is_incident_true": bool(is_incident_true),
                "incident_types_true": ",".join(incident_types_true),
                "vessel_true": coalesce(vessel_true),
                "imo_true": coalesce(imo_true),
                "port_true": coalesce(port_true),
                "date_true": coalesce(date_true),
                "notes": notes.strip(),
            })
            st.success("Saved!")

            # Decide whether to advance (never touch row_spin)
            should_advance = auto_adv and (row_done or not only_advance_when_done)
            if should_advance:
                st.session_state.row_idx = min(st.session_state.row_idx + 1, max_idx)
            st.rerun()

# ---------------------------
# Labeled Incidents Listing
# ---------------------------
st.markdown("---")
st.subheader(" Labeled Incidents")

lab_all = labels.copy()
if len(lab_all):
    lab_all["doc_id"] = lab_all["doc_id"].astype(str)

if len(lab_all):
    # keep only True incidents
    is_true = lab_all["is_incident_true"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    lab_inc = lab_all[is_true].copy()

    # Join with normalized/news data for extra columns
    cols_from_data = ["doc_id", "title", "published_at", "source_id", "url"]
    data_for_join = data[cols_from_data] if len(data) else pd.DataFrame(columns=cols_from_data)
    merged = lab_inc.merge(data_for_join, on="doc_id", how="left")

    if len(merged):
        merged["types"] = merged["incident_types_true"].apply(lambda s: ", ".join(parse_csv(s)))
        display_cols = [
            "published_at", "title", "types",
            "vessel_true", "port_true", "date_true", "imo_true",
            "source_id", "url", "doc_id"
        ]
        for c in display_cols:
            if c not in merged.columns:
                merged[c] = ""
        merged = merged[display_cols].sort_values("published_at", ascending=False)

        # Quick filters
        cA, cB = st.columns([2,1])
        q_list = cA.text_input("Search labeled incidents (title contains)", key="q_list")
        type_filter = cB.multiselect("Filter by type", INCIDENT_TYPES, key="type_filter")

        df_list = merged.copy()
        if q_list:
            df_list = df_list[df_list["title"].str.contains(q_list, case=False, na=False)]
        if type_filter:
            df_list = df_list[df_list["types"].apply(lambda s: any(t in s.split(', ') for t in type_filter))]

        st.dataframe(df_list, use_container_width=True, hide_index=True)

        # Download CSV
        csv_bytes = df_list.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download labeled incidents (CSV)",
            data=csv_bytes,
            file_name="labeled_incidents.csv",
            mime="text/csv"
        )

        # Summary: counts by type
        st.markdown("#### Counts by type")
        tmp = merged.copy()
        tmp["types_list"] = tmp["types"].apply(parse_csv)
        tmp = tmp.explode("types_list")
        tmp["types_list"] = tmp["types_list"].fillna("").str.strip()
        tmp = tmp[tmp["types_list"] != ""]
        if len(tmp):
            counts = tmp["types_list"].value_counts().rename_axis("incident_type").reset_index(name="count")
            st.dataframe(counts, hide_index=True, use_container_width=True)
        else:
            st.caption("No incident types found to summarize.")
    else:
        st.info("No labeled incidents yet. Once you save labels with 'Is incident? = True', they will appear here.")
else:
    st.info("No labels file found yet. Save at least one label to create it.")

