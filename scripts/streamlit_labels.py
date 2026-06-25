import streamlit as st
import polars as pl
import plotly.express as px

# Configuration - FIXED to match your pipeline's exact output path
PATH_TO_LABELED_PARQUET = "data/processed/labeled_subset.pqt"

# Set page configuration to wide mode for optimal text reading
st.set_page_config(page_title="DIMI Narrative Explorer", layout="wide")

@st.cache_data
def load_data(path):
    """Loads and caches the labeled parquet file."""
    df = pl.read_parquet(path)
    return df

try:
    df = load_data(PATH_TO_LABELED_PARQUET)
except Exception as e:
    st.error(f"Could not load data from {PATH_TO_LABELED_PARQUET}. Error: {e}")
    st.stop()

# Re-mapped to match the exact keys from your new JSON_SCHEMA unnest output
label_columns = {
    "elite_vs_mass_conflict": "Populist Narrative",
    "in_group_vs_out_group_exclusion": "Nativist Narrative",
    "institutional_knowledge_denial": "Denialist Narrative",
    "societal_moral_regression": "Declinist Narrative",
    "imminent_acute_crisis_panic": "Apocalypticist Narrative",
    "systemic_sovereignty_revival": "Revisionist Narrative"
}

st.title("DIMI Narrative Dataset Explorer")
st.markdown("Use this interactive environment to audit the LLM classifications, view the text distribution, and verify the model's rationale.")

# Sidebar Controls
st.sidebar.header("Data Filters")

# Filter out baseline noise if desired
show_baseline = st.sidebar.checkbox("Include Baseline Noise (All 0s)", value=True)
if not show_baseline:
    df = df.filter(pl.col("no_contested_narrative_present") == False)

# Filter by Unique Channel (peer_id)
all_channels = ["All Channels"] + [str(x) for x in df["peer_id"].unique().to_list()]
selected_channel = st.sidebar.selectbox("Filter by Channel (peer_id)", all_channels)
if selected_channel != "All Channels":
    df = df.filter(pl.col("peer_id") == int(selected_channel))

# Filter by Narrative Intensities
st.sidebar.subheader("Filter by Minimum Intensity")
min_scores = {}
for col_name, display_name in label_columns.items():
    min_scores[col_name] = st.sidebar.slider(f"Min {display_name}", 0, 3, 0)
    if min_scores[col_name] > 0:
        df = df.filter(pl.col(col_name) >= min_scores[col_name])

# Main Dashboard Metrics Layout
total_rows = len(df)
st.metric(label="Total Messages Matching Filters", value=total_rows)

if total_rows == 0:
    st.warning("No rows match the specified filters. Try reducing the minimum narrative thresholds.")
    st.stop()

# Distribution Charts Section
st.header("Narrative Distribution Summary")
col1, col2 = st.columns([1, 1])

with col1:
    # Calculate the average intensity of each narrative
    mean_intensities = [df[col].mean() for col in label_columns.keys()]
    fig_bar = px.bar(
        x=list(label_columns.values()),
        y=mean_intensities,
        labels={"x": "Narrative Label", "y": "Mean Intensity (0-3)"},
        title="Average Intensity Across Screened Subset",
        color=mean_intensities,
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # Show the baseline split distribution
    baseline_counts = df["no_contested_narrative_present"].value_counts()
    
    # FIXED: Modern structural column selection to handle Polars series output types cleanly
    label_col = "no_contested_narrative_present"
    count_col = [c for c in baseline_counts.columns if c != label_col][0]
    
    baseline_counts = baseline_counts.with_columns(
        pl.col(label_col).map_elements(
            lambda x: "Baseline Chatter" if x else "Active DIMI Narrative",
            return_dtype=pl.String
        )
    )
    fig_pie = px.pie(
        names=baseline_counts[label_col].to_list(),
        values=baseline_counts[count_col].to_list(),
        title="Active Narratives vs. Baseline Noise Ratio",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Data Audit Table Viewer 
st.header("Row-Level Text Audit View")
st.markdown("Select a row index below to extract the complete text payload alongside the classification array and LLM rationale statement.")

# Reformat dataframe safely for tabular viewing display slice (with fallback for message_id formatting)
id_col = "message_id" if "message_id" in df.columns else df.columns[0]
display_cols = [id_col, "peer_id"] + list(label_columns.keys()) + ["no_contested_narrative_present"]
df_display = df.select(display_cols).to_pandas()

# Add a row-selection index tool element
selected_row_idx = st.selectbox("Choose message row index to inspect:", range(len(df_display)))

# Build data grid column layout containers for the audited selection inspection card
inspect_col1, inspect_col2 = st.columns([2, 1])

with inspect_col1:
    st.subheader("Raw Message Text Payload")
    raw_text = df["message_text"][selected_row_idx]
    st.info(raw_text if raw_text else "[Empty/No Text Content]")
    
    st.subheader("Generation Chain Rationale")
    # FIXED: Re-mapped field lookup reference to "rationale" column parsed directly from the pipeline
    rationale = df["rationale"][selected_row_idx] if "rationale" in df.columns else "No rationale field saved."
    st.code(rationale, wrap_lines=True)

with inspect_col2:
    st.subheader("Label Intensities")
    for col_name, display_name in label_columns.items():
        score = df[col_name][selected_row_idx]
        
        # Color coding metrics depending on severity levels
        if score == 3:
            st.error(f"**{display_name}**: 3 (Severe / Critical)")
        elif score == 2:
            st.warning(f"**{display_name}**: 2 (Active / Moderate)")
        elif score == 1:
            st.info(f"**{display_name}**: 1 (Low / Suggestive)")
        else:
            st.text(f"⚪ {display_name}: 0")

st.markdown("---")
st.subheader("Slice Preview")
st.dataframe(df_display, use_container_width=True)