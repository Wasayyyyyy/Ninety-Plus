"""
Premier League Player Scouting & Similarity Dashboard.
Streamlit application implementing:

- Position-group filtering (All, GK, DEF, MID, ATT)
- Real-time user-weighted similarity rankings (Attacking vs Defensive weighting)
- 2D PCA cluster space visualization with selected player highlighting
- Player stat cards and verified squad overrides
- Standardized data freshness footer
Run with: streamlit run src/module3_scouting/app.py
"""

# Ensure the project root is on sys.path so `src.*` imports resolve
# regardless of how/where Streamlit is invoked.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rapidfuzz import process, fuzz

from src.config import RANDOM_SEED
from src.data_layer.metadata import get_metadata
from src.data_layer.squad_overrides import get_current_club
from src.module3_scouting.cluster import (
    load_scouting_bundle,
    compute_weighted_similarity,
    ALL_FEATURES,
)

# Page configuration
st.set_page_config(
    page_title="Premier League Scouting Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #f8fafc;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .player-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    
    .badge-pos {
        background-color: #3b82f6;
        color: white;
    }
    
    .badge-cluster {
        background-color: #8b5cf6;
        color: white;
    }
    
    .badge-override {
        background-color: #10b981;
        color: white;
    }
    
    .freshness-box {
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_scouting_bundle()


def main():
    bundle = get_bundle()
    df = bundle["data"].copy()
    scaler = bundle["scaler"]

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff;">
            ⚽ Premier League Scouting Intelligence
        </h1>
        <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 1.05rem;">
            Statistical similarity search and K-means cluster segmentation across Premier League player-seasons.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown("### 🔍 Search & Filters")

    # 1. Position Filter
    position_options = ["All", "GK", "DEF", "MID", "ATT"]
    selected_position = st.sidebar.selectbox("Filter by Position Group:", position_options, index=0)

    # Filter available players by position if desired
    if selected_position != "All":
        filtered_df = df[df["position_group"] == selected_position]
    else:
        filtered_df = df

    # 2. Player Selector
    player_options = sorted(filtered_df["display_name"].unique())
    # Default to a famous star if available
    default_idx = 0
    for i, name in enumerate(player_options):
        if "Bukayo Saka" in name or "Erling Haaland" in name or "Salah" in name:
            default_idx = i
            break

    selected_display_name = st.sidebar.selectbox("Select Target Player:", player_options, index=default_idx)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ Weighted Similarity Profile")
    st.sidebar.caption("Adjust relative importance between attacking production vs defensive/workrate output:")

    attack_weight = st.sidebar.slider("Attacking Weight (Goals & Assists)", min_value=0.2, max_value=3.0, value=1.5, step=0.1)
    defense_weight = st.sidebar.slider("Durability & Discipline Weight", min_value=0.2, max_value=3.0, value=0.8, step=0.1)
    top_n = st.sidebar.slider("Number of Similar Players:", min_value=3, max_value=15, value=6)

    # Extract target record
    target_idx = df[df["display_name"] == selected_display_name].index[0]
    target_row = df.loc[target_idx]

    # Resolve club via squad_overrides
    current_club, club_source = get_current_club(
        target_row["name"], fallback_club=target_row.get("current_club_name")
    )

    # Main dashboard body layout: Top Stats Card
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Position Group", f"{target_row['position_group']} ({target_row.get('sub_position', 'N/A')})")
    with col2:
        st.metric("Season Minutes", f"{target_row['minutes_played']:,.0f} mins")
    with col3:
        st.metric("Goals / 90", f"{target_row['goals_per_90']:.2f}")
    with col4:
        st.metric("Assists / 90", f"{target_row['assists_per_90']:.2f}")
    with col5:
        mkt_val = target_row.get("market_value_in_eur", 0)
        val_str = f"€{mkt_val / 1e6:.1f}M" if pd.notna(mkt_val) and mkt_val > 0 else "N/A"
        st.metric("Market Value", val_str)

    st.markdown(f"""
    <div style="margin-bottom: 20px; font-size: 0.95rem; color: #94a3b8;">
        <b>Current Club:</b> <span style="color: #38bdf8; font-weight: 600;">{current_club}</span> 
        <span style="font-size: 0.8rem; margin-left: 8px; color: #64748b;">({club_source})</span>
        <span class="metric-badge badge-cluster" style="margin-left: 12px;">Cluster #{target_row['cluster']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Calculate similarity rankings
    ranked_similar = compute_weighted_similarity(
        target_idx=target_idx,
        df=df,
        scaler=scaler,
        attack_weight=attack_weight,
        defense_weight=defense_weight,
        position_filter=selected_position,
        top_n=top_n,
    )

    # Layout: Table of Similar Players (Left) & 2D PCA Cluster Map (Right)
    t_col, p_col = st.columns([1.1, 0.9])

    with t_col:
        st.markdown(f"### 🎯 Top {top_n} Statistically Similar Players")
        table_view = ranked_similar[[
            "display_name", "position_group", "goals_per_90", "assists_per_90",
            "minutes_played", "similarity_pct"
        ]].copy()
        table_view.columns = ["Player (Season)", "Pos", "G/90", "A/90", "Minutes", "Similarity (%)"]
        table_view["G/90"] = table_view["G/90"].round(2)
        table_view["A/90"] = table_view["A/90"].round(2)
        table_view["Minutes"] = table_view["Minutes"].astype(int)

        st.dataframe(
            table_view,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "💡 **Architecture Note:** True similarity is calculated via weighted Euclidean distance in standardized feature space. "
            "K-Means clusters provide coarse tactical grouping, while the distance function drives ranking."
        )

    with p_col:
        st.markdown("### 🗺️ 2D Tactical Feature Space (PCA)")

        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=140)
        plt.style.use("dark_background")

        # Scatter of all players colored by cluster
        scatter = ax.scatter(
            df["pca_x"],
            df["pca_y"],
            c=df["cluster"],
            cmap="viridis",
            alpha=0.25,
            s=25,
            edgecolors="none",
        )

        # Plot similar players
        ax.scatter(
            ranked_similar["pca_x"],
            ranked_similar["pca_y"],
            color="#38bdf8",
            s=80,
            edgecolors="white",
            linewidth=1.5,
            label="Similar Matches",
            zorder=4,
        )

        # Plot target player prominently
        ax.scatter(
            target_row["pca_x"],
            target_row["pca_y"],
            color="#ef4444",
            s=160,
            marker="*",
            edgecolors="yellow",
            linewidth=1.8,
            label=f"Target: {target_row['name']}",
            zorder=5,
        )

        ax.set_title("Player Distribution & Clusters", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("PCA Component 1", fontsize=9)
        ax.set_ylabel("PCA Component 2", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # Standardized Data Freshness Footer
    meta = get_metadata()
    footer_text = meta.format_freshness_footer(current_club_source=club_source)

    st.markdown(f"""
    <div class="freshness-box">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 6px;">DATA FRESHNESS & PROVENANCE:</div>
        <pre style="margin: 0; background: transparent; border: none; color: #94a3b8; font-family: inherit;">{footer_text}</pre>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
