"""Enterprise-grade reusable KPI cards, headers, and UI components."""

from typing import Any, Dict, List, Optional
import streamlit as st
from dashboard.styles.theme import get_theme_tokens


def page_header(
    title: str,
    subtitle: str = "",
    category: Optional[str] = None,
    badge_text: Optional[str] = None,
    badge_variant: str = "blue",
) -> None:
    """Render an enterprise page header with optional category tag and status badge."""
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="badge badge-{badge_variant}">{badge_text}</span>'

    category_html = ""
    if category:
        category_html = (
            f'<div style="font-size:0.75rem;font-weight:600;letter-spacing:0.06em;'
            f'text-transform:uppercase;color:var(--text-muted);margin-bottom:0.2rem;">'
            f'{category}</div>'
        )

    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<p style="font-size:0.88rem;color:var(--text-muted);margin:0.25rem 0 0 0;">{subtitle}</p>'

    header_html = f"""
    <div style="margin-bottom: 0.75rem;">
        {category_html}
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.75rem;">
            <h1 style="margin:0;display:flex;align-items:center;gap:0.75rem;">
                {title}
            </h1>
            {badge_html}
        </div>
        {subtitle_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("<hr style='margin:0 0 1rem 0;'>", unsafe_allow_html=True)


def kpi_row(metrics: List[Dict[str, Any]]) -> None:
    """Render a responsive row of enterprise KPI cards.

    Each metric dictionary can contain:
    - label: str (metric title)
    - value: str | int | float
    - delta: Optional[str] (change/comparison text)
    - delta_color: Optional[str] ('normal', 'inverse', 'off')
    - caption: Optional[str] (neutral supporting context)
    - help: Optional[str] (tooltip explaining calculation/provenance)
    - provenance: Optional[str] (short source label displayed beneath metric)
    """
    cols = st.columns(len(metrics), gap="small")
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
                help=m.get("help"),
            )
            if m.get("caption"):
                st.caption(m["caption"])
            if "provenance" in m and m["provenance"]:
                st.markdown(
                    f'<div style="font-size:0.7rem;color:var(--text-dim);margin-top:-0.5rem;padding-left:0.2rem;">'
                    f'Source: {m["provenance"]}</div>',
                    unsafe_allow_html=True,
                )


def under_development_banner(
    module_title: str,
    target_module: str,
    description: str,
    disclaimer: Optional[str] = None,
) -> None:
    """Render a clean, professional banner for modules currently under development."""
    disclaimer_html = ""
    if disclaimer:
        disclaimer_html = (
            f'<div style="margin-top:0.75rem;padding:0.6rem 0.85rem;'
            f'background:color-mix(in srgb,var(--accent-orange) 8%,transparent);'
            f'border-radius:6px;border-left:3px solid var(--accent-orange);'
            f'font-size:0.8rem;color:var(--text-muted);">'
            f'<strong>Governance Notice:</strong> {disclaimer}</div>'
        )

    st.markdown(
        f"""
        <div class="dev-status-banner">
            <div class="dev-status-title">
                <span>Under Active Development</span>
                <span class="badge badge-orange">Roadmap</span>
            </div>
            <p class="dev-status-body">
                The backend analytical module for <strong>{module_title}</strong> (<code>{target_module}</code>) 
                is in progress. {description}
            </p>
            {disclaimer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def provenance_expander(items: List[Dict[str, str]]) -> None:
    """Render a clean collapsible provenance & methodology drawer for audits."""
    with st.expander("Audit & Data Provenance Details", expanded=False):
        for item in items:
            st.markdown(
                f"""
                <div style="font-size:0.82rem;margin-bottom:0.6rem;padding-bottom:0.6rem;border-bottom:1px solid var(--border-subtle);">
                    <strong style="color:var(--text-primary);">{item.get('metric', '')}</strong><br>
                    <span style="color:var(--text-muted);">Source File: <code>{item.get('source', '')}</code></span><br>
                    <span style="color:var(--text-dim);">{item.get('description', '')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Backwards compatibility
def section_header(title: str, subtitle: str = "") -> None:
    """Backwards-compatible wrapper for page headers."""
    page_header(title=title, subtitle=subtitle)


def placeholder_section(
    title: str,
    description: str,
    module_paths: Optional[List[str]] = None,
) -> None:
    """Backwards-compatible wrapper for development notice."""
    target = module_paths[0] if module_paths else "Backend Module"
    under_development_banner(
        module_title=title,
        target_module=target,
        description=description,
    )
