from __future__ import annotations

from collections.abc import Iterable

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    *,
    eyebrow: str | None = None,
) -> None:
    if eyebrow:
        st.caption(eyebrow.upper())
    st.title(title)
    st.caption(subtitle)


def render_foundation_notice(area: str, source_ids: Iterable[str]) -> None:
    st.info(
        "Dashboard foundation is active. "
        f"The {area} view will be built from the frozen canonical sources below."
    )
    st.code("\n".join(source_ids), language=None)


def render_validation_summary(validation) -> None:
    failures = validation[validation["status"] != "PASS"]
    if failures.empty:
        st.success(f"Dashboard data contract validated: {len(validation)} checks passed.")
        return

    st.error(
        f"Dashboard data contract has {len(failures)} failing checks. "
        "Do not interpret dashboard outputs until they are resolved."
    )
    st.dataframe(failures, hide_index=True, use_container_width=True)
