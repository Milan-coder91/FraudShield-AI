                )

            else:

                st.info("No valid date records found.")

        else:

            st.info("No date or time column found in this dataset.")

        st.warning(
            "This page shows historical trends only. "
            "A forecasting model has not been connected here."
        )


# =====================================================
# 12. GRAPH ANALYTICS
# =====================================================

elif page == "Graph Analytics":

    st.title("🕸️ Graph Analytics")
    st.caption("Explore relationships between transaction entities.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        entity_cols = [
            col for col in df.columns
            if any(
                key in col.lower()
                for key in [
                    "customer",
                    "merchant",
                    "device",
                    "location",
                    "account"
                ]
            )
        ]

        if entity_cols:

            st.subheader("Available Entity Columns")

            st.write(entity_cols)

            selected_entity = st.selectbox(
                "Select Entity",
                entity_cols
            )

            entity_counts = (
                df[selected_entity]
                .astype(str)
                .value_counts()
                .head(15)
                .reset_index()
            )

            entity_counts.columns = [
                "Entity",
                "Transactions"
            ]

            fig = px.bar(
                entity_counts,
                x="Entity",
                y="Transactions",
                title="Top Entities by Transaction Count"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No customer, merchant or entity columns detected."
            )

        st.caption(
            "This is entity frequency analysis, not a connected graph network."
        )


# =====================================================
# FOOTER
# =====================================================

st.sidebar.divider()
st.sidebar.caption("🛡️ FraudShield AI")
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("© 2026 FraudShield AI")
