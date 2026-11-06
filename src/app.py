                    df[fraud_mask(df)][selected_col].astype(str)
                    .value_counts().head(20)
                    .rename_axis(selected_col)
                    .reset_index(name="Fraud Count")
                )

                fig = px.bar(
                    fraud_counts,
                    x=selected_col,
                    y="Fraud Count",
                    color="Fraud Count",
                    title=f"Fraud Relationships by {selected_col}",
                    color_continuous_scale="Reds"
                )
                show_chart(fig)

            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No supported relationship columns found.")


# =========================================================
# 20. DATA UPLOAD & REPORTS
# =========================================================

elif page == "📤 Data Upload & Reports":

    st.title("📤 Data Upload & Reports")

    st.markdown("""
    Upload your own CSV dataset. It will be available in the dataset
    selector during this app session. The original GitHub dataset
    files will not be overwritten.
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.success("Dataset uploaded successfully!")

            st.write("Dataset Preview")
            show_table(uploaded_df)

            st.write("Dataset Information")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(uploaded_df):,}")
            c2.metric("Columns", uploaded_df.shape[1])
            c3.metric(
                "Missing Values",
                int(uploaded_df.isnull().sum().sum())
            )

            dataset_name = uploaded_file.name

            if st.button("➕ Add Dataset to Dashboard"):
                st.session_state.uploaded_datasets[dataset_name] = uploaded_df
                st.success(
                    "Dataset added. It is now available in the dashboard selectors."
                )
                st.rerun()

        except Exception as error:
            st.error(f"Could not read uploaded CSV: {error}")

    st.divider()
    st.subheader("📚 Available Datasets")

    if datasets:
        for name, df in datasets.items():
            with st.expander(name):
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{len(df):,}")
                c2.metric("Columns", df.shape[1])
                c3.metric("Fraud Cases", fraud_count(df))

                st.dataframe(
                    df.head(20),
                    use_container_width=True,
                    hide_index=True
                )

                report_download_button(
                    df, name, f"{name} Report"
                )

                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"download_{name}"
                )
    else:
        st.warning("No datasets are currently available.")

    st.divider()
    st.subheader("📄 Full Dataset Report")

    selected = dataset_selector("report_dataset_selector")

    if selected:
        report_download_button(
            datasets[selected],
            selected,
            "FraudShield AI Full Dataset Report"
        )


# =========================================================
# 21. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    🛡️ <b>FraudShield AI</b><br>
    Enterprise Fraud Detection & Risk Intelligence Platform<br>
    Built with Python • Machine Learning • Streamlit
