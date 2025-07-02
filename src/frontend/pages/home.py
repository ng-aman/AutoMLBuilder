# src/frontend/pages/home.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any


def render(session_state: Dict[str, Any]):
    """Render the home page"""
    st.title("🏠 AutoML Builder Dashboard")
    st.markdown("Welcome to your AI-powered AutoML platform")

    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 New Analysis", use_container_width=True):
            session_state["current_page"] = "Chat"
            st.rerun()

    with col2:
        if st.button("📁 Upload Dataset", use_container_width=True):
            session_state["current_page"] = "Chat"
            session_state["show_upload"] = True
            st.rerun()

    with col3:
        if st.button("🔬 View Experiments", use_container_width=True):
            session_state["current_page"] = "Experiments"
            st.rerun()

    with col4:
        if st.button("💬 Continue Chat", use_container_width=True):
            session_state["current_page"] = "Chat"
            st.rerun()

    # Statistics
    st.subheader("📈 Overview")

    # Get user statistics
    api_client = session_state.get("api_client")
    if api_client:
        # Fetch data
        sessions_data = api_client.get_chat_sessions(limit=100)
        datasets_data = api_client.get_datasets(limit=100)
        experiments_data = api_client.get_experiments(limit=100)

        # Create metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_sessions = sessions_data.get("total", 0) if sessions_data else 0
            st.metric(
                "Total Sessions",
                total_sessions,
                delta="+2 this week",
                delta_color="normal",
            )

        with col2:
            total_datasets = datasets_data.get("total", 0) if datasets_data else 0
            st.metric("Datasets", total_datasets, delta="+1 this week")

        with col3:
            total_experiments = (
                experiments_data.get("total", 0) if experiments_data else 0
            )
            st.metric("Experiments", total_experiments, delta="+5 this week")

        with col4:
            # Calculate average accuracy from experiments
            avg_accuracy = 0.0
            if experiments_data and experiments_data.get("experiments"):
                accuracies = []
                for exp in experiments_data["experiments"]:
                    if exp.get("results", {}).get("accuracy"):
                        accuracies.append(exp["results"]["accuracy"])
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)

            st.metric("Avg Accuracy", f"{avg_accuracy:.1%}", delta="+2.3%")

    # Recent activity
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Recent Sessions")
        if sessions_data and sessions_data.get("sessions"):
            recent_sessions = sessions_data["sessions"][:5]

            for session in recent_sessions:
                with st.container():
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.write(f"**{session['title'][:50]}...**")
                        st.caption(
                            f"Messages: {session['message_count']} | {session['created_at'][:10]}"
                        )
                    with cols[1]:
                        if st.button("Open", key=f"open_session_{session['id']}"):
                            session_state["current_session_id"] = session["id"]
                            session_state["current_page"] = "Chat"
                            st.rerun()
        else:
            st.info("No sessions yet. Start a new analysis!")

    with col2:
        st.subheader("🗂️ Recent Datasets")
        if datasets_data and datasets_data.get("datasets"):
            recent_datasets = datasets_data["datasets"][:5]

            for dataset in recent_datasets:
                with st.container():
                    st.write(f"**{dataset['filename']}**")
                    size_mb = dataset["file_size"] / (1024 * 1024)
                    st.caption(
                        f"Size: {size_mb:.1f} MB | Rows: {dataset['rows_count']} | Cols: {dataset['columns_count']}"
                    )
        else:
            st.info("No datasets uploaded yet.")

    # Visualizations
    st.subheader("📊 Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Model performance over time
        st.write("**Model Performance Trend**")

        if experiments_data and experiments_data.get("experiments"):
            # Create sample data for visualization
            experiments = experiments_data["experiments"]

            # Extract data for chart
            dates = []
            accuracies = []
            models = []

            for exp in experiments:
                if exp.get("created_at") and exp.get("results", {}).get("accuracy"):
                    dates.append(exp["created_at"][:10])
                    accuracies.append(exp["results"]["accuracy"])
                    models.append(exp["results"].get("model", "Unknown"))

            if dates:
                df = pd.DataFrame(
                    {"Date": dates, "Accuracy": accuracies, "Model": models}
                )

                fig = px.line(
                    df, x="Date", y="Accuracy", color="Model", markers=True, title=""
                )
                fig.update_layout(
                    showlegend=True, height=300, margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No experiment data available")
        else:
            # Show placeholder chart
            dates = pd.date_range(end=datetime.now(), periods=7, freq="D")
            df = pd.DataFrame(
                {"Date": dates, "Accuracy": [0.75, 0.78, 0.80, 0.82, 0.83, 0.85, 0.87]}
            )

            fig = px.line(df, x="Date", y="Accuracy", markers=True, title="")
            fig.update_layout(
                showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Dataset types distribution
        st.write("**Dataset Types**")

        if datasets_data and datasets_data.get("datasets"):
            # Count file types
            file_types = {}
            for dataset in datasets_data["datasets"]:
                ext = dataset["filename"].split(".")[-1].upper()
                file_types[ext] = file_types.get(ext, 0) + 1

            if file_types:
                df = pd.DataFrame(list(file_types.items()), columns=["Type", "Count"])

                fig = px.pie(df, values="Count", names="Type", title="")
                fig.update_layout(
                    showlegend=True, height=300, margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Show placeholder
            df = pd.DataFrame({"Type": ["CSV", "Excel", "JSON"], "Count": [5, 3, 2]})

            fig = px.pie(df, values="Count", names="Type", title="")
            fig.update_layout(
                showlegend=True, height=300, margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tips section
    st.subheader("💡 Getting Started Tips")

    with st.expander("How to use AutoML Builder"):
        st.markdown(
            """
        1. **Upload a Dataset**: Click "Upload Dataset" or go to the Chat page
        2. **Describe Your Goal**: Tell the AI what you want to predict or analyze
        3. **Review Analysis**: The AI will analyze your data and identify patterns
        4. **Approve Steps**: In Interactive mode, approve each preprocessing step
        5. **Train Models**: Multiple ML models will be trained automatically
        6. **Optimize**: Hyperparameters will be tuned for best performance
        7. **Export Results**: Download your trained model and predictions
        """
        )

    with st.expander("Best Practices"):
        st.markdown(
            """
        - **Data Quality**: Ensure your dataset is clean and well-formatted
        - **Target Variable**: Clearly identify what you want to predict
        - **Feature Selection**: Include relevant features for better predictions
        - **Data Size**: Larger datasets generally lead to better models
        - **Experiment Tracking**: All experiments are automatically tracked in MLflow
        """
        )

    with st.expander("Supported File Types"):
        st.markdown(
            """
        - **CSV**: Comma-separated values (recommended)
        - **Excel**: .xlsx and .xls files
        - **JSON**: JavaScript Object Notation
        
        Maximum file size: 100 MB
        """
        )

    # Footer
    st.markdown("---")
    st.caption(
        "Need help? Check out our [documentation](https://docs.automl-builder.com) or [contact support](mailto:support@automl-builder.com)"
    )
