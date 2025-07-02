# src/frontend/pages/experiments.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List


def render(session_state: Dict[str, Any]):
    """Render the experiments page"""
    st.title("🔬 Experiments")
    st.markdown("Track and compare your ML experiments")

    api_client = session_state.get("api_client")
    if not api_client:
        st.error("API client not initialized")
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Compare", "📈 MLflow"])

    with tab1:
        render_experiments_overview(api_client)

    with tab2:
        render_experiments_comparison(api_client)

    with tab3:
        render_mlflow_integration()


def render_experiments_overview(api_client):
    """Render experiments overview"""
    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        session_filter = st.selectbox(
            "Filter by Session",
            ["All Sessions", "Current Session Only"],
            key="exp_session_filter",
        )

    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "completed", "running", "failed"],
            key="exp_status_filter",
        )

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Recent First", "Oldest First", "Best Performance"],
            key="exp_sort",
        )

    # Get experiments
    params = {}
    if session_filter == "Current Session Only" and st.session_state.get(
        "current_session_id"
    ):
        params["session_id"] = st.session_state["current_session_id"]
    if status_filter != "All":
        params["status"] = status_filter

    experiments_data = api_client.get_experiments(**params)

    if experiments_data and experiments_data.get("experiments"):
        experiments = experiments_data["experiments"]

        # Sort experiments
        if sort_by == "Recent First":
            experiments.sort(key=lambda x: x["created_at"], reverse=True)
        elif sort_by == "Oldest First":
            experiments.sort(key=lambda x: x["created_at"])
        elif sort_by == "Best Performance":
            experiments.sort(
                key=lambda x: x.get("results", {}).get("accuracy", 0), reverse=True
            )

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Experiments", len(experiments))

        with col2:
            completed = sum(1 for e in experiments if e["status"] == "completed")
            st.metric("Completed", completed)

        with col3:
            if completed > 0:
                avg_accuracy = (
                    sum(
                        e.get("results", {}).get("accuracy", 0)
                        for e in experiments
                        if e["status"] == "completed"
                    )
                    / completed
                )
                st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
            else:
                st.metric("Avg Accuracy", "N/A")

        with col4:
            running = sum(1 for e in experiments if e["status"] == "running")
            st.metric("Running", running)

        st.divider()

        # Experiments table
        st.subheader("Experiments List")

        # Create dataframe for display
        df_data = []
        for exp in experiments:
            df_data.append(
                {
                    "ID": exp["id"][:8] + "...",
                    "Session": exp["session_id"][:8] + "...",
                    "Status": exp["status"],
                    "Model": exp.get("results", {}).get("model", "N/A"),
                    "Accuracy": exp.get("results", {}).get("accuracy", 0),
                    "Created": exp["created_at"][:19],
                    "Duration": calculate_duration(exp),
                    "Full_ID": exp["id"],
                }
            )

        df = pd.DataFrame(df_data)

        # Configure grid options
        selected_rows = st.dataframe(
            df, use_container_width=True, hide_index=True, selection_mode="multi-select"
        )

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 View Details", disabled=len(selected_rows) != 1):
                if selected_rows:
                    view_experiment_details(
                        api_client, df.iloc[selected_rows[0]]["Full_ID"]
                    )

        with col2:
            if st.button("📊 Compare Selected", disabled=len(selected_rows) < 2):
                if len(selected_rows) >= 2:
                    selected_ids = [df.iloc[i]["Full_ID"] for i in selected_rows]
                    st.session_state["compare_experiments"] = selected_ids
                    st.rerun()

        with col3:
            if st.button("🗑️ Delete Selected", disabled=len(selected_rows) == 0):
                if selected_rows:
                    for idx in selected_rows:
                        api_client.delete_experiment(df.iloc[idx]["Full_ID"])
                    st.success(f"Deleted {len(selected_rows)} experiments")
                    st.rerun()

        # Visualization
        if len(experiments) > 0:
            st.divider()
            render_experiments_visualization(experiments)
    else:
        st.info("No experiments found. Run an AutoML workflow to create experiments!")


def render_experiments_comparison(api_client):
    """Render experiment comparison view"""
    st.subheader("Compare Experiments")

    # Get experiments for selection
    experiments_data = api_client.get_experiments()

    if not experiments_data or not experiments_data.get("experiments"):
        st.info("No experiments available for comparison")
        return

    experiments = experiments_data["experiments"]

    # Create selection options
    options = {
        f"{exp['id'][:8]}... - {exp.get('results', {}).get('model', 'Unknown')} ({exp.get('results', {}).get('accuracy', 0):.1%})": exp[
            "id"
        ]
        for exp in experiments
        if exp["status"] == "completed"
    }

    # Multi-select for experiments
    selected_names = st.multiselect(
        "Select experiments to compare (2-10)",
        list(options.keys()),
        default=list(options.keys())[:2] if len(options) >= 2 else list(options.keys()),
        max_selections=10,
    )

    if len(selected_names) < 2:
        st.warning("Select at least 2 experiments to compare")
        return

    selected_ids = [options[name] for name in selected_names]

    # Comparison metric
    metric = st.selectbox(
        "Comparison Metric",
        ["accuracy", "precision", "recall", "f1", "training_time"],
        key="comparison_metric",
    )

    # Get comparison data
    comparison_data = api_client.compare_experiments(selected_ids, metric)

    if comparison_data:
        # Display comparison results
        render_comparison_results(comparison_data, metric)


def render_comparison_results(comparison_data: Dict[str, Any], metric: str):
    """Render comparison results"""
    st.divider()

    # Best experiment
    if comparison_data.get("best_experiment"):
        best = comparison_data["best_experiment"]
        st.success(
            f"🏆 Best performing model: **{best.get('model', 'Unknown')}** with {metric}: {best['metrics'][metric]:.3f}"
        )

    # Metrics comparison chart
    st.subheader("Metrics Comparison")

    # Prepare data for visualization
    experiments = comparison_data["experiments"]
    metrics_to_show = ["accuracy", "precision", "recall", "f1"]

    # Create bar chart
    fig = go.Figure()

    for exp in experiments:
        if exp.get("metrics"):
            values = [exp["metrics"].get(m, 0) for m in metrics_to_show]
            fig.add_trace(
                go.Bar(
                    name=f"{exp['id'][:8]}... - {exp.get('parameters', {}).get('model', 'Unknown')}",
                    x=metrics_to_show,
                    y=values,
                )
            )

    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Parameters comparison
    st.subheader("Parameters Comparison")

    # Create parameters dataframe
    params_data = []
    for exp in experiments:
        if exp.get("parameters"):
            params = exp["parameters"].copy()
            params["Experiment"] = f"{exp['id'][:8]}..."
            params_data.append(params)

    if params_data:
        df_params = pd.DataFrame(params_data)
        # Move Experiment column to first position
        cols = ["Experiment"] + [
            col for col in df_params.columns if col != "Experiment"
        ]
        df_params = df_params[cols]

        st.dataframe(df_params, use_container_width=True, hide_index=True)

    # Training time comparison
    if any(exp.get("metrics", {}).get("training_time") for exp in experiments):
        st.subheader("Training Time Comparison")

        times = []
        names = []
        for exp in experiments:
            if exp.get("metrics", {}).get("training_time"):
                times.append(exp["metrics"]["training_time"])
                names.append(f"{exp['id'][:8]}...")

        fig = px.bar(
            x=names,
            y=times,
            labels={"x": "Experiment", "y": "Training Time (seconds)"},
            title="",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_experiments_visualization(experiments: List[Dict[str, Any]]):
    """Render experiments visualization"""
    col1, col2 = st.columns(2)

    with col1:
        # Model distribution
        st.write("**Model Distribution**")

        model_counts = {}
        for exp in experiments:
            model = exp.get("results", {}).get("model", "Unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

        if model_counts:
            fig = px.pie(
                values=list(model_counts.values()),
                names=list(model_counts.keys()),
                title="",
            )
            fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Status distribution
        st.write("**Status Distribution**")

        status_counts = {}
        for exp in experiments:
            status = exp["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        if status_counts:
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="",
                color_discrete_map={
                    "completed": "#4CAF50",
                    "running": "#2196F3",
                    "failed": "#F44336",
                    "pending": "#FFC107",
                },
            )
            fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)


def render_mlflow_integration():
    """Render MLflow integration information"""
    st.subheader("MLflow Integration")

    st.info("MLflow UI is available at http://localhost:5000")

    st.markdown(
        """
    ### Features Available in MLflow:
    
    - **Experiment Tracking**: All experiments are automatically logged
    - **Model Registry**: Register and version your best models
    - **Artifact Storage**: Access trained models, plots, and datasets
    - **Metric Visualization**: Compare runs with interactive charts
    - **Parameter Search**: Filter experiments by parameters
    
    ### How to Access:
    
    1. Open [http://localhost:5000](http://localhost:5000) in your browser
    2. Select your experiment from the list
    3. Click on runs to see details
    4. Compare multiple runs using the comparison feature
    
    ### Logged Information:
    
    - **Parameters**: All model hyperparameters
    - **Metrics**: Accuracy, precision, recall, F1, etc.
    - **Artifacts**: Trained models, confusion matrices, feature importance
    - **Tags**: Problem type, dataset info, preprocessing steps
    """
    )

    # Add button to open MLflow
    if st.button("🚀 Open MLflow UI", type="primary"):
        st.markdown("[Click here to open MLflow](http://localhost:5000)")


def view_experiment_details(api_client, experiment_id: str):
    """View detailed experiment information"""
    with st.expander("Experiment Details", expanded=True):
        details = api_client.get_experiment_detail(experiment_id)

        if details:
            # Basic info
            st.write(f"**ID**: {details['id']}")
            st.write(f"**Session**: {details['session_id']}")
            st.write(f"**Status**: {details['status']}")
            st.write(f"**Created**: {details['created_at']}")

            if details["completed_at"]:
                st.write(f"**Completed**: {details['completed_at']}")

            # Metrics
            if details.get("metrics"):
                st.subheader("Metrics")
                metrics_df = pd.DataFrame([details["metrics"]])
                st.dataframe(metrics_df, use_container_width=True)

            # Parameters
            if details.get("parameters"):
                st.subheader("Parameters")
                params_df = pd.DataFrame([details["parameters"]])
                st.dataframe(params_df, use_container_width=True)

            # Artifacts
            if details.get("artifacts"):
                st.subheader("Artifacts")
                for artifact in details["artifacts"]:
                    st.write(f"- {artifact}")


def calculate_duration(experiment: Dict[str, Any]) -> str:
    """Calculate experiment duration"""
    if experiment.get("completed_at") and experiment.get("created_at"):
        try:
            start = datetime.fromisoformat(
                experiment["created_at"].replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                experiment["completed_at"].replace("Z", "+00:00")
            )
            duration = end - start

            # Format duration
            if duration.total_seconds() < 60:
                return f"{int(duration.total_seconds())}s"
            elif duration.total_seconds() < 3600:
                return f"{int(duration.total_seconds() / 60)}m"
            else:
                return f"{duration.total_seconds() / 3600:.1f}h"
        except:
            pass

    return "N/A"
