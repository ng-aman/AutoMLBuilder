"""
File uploader component for AutoML Builder.

This module provides a comprehensive file upload interface for datasets
with validation, preview, and metadata extraction capabilities.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Tuple, Callable
import pandas as pd
import json
import io
import asyncio
from pathlib import Path
from datetime import datetime
import hashlib

from ..utils.api_client import APIClient
from ...core.config import settings


class FileUploader:
    """
    Streamlit file uploader component.

    Provides an interface for uploading datasets with validation,
    preview, and automatic metadata extraction.
    """

    def __init__(
        self,
        api_client: APIClient,
        session_state: Any,
        on_upload_callback: Optional[Callable] = None,
    ):
        """
        Initialize file uploader.

        Args:
            api_client: API client instance
            session_state: Streamlit session state
            on_upload_callback: Optional callback after successful upload
        """
        self.api_client = api_client
        self.session_state = session_state
        self.on_upload_callback = on_upload_callback

        # Configuration
        self.max_file_size_mb = settings.files.max_upload_size_mb
        self.allowed_extensions = settings.files.allowed_extensions

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "uploaded_datasets" not in self.session_state:
            self.session_state.uploaded_datasets = []

        if "current_dataset" not in self.session_state:
            self.session_state.current_dataset = None

        if "dataset_preview" not in self.session_state:
            self.session_state.dataset_preview = None

        if "upload_status" not in self.session_state:
            self.session_state.upload_status = None

    def render(self):
        """Render the file uploader interface."""
        # Upload section
        self._render_upload_section()

        # Dataset list
        if self.session_state.uploaded_datasets:
            self._render_dataset_list()

        # Dataset preview
        if self.session_state.current_dataset:
            self._render_dataset_preview()

    def _render_upload_section(self):
        """Render the upload section."""
        st.markdown("### 📁 Upload Dataset")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[ext.lstrip(".") for ext in self.allowed_extensions],
            help=f"Supported formats: {', '.join(self.allowed_extensions)}. Max size: {self.max_file_size_mb}MB",
            key="file_uploader_widget",
        )

        # Upload options
        col1, col2 = st.columns(2)

        with col1:
            auto_detect_target = st.checkbox(
                "Auto-detect target variable",
                value=True,
                help="Automatically identify the target variable for ML",
            )

        with col2:
            validate_data = st.checkbox(
                "Validate data quality",
                value=True,
                help="Check for missing values, duplicates, and data types",
            )

        # Handle file upload
        if uploaded_file is not None:
            self._handle_file_upload(
                uploaded_file,
                auto_detect_target=auto_detect_target,
                validate_data=validate_data,
            )

    def _handle_file_upload(
        self, uploaded_file, auto_detect_target: bool = True, validate_data: bool = True
    ):
        """
        Handle file upload process.

        Args:
            uploaded_file: Streamlit uploaded file object
            auto_detect_target: Whether to auto-detect target variable
            validate_data: Whether to validate data quality
        """
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            st.error(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({self.max_file_size_mb}MB)"
            )
            return

        # Show upload progress
        with st.spinner("Uploading and processing file..."):
            try:
                # Read file content
                file_content, file_df = self._read_file(uploaded_file)

                if file_df is None:
                    st.error("Unable to read file. Please check the format.")
                    return

                # Generate file hash
                file_hash = hashlib.md5(file_content).hexdigest()

                # Check if already uploaded
                if any(
                    d["hash"] == file_hash for d in self.session_state.uploaded_datasets
                ):
                    st.warning("This file has already been uploaded.")
                    return

                # Extract metadata
                metadata = self._extract_metadata(
                    uploaded_file, file_df, file_hash, auto_detect_target, validate_data
                )

                # Upload to backend
                dataset_id = asyncio.run(
                    self._upload_to_backend(uploaded_file, file_content, metadata)
                )

                if dataset_id:
                    # Add to uploaded datasets
                    dataset_info = {
                        "id": dataset_id,
                        "name": uploaded_file.name,
                        "hash": file_hash,
                        "metadata": metadata,
                        "upload_time": datetime.now(),
                    }
                    self.session_state.uploaded_datasets.append(dataset_info)
                    self.session_state.current_dataset = dataset_info
                    self.session_state.dataset_preview = file_df

                    # Show success message
                    st.success(f"✅ Successfully uploaded: {uploaded_file.name}")

                    # Trigger callback
                    if self.on_upload_callback:
                        self.on_upload_callback(dataset_info)
                else:
                    st.error("Failed to upload file. Please try again.")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    def _read_file(self, uploaded_file) -> Tuple[bytes, Optional[pd.DataFrame]]:
        """
        Read uploaded file and return content and DataFrame.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Tuple of (file content bytes, DataFrame or None)
        """
        # Get file content
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for reading again

        # Try to read as DataFrame
        df = None
        file_extension = Path(uploaded_file.name).suffix.lower()

        try:
            if file_extension == ".csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_extension == ".json":
                df = pd.read_json(uploaded_file)
            elif file_extension == ".parquet":
                df = pd.read_parquet(uploaded_file)
        except Exception as e:
            print(f"Error reading file: {e}")

        uploaded_file.seek(0)  # Reset again
        return file_content, df

    def _extract_metadata(
        self,
        uploaded_file,
        df: pd.DataFrame,
        file_hash: str,
        auto_detect_target: bool,
        validate_data: bool,
    ) -> Dict[str, Any]:
        """
        Extract metadata from uploaded file.

        Args:
            uploaded_file: Uploaded file object
            df: DataFrame of the file content
            file_hash: MD5 hash of file
            auto_detect_target: Whether to auto-detect target
            validate_data: Whether to validate data

        Returns:
            Dictionary containing metadata
        """
        metadata = {
            "file_name": uploaded_file.name,
            "file_size": uploaded_file.size,
            "file_hash": file_hash,
            "file_type": Path(uploaded_file.name).suffix.lower(),
            "upload_time": datetime.now().isoformat(),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        # Basic statistics
        metadata["statistics"] = {
            "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
            "categorical_columns": list(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime_columns": list(df.select_dtypes(include=["datetime"]).columns),
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict(),
        }

        # Data quality metrics
        if validate_data:
            metadata["quality"] = {
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "duplicate_percentage": df.duplicated().sum() / len(df) * 100,
                "constant_columns": [
                    col for col in df.columns if df[col].nunique() == 1
                ],
                "high_cardinality_columns": [
                    col
                    for col in df.select_dtypes(include=["object"]).columns
                    if df[col].nunique() / len(df) > 0.9
                ],
            }

        # Auto-detect target variable
        if auto_detect_target:
            target_candidates = self._detect_target_variable(df)
            metadata["target_candidates"] = target_candidates

        # Sample data
        metadata["sample"] = df.head(10).to_dict(orient="records")

        return metadata

    def _detect_target_variable(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Auto-detect potential target variables.

        Args:
            df: DataFrame to analyze

        Returns:
            List of target variable candidates with scores
        """
        candidates = []

        for col in df.columns:
            score = 0
            reasons = []

            # Check column name patterns
            target_keywords = ["target", "label", "class", "y", "outcome", "result"]
            if any(keyword in col.lower() for keyword in target_keywords):
                score += 30
                reasons.append("Column name suggests target variable")

            # Check if it's the last column
            if col == df.columns[-1]:
                score += 10
                reasons.append("Last column in dataset")

            # Check data characteristics
            unique_ratio = df[col].nunique() / len(df)

            # Binary classification candidate
            if df[col].nunique() == 2:
                score += 20
                reasons.append("Binary variable")

            # Multi-class classification candidate
            elif 2 < df[col].nunique() <= 20 and df[col].dtype == "object":
                score += 15
                reasons.append("Categorical with few unique values")

            # Regression candidate
            elif df[col].dtype in ["int64", "float64"] and 0.1 < unique_ratio < 0.9:
                score += 10
                reasons.append("Numeric with moderate uniqueness")

            if score > 0:
                candidates.append(
                    {
                        "column": col,
                        "score": score,
                        "reasons": reasons,
                        "dtype": str(df[col].dtype),
                        "unique_values": df[col].nunique(),
                        "null_count": df[col].isnull().sum(),
                    }
                )

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:5]  # Return top 5 candidates

    async def _upload_to_backend(
        self, uploaded_file, file_content: bytes, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Upload file to backend API.

        Args:
            uploaded_file: Uploaded file object
            file_content: File content as bytes
            metadata: File metadata

        Returns:
            Dataset ID if successful, None otherwise
        """
        try:
            # Prepare multipart upload
            files = {
                "file": (
                    uploaded_file.name,
                    io.BytesIO(file_content),
                    uploaded_file.type,
                )
            }

            data = {"metadata": json.dumps(metadata)}

            # Upload file
            response = await self.api_client.post_async(
                "/api/datasets/upload", files=files, data=data
            )

            return response.get("dataset_id")

        except Exception as e:
            print(f"Upload error: {e}")
            return None

    def _render_dataset_list(self):
        """Render list of uploaded datasets."""
        st.markdown("### 📊 Uploaded Datasets")

        for dataset in self.session_state.uploaded_datasets:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    st.markdown(f"**{dataset['name']}**")
                    st.caption(f"ID: {dataset['id'][:8]}...")

                with col2:
                    shape = dataset["metadata"]["shape"]
                    st.text(f"{shape[0]:,} rows × {shape[1]} cols")

                with col3:
                    upload_time = dataset["upload_time"]
                    st.text(upload_time.strftime("%Y-%m-%d %H:%M"))

                with col4:
                    if st.button("View", key=f"view_{dataset['id']}"):
                        self._load_dataset(dataset)

                st.markdown("---")

    def _load_dataset(self, dataset: Dict[str, Any]):
        """
        Load dataset for preview.

        Args:
            dataset: Dataset information
        """
        self.session_state.current_dataset = dataset

        # Fetch preview data from backend
        try:
            response = self.api_client.get(f"/api/datasets/{dataset['id']}/preview")
            df = pd.DataFrame(response["data"])
            self.session_state.dataset_preview = df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    def _render_dataset_preview(self):
        """Render dataset preview and information."""
        dataset = self.session_state.current_dataset
        df = self.session_state.dataset_preview

        st.markdown(f"### 🔍 Dataset Preview: {dataset['name']}")

        # Dataset info tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Schema", "Statistics", "Quality"])

        with tab1:
            # Data preview
            st.dataframe(df.head(100), use_container_width=True)

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Sample (CSV)",
                    data=df.head(1000).to_csv(index=False),
                    file_name=f"sample_{dataset['name']}",
                    mime="text/csv",
                )

        with tab2:
            # Schema information
            schema_df = pd.DataFrame(
                {
                    "Column": dataset["metadata"]["columns"],
                    "Type": dataset["metadata"]["dtypes"].values(),
                    "Non-Null": [
                        len(df) - dataset["metadata"]["statistics"]["null_counts"][col]
                        for col in dataset["metadata"]["columns"]
                    ],
                    "Unique": [
                        dataset["metadata"]["statistics"]["unique_counts"][col]
                        for col in dataset["metadata"]["columns"]
                    ],
                }
            )
            st.dataframe(schema_df, use_container_width=True)

        with tab3:
            # Statistics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Numeric Columns**")
                numeric_cols = dataset["metadata"]["statistics"]["numeric_columns"]
                if numeric_cols and df is not None:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found")

            with col2:
                st.markdown("**Categorical Columns**")
                cat_cols = dataset["metadata"]["statistics"]["categorical_columns"]
                if cat_cols:
                    for col in cat_cols[:5]:  # Show first 5
                        unique_count = dataset["metadata"]["statistics"][
                            "unique_counts"
                        ][col]
                        st.text(f"{col}: {unique_count} unique values")
                else:
                    st.info("No categorical columns found")

        with tab4:
            # Data quality report
            quality = dataset["metadata"].get("quality", {})

            # Quality metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                duplicate_pct = quality.get("duplicate_percentage", 0)
                st.metric(
                    "Duplicate Rows",
                    f"{duplicate_pct:.1f}%",
                    delta=(
                        None
                        if duplicate_pct == 0
                        else f"{quality.get('duplicate_rows', 0)} rows"
                    ),
                )

            with col2:
                const_cols = len(quality.get("constant_columns", []))
                st.metric(
                    "Constant Columns",
                    const_cols,
                    delta="Remove" if const_cols > 0 else None,
                )

            with col3:
                high_card = len(quality.get("high_cardinality_columns", []))
                st.metric(
                    "High Cardinality",
                    high_card,
                    delta="Review" if high_card > 0 else None,
                )

            # Missing data visualization
            st.markdown("**Missing Data Analysis**")
            missing_pct = quality.get("missing_percentage", {})
            if missing_pct:
                missing_df = pd.DataFrame(
                    {"Column": missing_pct.keys(), "Missing %": missing_pct.values()}
                ).sort_values("Missing %", ascending=False)

                # Only show columns with missing data
                missing_df = missing_df[missing_df["Missing %"] > 0]

                if not missing_df.empty:
                    st.bar_chart(missing_df.set_index("Column"))
                else:
                    st.success("No missing data found!")

        # Target variable selection
        if dataset["metadata"].get("target_candidates"):
            st.markdown("### 🎯 Target Variable Selection")

            candidates = dataset["metadata"]["target_candidates"]

            # Auto-suggestion
            if candidates:
                st.info(
                    f"Suggested target: **{candidates[0]['column']}** (confidence: {candidates[0]['score']}%)"
                )

            # Manual selection
            target_col = st.selectbox(
                "Select target variable",
                options=[None] + dataset["metadata"]["columns"],
                index=0,
                format_func=lambda x: "No target (unsupervised)" if x is None else x,
            )

            if st.button("Confirm Selection", type="primary"):
                self.session_state.dataset_id = dataset["id"]
                self.session_state.target_variable = target_col
                st.success(
                    f"Dataset ready for ML! Target: {target_col if target_col else 'None (unsupervised)'}"
                )


# Helper function for creating file uploader
def create_file_uploader(
    api_client: APIClient,
    session_state: Any,
    on_upload_callback: Optional[Callable] = None,
) -> FileUploader:
    """
    Create and return a file uploader instance.

    Args:
        api_client: API client instance
        session_state: Streamlit session state
        on_upload_callback: Optional callback after upload

    Returns:
        FileUploader instance
    """
    return FileUploader(api_client, session_state, on_upload_callback)
