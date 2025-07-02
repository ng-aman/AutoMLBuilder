# src/api/routers/datasets.py
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from src.api.dependencies.database import get_db
from src.api.dependencies.auth import get_current_user
from src.api.models.user import User
from src.api.models.dataset import Dataset
from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
    ResourceNotFoundError,
    ValidationError,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/datasets")

# Allowed file extensions
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


# Pydantic models
class DatasetResponse(BaseModel):
    """Dataset response model"""

    id: str
    filename: str
    file_size: int
    rows_count: Optional[int]
    columns_count: Optional[int]
    metadata: Optional[dict]
    created_at: str


class DatasetListResponse(BaseModel):
    """Dataset list response"""

    datasets: List[DatasetResponse]
    total: int


class DatasetInfoResponse(BaseModel):
    """Detailed dataset info"""

    id: str
    filename: str
    file_size: int
    rows_count: int
    columns_count: int
    columns: List[dict]
    sample_data: List[dict]
    statistics: dict
    missing_values: dict
    created_at: str


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(file_ext, list(ALLOWED_EXTENSIONS))

    # Check file size (this is approximate, actual check happens during upload)
    if file.size and file.size > settings.max_upload_size_bytes:
        raise FileTooLargeError(file.size, settings.max_upload_size_bytes)


def analyze_dataset(file_path: str, file_ext: str) -> dict:
    """Analyze dataset and extract metadata"""
    try:
        # Load dataset based on file type
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Basic info
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items() if v > 0}

        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_dict()
            # Convert numpy types to native Python types
            for col in stats:
                for stat in stats[col]:
                    if hasattr(stats[col][stat], "item"):
                        stats[col][stat] = stats[col][stat].item()
        else:
            stats = {}

        # Sample data (first 5 rows)
        sample = df.head(5).to_dict(orient="records")

        return {
            "info": info,
            "missing_values": missing_values,
            "statistics": stats,
            "sample_data": sample,
        }

    except Exception as e:
        logger.error(f"Dataset analysis error", error=str(e), file_path=file_path)
        raise ValidationError(f"Failed to analyze dataset: {str(e)}")


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a dataset"""
    # Validate file
    validate_file(file)

    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.upload_dir) / str(current_user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_ext = Path(file.filename).suffix.lower()
    safe_filename = f"{timestamp}_{Path(file.filename).stem}{file_ext}"
    file_path = upload_dir / safe_filename

    try:
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()

            # Check actual file size
            if len(content) > settings.max_upload_size_bytes:
                os.remove(file_path)
                raise FileTooLargeError(len(content), settings.max_upload_size_bytes)

            f.write(content)

        # Analyze dataset
        analysis = analyze_dataset(str(file_path), file_ext)

        # Create dataset record
        dataset = Dataset(
            user_id=current_user.id,
            filename=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            rows_count=analysis["info"]["rows"],
            columns_count=analysis["info"]["columns"],
            metadata={
                "original_filename": file.filename,
                "file_type": file_ext,
                "column_names": analysis["info"]["column_names"],
                "column_types": analysis["info"]["column_types"],
                "missing_values": analysis["missing_values"],
                "upload_timestamp": timestamp,
            },
        )

        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        logger.info(
            "Dataset uploaded",
            user_id=str(current_user.id),
            dataset_id=str(dataset.id),
            filename=file.filename,
            size=len(content),
        )

        return DatasetResponse(
            id=str(dataset.id),
            filename=dataset.filename,
            file_size=dataset.file_size,
            rows_count=dataset.rows_count,
            columns_count=dataset.columns_count,
            metadata=dataset.metadata,
            created_at=dataset.created_at.isoformat(),
        )

    except Exception as e:
        # Clean up file if upload failed
        if file_path.exists():
            os.remove(file_path)

        logger.error("Dataset upload failed", error=str(e))
        if isinstance(e, (FileTooLargeError, InvalidFileTypeError, ValidationError)):
            raise e
        raise HTTPException(status_code=500, detail="Failed to upload dataset")


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List user's datasets"""
    # Get total count
    total = db.query(Dataset).filter(Dataset.user_id == current_user.id).count()

    # Get datasets
    datasets = (
        db.query(Dataset)
        .filter(Dataset.user_id == current_user.id)
        .order_by(Dataset.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return DatasetListResponse(
        datasets=[
            DatasetResponse(
                id=str(d.id),
                filename=d.filename,
                file_size=d.file_size,
                rows_count=d.rows_count,
                columns_count=d.columns_count,
                metadata=d.metadata,
                created_at=d.created_at.isoformat(),
            )
            for d in datasets
        ],
        total=total,
    )


@router.get("/{dataset_id}", response_model=DatasetInfoResponse)
async def get_dataset_info(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed dataset information"""
    # Get dataset
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
        .first()
    )

    if not dataset:
        raise ResourceNotFoundError("Dataset", dataset_id)

    # Re-analyze dataset for detailed info
    try:
        analysis = analyze_dataset(
            dataset.file_path, dataset.metadata.get("file_type", ".csv")
        )

        # Prepare column info
        columns = []
        for col_name in analysis["info"]["column_names"]:
            columns.append(
                {
                    "name": col_name,
                    "type": analysis["info"]["column_types"].get(col_name, "unknown"),
                    "missing_count": analysis["missing_values"].get(col_name, 0),
                    "missing_percentage": (
                        analysis["missing_values"].get(col_name, 0)
                        / analysis["info"]["rows"]
                        * 100
                        if analysis["info"]["rows"] > 0
                        else 0
                    ),
                }
            )

        return DatasetInfoResponse(
            id=str(dataset.id),
            filename=dataset.filename,
            file_size=dataset.file_size,
            rows_count=analysis["info"]["rows"],
            columns_count=analysis["info"]["columns"],
            columns=columns,
            sample_data=analysis["sample_data"],
            statistics=analysis["statistics"],
            missing_values=analysis["missing_values"],
            created_at=dataset.created_at.isoformat(),
        )

    except Exception as e:
        logger.error("Failed to analyze dataset", dataset_id=dataset_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to analyze dataset")


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a dataset"""
    # Get dataset
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
        .first()
    )

    if not dataset:
        raise ResourceNotFoundError("Dataset", dataset_id)

    # Delete file
    try:
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
            logger.info("Deleted dataset file", path=dataset.file_path)
    except Exception as e:
        logger.error(
            "Failed to delete dataset file", path=dataset.file_path, error=str(e)
        )

    # Delete database record
    db.delete(dataset)
    db.commit()

    logger.info("Dataset deleted", dataset_id=dataset_id, user_id=str(current_user.id))

    return {"message": "Dataset deleted successfully"}


@router.post("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    rows: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Preview dataset contents"""
    # Get dataset
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
        .first()
    )

    if not dataset:
        raise ResourceNotFoundError("Dataset", dataset_id)

    try:
        # Load dataset
        file_ext = dataset.metadata.get("file_type", ".csv")

        if file_ext == ".csv":
            df = pd.read_csv(dataset.file_path, nrows=rows)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(dataset.file_path, nrows=rows)
        elif file_ext == ".json":
            df = pd.read_json(dataset.file_path)
            df = df.head(rows)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        return {
            "rows": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "shape": list(df.shape),
        }

    except Exception as e:
        logger.error("Failed to preview dataset", dataset_id=dataset_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to preview dataset")
