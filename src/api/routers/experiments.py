# src/api/routers/experiments.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import mlflow
from datetime import datetime
from pydantic import BaseModel
from src.api.dependencies.database import get_db
from src.api.dependencies.auth import get_current_user
from src.api.models.user import User
from src.api.models.experiment import Experiment
from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import ResourceNotFoundError, ValidationError

logger = get_logger(__name__)

router = APIRouter(prefix="/api/experiments")


# Pydantic models
class ExperimentResponse(BaseModel):
    """Experiment response model"""

    id: str
    session_id: str
    mlflow_run_id: Optional[str]
    status: str
    results: Optional[dict]
    created_at: str
    completed_at: Optional[str]


class ExperimentListResponse(BaseModel):
    """Experiment list response"""

    experiments: List[ExperimentResponse]
    total: int


class ExperimentDetailResponse(BaseModel):
    """Detailed experiment response"""

    id: str
    session_id: str
    mlflow_run_id: Optional[str]
    status: str
    results: Optional[dict]
    metrics: Optional[dict]
    parameters: Optional[dict]
    artifacts: Optional[List[str]]
    created_at: str
    completed_at: Optional[str]


class ExperimentComparisonResponse(BaseModel):
    """Experiment comparison response"""

    experiments: List[dict]
    metrics_comparison: dict
    best_experiment: Optional[dict]


# Initialize MLflow
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def get_mlflow_run_details(run_id: str) -> Dict[str, Any]:
    """Get details from MLflow run"""
    try:
        run = mlflow.get_run(run_id)

        # Get metrics
        metrics = {}
        for key, value in run.data.metrics.items():
            metrics[key] = value

        # Get parameters
        parameters = {}
        for key, value in run.data.params.items():
            parameters[key] = value

        # Get artifacts
        artifacts = []
        artifact_uri = run.info.artifact_uri
        if artifact_uri:
            try:
                artifacts = mlflow.list_artifacts(run_id)
                artifacts = [a.path for a in artifacts]
            except:
                pass

        return {
            "metrics": metrics,
            "parameters": parameters,
            "artifacts": artifacts,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "tags": run.data.tags,
        }
    except Exception as e:
        logger.error(f"Failed to get MLflow run details", run_id=run_id, error=str(e))
        return {}


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List experiments"""
    # Build query
    query = (
        db.query(Experiment)
        .join(Experiment.session)
        .filter(Experiment.session.has(user_id=current_user.id))
    )

    if session_id:
        query = query.filter(Experiment.session_id == session_id)

    if status:
        query = query.filter(Experiment.status == status)

    # Get total count
    total = query.count()

    # Get experiments
    experiments = (
        query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()
    )

    return ExperimentListResponse(
        experiments=[
            ExperimentResponse(
                id=str(e.id),
                session_id=str(e.session_id),
                mlflow_run_id=e.mlflow_run_id,
                status=e.status,
                results=e.results,
                created_at=e.created_at.isoformat(),
                completed_at=e.completed_at.isoformat() if e.completed_at else None,
            )
            for e in experiments
        ],
        total=total,
    )


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment_detail(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed experiment information"""
    # Get experiment
    experiment = (
        db.query(Experiment)
        .join(Experiment.session)
        .filter(
            Experiment.id == experiment_id,
            Experiment.session.has(user_id=current_user.id),
        )
        .first()
    )

    if not experiment:
        raise ResourceNotFoundError("Experiment", experiment_id)

    # Get MLflow details if available
    mlflow_details = {}
    if experiment.mlflow_run_id:
        mlflow_details = get_mlflow_run_details(experiment.mlflow_run_id)

    return ExperimentDetailResponse(
        id=str(experiment.id),
        session_id=str(experiment.session_id),
        mlflow_run_id=experiment.mlflow_run_id,
        status=experiment.status,
        results=experiment.results,
        metrics=mlflow_details.get("metrics"),
        parameters=mlflow_details.get("parameters"),
        artifacts=mlflow_details.get("artifacts"),
        created_at=experiment.created_at.isoformat(),
        completed_at=(
            experiment.completed_at.isoformat() if experiment.completed_at else None
        ),
    )


@router.post("/compare", response_model=ExperimentComparisonResponse)
async def compare_experiments(
    experiment_ids: List[str],
    metric: str = "accuracy",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compare multiple experiments"""
    if len(experiment_ids) < 2:
        raise ValidationError("At least 2 experiments required for comparison")

    if len(experiment_ids) > 10:
        raise ValidationError("Maximum 10 experiments can be compared at once")

    # Get experiments
    experiments = (
        db.query(Experiment)
        .join(Experiment.session)
        .filter(
            Experiment.id.in_(experiment_ids),
            Experiment.session.has(user_id=current_user.id),
        )
        .all()
    )

    if len(experiments) != len(experiment_ids):
        raise ValidationError("Some experiments not found or not accessible")

    # Prepare comparison data
    comparison_data = []
    metrics_data = {}
    best_experiment = None
    best_metric_value = None

    for exp in experiments:
        exp_data = {
            "id": str(exp.id),
            "session_id": str(exp.session_id),
            "status": exp.status,
            "created_at": exp.created_at.isoformat(),
        }

        # Get MLflow details
        if exp.mlflow_run_id:
            mlflow_details = get_mlflow_run_details(exp.mlflow_run_id)
            exp_data["metrics"] = mlflow_details.get("metrics", {})
            exp_data["parameters"] = mlflow_details.get("parameters", {})

            # Collect metrics for comparison
            for metric_name, value in exp_data["metrics"].items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(
                    {"experiment_id": str(exp.id), "value": value}
                )

            # Track best experiment
            if metric in exp_data["metrics"]:
                metric_value = exp_data["metrics"][metric]
                if best_metric_value is None or metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_experiment = exp_data

        comparison_data.append(exp_data)

    return ExperimentComparisonResponse(
        experiments=comparison_data,
        metrics_comparison=metrics_data,
        best_experiment=best_experiment,
    )


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an experiment"""
    # Get experiment
    experiment = (
        db.query(Experiment)
        .join(Experiment.session)
        .filter(
            Experiment.id == experiment_id,
            Experiment.session.has(user_id=current_user.id),
        )
        .first()
    )

    if not experiment:
        raise ResourceNotFoundError("Experiment", experiment_id)

    # Delete MLflow run if exists
    if experiment.mlflow_run_id:
        try:
            mlflow.delete_run(experiment.mlflow_run_id)
            logger.info("Deleted MLflow run", run_id=experiment.mlflow_run_id)
        except Exception as e:
            logger.error(
                "Failed to delete MLflow run",
                run_id=experiment.mlflow_run_id,
                error=str(e),
            )

    # Delete database record
    db.delete(experiment)
    db.commit()

    logger.info("Experiment deleted", experiment_id=experiment_id)

    return {"message": "Experiment deleted successfully"}


@router.get("/{experiment_id}/artifacts/{artifact_path:path}")
async def get_experiment_artifact(
    experiment_id: str,
    artifact_path: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download experiment artifact"""
    # Get experiment
    experiment = (
        db.query(Experiment)
        .join(Experiment.session)
        .filter(
            Experiment.id == experiment_id,
            Experiment.session.has(user_id=current_user.id),
        )
        .first()
    )

    if not experiment:
        raise ResourceNotFoundError("Experiment", experiment_id)

    if not experiment.mlflow_run_id:
        raise ValidationError("Experiment has no MLflow run")

    try:
        # Download artifact from MLflow
        import tempfile
        import os
        from fastapi.responses import FileResponse

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download artifact
            local_path = mlflow.artifacts.download_artifacts(
                run_id=experiment.mlflow_run_id,
                artifact_path=artifact_path,
                dst_path=tmp_dir,
            )

            if os.path.isfile(local_path):
                return FileResponse(
                    local_path, filename=os.path.basename(artifact_path)
                )
            else:
                raise ValidationError("Artifact is not a file")

    except Exception as e:
        logger.error(
            "Failed to download artifact",
            experiment_id=experiment_id,
            artifact_path=artifact_path,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to download artifact")


@router.post("/{experiment_id}/rerun")
async def rerun_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rerun an experiment with same parameters"""
    # Get experiment
    experiment = (
        db.query(Experiment)
        .join(Experiment.session)
        .filter(
            Experiment.id == experiment_id,
            Experiment.session.has(user_id=current_user.id),
        )
        .first()
    )

    if not experiment:
        raise ResourceNotFoundError("Experiment", experiment_id)

    # Create new experiment
    new_experiment = Experiment(
        session_id=experiment.session_id, status="pending", results=None
    )
    db.add(new_experiment)
    db.commit()
    db.refresh(new_experiment)

    # TODO: Add background task to rerun the experiment
    # This would involve recreating the workflow with the same parameters

    logger.info(
        "Experiment rerun initiated",
        original_id=experiment_id,
        new_id=str(new_experiment.id),
    )

    return {
        "message": "Experiment rerun initiated",
        "new_experiment_id": str(new_experiment.id),
    }
