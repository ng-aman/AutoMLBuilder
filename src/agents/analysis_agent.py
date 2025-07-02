# src/agents/analysis_agent.py
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from langchain.schema import BaseMessage, HumanMessage
from langchain.tools import Tool
from src.agents.base_agent import BaseAgent
from src.agents.tools.data_tools import (
    DataProfilerTool,
    DataQualityTool,
    TargetAnalyzerTool,
    FeatureAnalyzerTool,
)
from src.agents.tools.visualization_tools import VisualizationTools, plot_eda_report
from src.core.config import AGENT_PROMPTS
from src.core.state import (
    ConversationState,
    AgentType,
    ProblemType,
    DatasetInfo,
    add_message,
    add_warning,
    update_state_timestamp,
)
from src.utils.logger import get_logger
import json

logger = get_logger(__name__)


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for exploratory data analysis (EDA),
    problem type identification, and initial data assessment.
    """

    def __init__(self):
        # Initialize tools
        tools = [
            DataProfilerTool(),
            DataQualityTool(),
            TargetAnalyzerTool(),
            FeatureAnalyzerTool(),
        ]

        super().__init__(
            name="AnalysisAgent",
            agent_type=AgentType.ANALYSIS,
            tools=tools,
            system_prompt=AGENT_PROMPTS["analysis"],
        )

        # Initialize visualization tools
        self.viz_tools = VisualizationTools(save_to_mlflow=True)

    def _get_default_prompt(self) -> str:
        """Get default system prompt"""
        return AGENT_PROMPTS["analysis"]

    async def _execute(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Tuple[ConversationState, str]:
        """Execute analysis workflow"""

        # Check if dataset is available
        if not state["dataset_id"]:
            return state, "Please upload a dataset first before I can perform analysis."

        # Record initial thought
        state = self._record_thought(
            state,
            "Starting dataset analysis to understand data structure and identify problem type",
        )

        try:
            # Step 1: Profile the dataset
            profile_result = await self._profile_dataset(state)
            state["dataset_info"] = profile_result["dataset_info"]

            # Step 2: Assess data quality
            quality_result = await self._assess_data_quality(state)

            # Step 3: Identify target variable and problem type
            if not state["target_variable"]:
                target_result = await self._identify_target_variable(state, messages)
                state["target_variable"] = target_result["target_variable"]
                state["problem_type"] = target_result["problem_type"]

            # Step 4: Analyze features
            feature_result = await self._analyze_features(state)
            state["feature_importance"] = feature_result.get("feature_importance", {})

            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(
                profile_result, quality_result, feature_result
            )

            # Step 6: Create visualizations
            visualizations = await self._create_visualizations(
                state, quality_result, feature_result
            )

            # Store visualization references in state
            if visualizations:
                state["analysis_visualizations"] = {
                    name: fig.to_json() for name, fig in visualizations.items()
                }
            else:
                state["analysis_visualizations"] = {}

            # Create summary message
            summary = self._create_analysis_summary(
                profile_result,
                quality_result,
                target_result if "target_result" in locals() else None,
                feature_result,
                recommendations,
                visualizations,
            )

            # Update state with analysis results
            state = add_message(
                state,
                "assistant",
                summary,
                metadata={
                    "agent": self.agent_type.value,
                    "analysis_complete": True,
                    "recommendations": recommendations,
                    "visualizations_created": list(visualizations.keys()),
                },
            )

            # Record completion
            state = self._record_thought(
                state,
                "Analysis completed successfully",
                decision=f"Problem type: {state['problem_type']}, Target: {state['target_variable']}",
            )

            return state, summary

        except Exception as e:
            logger.error("Analysis failed", error=str(e))
            error_message = f"Analysis encountered an error: {str(e)}"
            state = add_message(state, "assistant", error_message)
            state = self._record_thought(state, f"Analysis failed: {str(e)}")
            raise

    async def _profile_dataset(self, state: ConversationState) -> Dict[str, Any]:
        """Profile the dataset to get basic information"""
        profiler = self._get_tool_by_name("data_profiler")

        result = await self._use_tool(
            tool=profiler,
            tool_input={"dataset_id": state["dataset_id"]},
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        return result

    async def _assess_data_quality(self, state: ConversationState) -> Dict[str, Any]:
        """Assess data quality issues"""
        quality_tool = self._get_tool_by_name("data_quality")

        result = await self._use_tool(
            tool=quality_tool,
            tool_input={"dataset_info": state["dataset_info"]},
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        # Add warnings for critical issues
        if result.get("critical_issues"):
            for issue in result["critical_issues"]:
                state = add_warning(
                    state,
                    warning=issue,
                    agent=self.agent_type,
                    details={"source": "data_quality_assessment"},
                )

        return result

    async def _identify_target_variable(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Dict[str, Any]:
        """Identify target variable and problem type"""

        # First, check if user mentioned target in their message
        user_message = messages[-1].content if messages else ""

        # Create prompt for target identification
        prompt = f"""
Based on the dataset information and user request, identify the target variable and problem type.

Dataset columns: {state['dataset_info']['column_names']}
Column types: {state['dataset_info']['column_types']}
User message: {user_message}

Please identify:
1. The target variable (column name)
2. The problem type (classification/regression)
3. Your reasoning

Format your response as:
TARGET_VARIABLE: [column_name]
PROBLEM_TYPE: [classification/regression]
REASONING: [your explanation]
"""

        # Get LLM response
        response = await self._call_llm(
            messages + [HumanMessage(content=prompt)],
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        # Parse response
        result = self._parse_target_response(response.content)

        # Validate using tool
        analyzer = self._get_tool_by_name("target_analyzer")
        validation = await self._use_tool(
            tool=analyzer,
            tool_input={
                "dataset_info": state["dataset_info"],
                "target_variable": result["target_variable"],
            },
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        # Update result with validation
        result.update(validation)

        return result

    def _parse_target_response(self, response: str) -> Dict[str, Any]:
        """Parse target variable identification response"""
        result = {"target_variable": None, "problem_type": None, "reasoning": ""}

        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("TARGET_VARIABLE:"):
                result["target_variable"] = line.replace("TARGET_VARIABLE:", "").strip()
            elif line.startswith("PROBLEM_TYPE:"):
                problem_type_str = line.replace("PROBLEM_TYPE:", "").strip().lower()
                if problem_type_str in ["classification", "regression"]:
                    result["problem_type"] = ProblemType(problem_type_str)
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()

        return result

    async def _analyze_features(self, state: ConversationState) -> Dict[str, Any]:
        """Analyze feature characteristics and relationships"""
        analyzer = self._get_tool_by_name("feature_analyzer")

        result = await self._use_tool(
            tool=analyzer,
            tool_input={
                "dataset_info": state["dataset_info"],
                "target_variable": state["target_variable"],
            },
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        return result

    async def _load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset from storage"""
        # This would be implemented based on your storage mechanism
        # For now, using a placeholder that would load from your data directory
        import os

        dataset_path = os.path.join("data", "uploads", f"{dataset_id}.parquet")
        if os.path.exists(dataset_path):
            return pd.read_parquet(dataset_path)

        # Try CSV if parquet doesn't exist
        dataset_path = os.path.join("data", "uploads", f"{dataset_id}.csv")
        if os.path.exists(dataset_path):
            return pd.read_csv(dataset_path)

        raise FileNotFoundError(f"Dataset {dataset_id} not found")

    async def _create_visualizations(
        self,
        state: ConversationState,
        quality_result: Dict[str, Any],
        feature_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create EDA visualizations based on analysis results"""

        visualizations = {}

        try:
            # Load dataset
            df = await self._load_dataset(state["dataset_id"])

            # Record visualization creation
            state = self._record_thought(
                state, "Creating visualizations to better understand the data"
            )

            # 1. Data distribution plots
            try:
                # Select numeric columns for distribution plots
                numeric_cols = feature_result.get("numeric_features", [])
                if numeric_cols:
                    # Limit to top 12 most important features if too many
                    if len(numeric_cols) > 12 and state.get("feature_importance"):
                        importance_scores = state["feature_importance"]
                        important_numeric = [
                            col for col in numeric_cols if col in importance_scores
                        ]
                        # Sort by importance and take top 12
                        important_numeric.sort(
                            key=lambda x: importance_scores.get(x, 0), reverse=True
                        )
                        numeric_cols = important_numeric[:12]

                    visualizations["distribution"] = (
                        self.viz_tools.plot_data_distribution(
                            df, columns=numeric_cols[:12]  # Max 12 columns
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to create distribution plots: {e}")

            # 2. Missing values visualization
            try:
                if quality_result.get("missing_values"):
                    visualizations["missing_values"] = (
                        self.viz_tools.plot_missing_values(df)
                    )
            except Exception as e:
                logger.warning(f"Failed to create missing values plot: {e}")

            # 3. Correlation matrix
            try:
                # Only create correlation matrix if we have numeric features
                if len(feature_result.get("numeric_features", [])) > 1:
                    visualizations["correlation"] = (
                        self.viz_tools.plot_correlation_matrix(df)
                    )
            except Exception as e:
                logger.warning(f"Failed to create correlation matrix: {e}")

            # 4. Target distribution (if target is identified)
            try:
                if (
                    state.get("target_variable")
                    and state["target_variable"] in df.columns
                ):
                    visualizations["target"] = self.viz_tools.plot_target_distribution(
                        df[state["target_variable"]], state["problem_type"].value
                    )
            except Exception as e:
                logger.warning(f"Failed to create target distribution plot: {e}")

            # 5. Feature importance (if available)
            try:
                if (
                    state.get("feature_importance")
                    and len(state["feature_importance"]) > 0
                ):
                    feature_names = list(state["feature_importance"].keys())
                    importance_values = np.array(
                        list(state["feature_importance"].values())
                    )

                    visualizations["feature_importance"] = (
                        self.viz_tools.plot_feature_importance(
                            feature_names, importance_values, top_n=20
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to create feature importance plot: {e}")

            # Log successful visualization creation
            if visualizations:
                state = self._record_thought(
                    state,
                    f"Successfully created {len(visualizations)} visualizations",
                    decision=f"Visualizations: {', '.join(visualizations.keys())}",
                )

        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            # Return empty dict but don't fail the entire analysis
            state = self._record_thought(
                state,
                f"Visualization creation encountered issues: {str(e)}",
                decision="Proceeding without visualizations",
            )

        return visualizations

    def _generate_recommendations(
        self,
        profile_result: Dict[str, Any],
        quality_result: Dict[str, Any],
        feature_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate preprocessing recommendations based on analysis"""
        recommendations = []

        # Missing value recommendations
        if quality_result.get("missing_values"):
            for col, missing_pct in quality_result["missing_values"].items():
                if missing_pct > 50:
                    recommendations.append(
                        {
                            "type": "drop_column",
                            "column": col,
                            "reason": f"High missing percentage ({missing_pct:.1f}%)",
                            "priority": "high",
                        }
                    )
                elif missing_pct > 0:
                    recommendations.append(
                        {
                            "type": "impute",
                            "column": col,
                            "reason": f"Missing values ({missing_pct:.1f}%)",
                            "strategy": (
                                "mean"
                                if col in feature_result.get("numeric_features", [])
                                else "mode"
                            ),
                            "priority": "medium",
                        }
                    )

        # Outlier recommendations
        if quality_result.get("outliers"):
            for col, outlier_info in quality_result["outliers"].items():
                if outlier_info["percentage"] > 5:
                    recommendations.append(
                        {
                            "type": "handle_outliers",
                            "column": col,
                            "reason": f"High outlier percentage ({outlier_info['percentage']:.1f}%)",
                            "strategy": "clip",
                            "priority": "medium",
                        }
                    )

        # Feature engineering recommendations
        if feature_result.get("high_cardinality_categorical"):
            for col in feature_result["high_cardinality_categorical"]:
                recommendations.append(
                    {
                        "type": "encode_categorical",
                        "column": col,
                        "reason": "High cardinality categorical feature",
                        "strategy": "target_encoding",
                        "priority": "high",
                    }
                )

        # Scaling recommendations
        if feature_result.get("varying_scales"):
            recommendations.append(
                {
                    "type": "scale_features",
                    "reason": "Features have varying scales",
                    "strategy": "standard",
                    "priority": "high",
                }
            )

        # Constant feature recommendations
        if feature_result.get("constant_features"):
            for col in feature_result["constant_features"]:
                recommendations.append(
                    {
                        "type": "drop_column",
                        "column": col,
                        "reason": "Constant feature (no variation)",
                        "priority": "high",
                    }
                )

        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(
            key=lambda x: priority_order.get(x.get("priority", "low"), 3)
        )

        return recommendations

    def _create_analysis_summary(
        self,
        profile_result: Dict[str, Any],
        quality_result: Dict[str, Any],
        target_result: Optional[Dict[str, Any]],
        feature_result: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        visualizations: Dict[str, Any],
    ) -> str:
        """Create a comprehensive analysis summary"""

        dataset_info = profile_result["dataset_info"]

        summary = f"""
📊 **Dataset Analysis Complete**

**Dataset Overview:**
- Shape: {dataset_info['rows']:,} rows × {dataset_info['columns']} columns
- Memory Usage: {dataset_info['memory_usage'] / 1024 / 1024:.1f} MB
- File: {dataset_info['filename']}

**Problem Identification:**
"""

        if target_result:
            summary += f"""- Target Variable: `{target_result['target_variable']}`
- Problem Type: **{target_result['problem_type'].value.title()}**
- Reasoning: {target_result['reasoning']}
"""
        else:
            summary += f"""- Target Variable: `{state['target_variable']}`
- Problem Type: **{state['problem_type'].value.title()}**
"""

        summary += f"""
**Data Quality Assessment:**
- Missing Values: {len(quality_result.get('missing_values', {}))} columns affected
- Duplicates: {quality_result.get('duplicate_rows', 0):,} rows
- Outliers Detected: {len(quality_result.get('outliers', {}))} columns
"""

        if quality_result.get("critical_issues"):
            summary += "\n⚠️ **Critical Issues:**\n"
            for issue in quality_result["critical_issues"]:
                summary += f"- {issue}\n"

        summary += f"""
**Feature Analysis:**
- Numeric Features: {len(feature_result.get('numeric_features', []))}
- Categorical Features: {len(feature_result.get('categorical_features', []))}
- High Cardinality: {len(feature_result.get('high_cardinality_categorical', []))} features
"""

        if feature_result.get("constant_features"):
            summary += f"- Constant Features: {len(feature_result['constant_features'])} (will be dropped)\n"

        # Add visualization section
        if visualizations:
            summary += "\n**📈 Visualizations Created:**\n"
            viz_descriptions = {
                "distribution": "✓ Data distribution plots for numeric features",
                "missing_values": "✓ Missing values analysis chart",
                "correlation": "✓ Feature correlation heatmap",
                "target": "✓ Target variable distribution",
                "feature_importance": "✓ Feature importance rankings",
            }

            for viz_name in visualizations:
                if viz_name in viz_descriptions:
                    summary += f"{viz_descriptions[viz_name]}\n"

        if recommendations:
            summary += "\n**🔧 Preprocessing Recommendations:**\n"

            # Group by priority
            high_priority = [r for r in recommendations if r.get("priority") == "high"]
            medium_priority = [
                r for r in recommendations if r.get("priority") == "medium"
            ]

            if high_priority:
                summary += "\n*High Priority:*\n"
                for i, rec in enumerate(high_priority[:3], 1):
                    summary += f"{i}. **{rec['type'].replace('_', ' ').title()}**: {rec['reason']}\n"
                    if "column" in rec:
                        summary += f"   - Column: `{rec['column']}`\n"
                    if "strategy" in rec:
                        summary += f"   - Strategy: {rec['strategy']}\n"

            if medium_priority and len(high_priority) < 5:
                summary += "\n*Medium Priority:*\n"
                remaining = 5 - len(high_priority)
                for i, rec in enumerate(medium_priority[:remaining], 1):
                    summary += f"{i}. {rec['type'].replace('_', ' ').title()}: {rec['reason']}\n"

            total_shown = min(5, len(high_priority) + len(medium_priority))
            if len(recommendations) > total_shown:
                summary += f"\n... and {len(recommendations) - total_shown} more recommendations\n"

        summary += (
            "\n✅ **Analysis complete!** Ready to proceed with data preprocessing."
        )

        return summary
