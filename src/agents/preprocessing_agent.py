# src/agents/preprocessing_agent.py
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from langchain.schema import BaseMessage, HumanMessage
from src.agents.base_agent import BaseAgent
from src.agents.tools.ml_tools import (
    DataCleanerTool,
    FeatureEngineerTool,
    DataTransformerTool,
    DataSplitterTool,
)
from src.core.config import AGENT_PROMPTS, settings
from src.core.state import (
    ConversationState,
    AgentType,
    PreprocessingStep,
    add_message,
    set_pending_approval,
    update_state_timestamp,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingAgent(BaseAgent):
    """
    Agent responsible for data cleaning, feature engineering,
    and data preparation for model training.
    """

    def __init__(self):
        # Initialize tools
        tools = [
            DataCleanerTool(),
            FeatureEngineerTool(),
            DataTransformerTool(),
            DataSplitterTool(),
        ]

        super().__init__(
            name="PreprocessingAgent",
            agent_type=AgentType.PREPROCESSING,
            tools=tools,
            system_prompt=AGENT_PROMPTS["preprocessing"],
        )

    def _get_default_prompt(self) -> str:
        """Get default system prompt"""
        return AGENT_PROMPTS["preprocessing"]

    async def _execute(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Tuple[ConversationState, str]:
        """Execute preprocessing workflow"""

        # Check prerequisites
        if not state["dataset_id"] or not state["dataset_info"]:
            return state, "Dataset analysis must be completed before preprocessing."

        if not state["target_variable"]:
            return state, "Target variable must be identified before preprocessing."

        # Record initial thought
        state = self._record_thought(
            state, "Starting data preprocessing based on analysis recommendations"
        )

        try:
            # Get preprocessing plan
            preprocessing_plan = await self._create_preprocessing_plan(state, messages)

            # Check if approval needed in interactive mode
            if state["mode"] == "interactive" and not self._is_plan_approved(state):
                state = set_pending_approval(
                    state,
                    agent=self.agent_type,
                    action="preprocessing_plan",
                    description=self._format_plan_for_approval(preprocessing_plan),
                    options=["Approve", "Modify", "Skip"],
                    default_option="Approve",
                )
                return (
                    state,
                    "I've created a preprocessing plan. Please review and approve.",
                )

            # Execute preprocessing steps
            results = []
            for step in preprocessing_plan:
                step_result = await self._execute_preprocessing_step(state, step)
                results.append(step_result)

                # Update state with applied transformation
                state["applied_transformations"].append(
                    {
                        "step": step["name"],
                        "parameters": step["parameters"],
                        "result": step_result,
                    }
                )

            # Save preprocessed data
            processed_path = await self._save_processed_data(state)

            # Create summary
            summary = self._create_preprocessing_summary(
                preprocessing_plan, results, processed_path
            )

            # Update state
            state = add_message(
                state,
                "assistant",
                summary,
                metadata={
                    "agent": self.agent_type.value,
                    "preprocessing_complete": True,
                    "processed_data_path": processed_path,
                    "steps_applied": len(preprocessing_plan),
                },
            )

            # Record completion
            state = self._record_thought(
                state,
                "Preprocessing completed successfully",
                decision=f"Applied {len(preprocessing_plan)} preprocessing steps",
            )

            return state, summary

        except Exception as e:
            logger.error("Preprocessing failed", error=str(e))
            error_message = f"Preprocessing encountered an error: {str(e)}"
            state = add_message(state, "assistant", error_message)
            raise

    async def _create_preprocessing_plan(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Create preprocessing plan based on data analysis"""

        # Get recommendations from state (set by analysis agent)
        recommendations = []
        for msg in state["messages"]:
            if msg.get("metadata", {}).get("recommendations"):
                recommendations = msg["metadata"]["recommendations"]
                break

        # Create prompt for preprocessing plan
        prompt = f"""
Based on the data analysis and recommendations, create a preprocessing plan.

Dataset Info:
- Columns: {state['dataset_info']['column_names']}
- Target: {state['target_variable']}
- Problem Type: {state['problem_type']}

Data Quality Issues:
{self._format_quality_issues(state)}

Recommendations:
{self._format_recommendations(recommendations)}

Create a step-by-step preprocessing plan. For each step provide:
STEP: [step name]
ACTION: [specific action to take]
PARAMETERS: [parameters for the action]
REASON: [why this step is needed]

Order the steps logically (e.g., handle missing values before scaling).
"""

        # Get LLM response
        response = await self._call_llm(
            messages + [HumanMessage(content=prompt)],
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        # Parse plan
        plan = self._parse_preprocessing_plan(response.content)

        # Add to state
        for step in plan:
            preprocessing_step = PreprocessingStep(
                name=step["name"],
                description=step["reason"],
                parameters=step["parameters"],
                applied=False,
            )
            state["preprocessing_steps"].append(preprocessing_step.dict())

        return plan

    def _format_quality_issues(self, state: ConversationState) -> str:
        """Format quality issues for prompt"""
        issues = []

        # Extract from messages
        for msg in state["messages"]:
            if msg.get("metadata", {}).get("agent") == "analysis":
                # This would contain quality assessment results
                pass

        # Extract from warnings
        for warning in state["warnings"]:
            if warning.get("agent") == "analysis":
                issues.append(f"- {warning['warning']}")

        return "\n".join(issues) if issues else "No critical issues found"

    def _format_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations for prompt"""
        if not recommendations:
            return "No specific recommendations"

        formatted = []
        for rec in recommendations:
            formatted.append(f"- {rec['type']}: {rec['reason']}")

        return "\n".join(formatted)

    def _parse_preprocessing_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse preprocessing plan from LLM response"""
        plan = []
        current_step = {}

        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("STEP:"):
                if current_step:
                    plan.append(current_step)
                current_step = {"name": line.replace("STEP:", "").strip()}
            elif line.startswith("ACTION:"):
                current_step["action"] = line.replace("ACTION:", "").strip()
            elif line.startswith("PARAMETERS:"):
                # Parse parameters as key-value pairs
                param_str = line.replace("PARAMETERS:", "").strip()
                params = {}
                if param_str and param_str != "None":
                    # Simple parsing - in production, use proper parsing
                    for param in param_str.split(","):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            params[key.strip()] = value.strip()
                current_step["parameters"] = params
            elif line.startswith("REASON:"):
                current_step["reason"] = line.replace("REASON:", "").strip()

        if current_step:
            plan.append(current_step)

        return plan

    def _is_plan_approved(self, state: ConversationState) -> bool:
        """Check if preprocessing plan is approved"""
        # Check if there was a recent approval
        for decision in reversed(state["user_decisions"]):
            if decision.get("action") == "preprocessing_plan":
                return decision.get("decision") == "Approve"
        return False

    def _format_plan_for_approval(self, plan: List[Dict[str, Any]]) -> str:
        """Format preprocessing plan for user approval"""
        description = "I'm planning to apply the following preprocessing steps:\n\n"

        for i, step in enumerate(plan, 1):
            description += f"{i}. **{step['name']}**\n"
            description += f"   - Action: {step['action']}\n"
            if step["parameters"]:
                description += f"   - Parameters: {step['parameters']}\n"
            description += f"   - Reason: {step['reason']}\n\n"

        description += "Would you like me to proceed with this plan?"
        return description

    async def _execute_preprocessing_step(
        self, state: ConversationState, step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single preprocessing step"""
        action = step["action"].lower()

        # Map actions to tools
        if "clean" in action or "missing" in action:
            tool = self._get_tool_by_name("data_cleaner")
        elif "engineer" in action or "feature" in action:
            tool = self._get_tool_by_name("feature_engineer")
        elif "transform" in action or "scale" in action or "encode" in action:
            tool = self._get_tool_by_name("data_transformer")
        elif "split" in action:
            tool = self._get_tool_by_name("data_splitter")
        else:
            # Default to transformer
            tool = self._get_tool_by_name("data_transformer")

        # Prepare tool input
        tool_input = {
            "dataset_id": state["dataset_id"],
            "action": step["action"],
            "parameters": step["parameters"],
            "target_variable": state["target_variable"],
        }

        # Execute tool
        result = await self._use_tool(
            tool=tool,
            tool_input=tool_input,
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        return result

    async def _save_processed_data(self, state: ConversationState) -> str:
        """Save processed data and return path"""
        try:
            # Get splitter tool to save final processed data
            splitter = self._get_tool_by_name("data_splitter")

            result = await self._use_tool(
                tool=splitter,
                tool_input={
                    "dataset_id": state["dataset_id"],
                    "test_size": 0.2,
                    "random_state": 42,
                    "stratify": state["problem_type"] == "classification",
                },
                session_id=state["session_id"],
                user_id=state["user_id"],
            )

            return result.get("output_path", "data/processed/")

        except Exception as e:
            logger.error("Failed to save processed data", error=str(e))
            return "data/processed/"

    def _create_preprocessing_summary(
        self,
        plan: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        processed_path: str,
    ) -> str:
        """Create preprocessing summary"""
        summary = """
🔧 **Data Preprocessing Complete**

**Steps Applied:**
"""

        for i, (step, result) in enumerate(zip(plan, results), 1):
            status = "✅" if not result.get("error") else "❌"
            summary += f"{i}. {status} {step['name']}\n"

            if result.get("changes"):
                summary += f"   - Changes: {result['changes']}\n"
            if result.get("error"):
                summary += f"   - Error: {result['error']}\n"

        # Add statistics
        successful_steps = sum(1 for r in results if not r.get("error"))
        summary += f"""
**Summary:**
- Total Steps: {len(plan)}
- Successful: {successful_steps}
- Failed: {len(plan) - successful_steps}

**Output:**
- Processed data saved to: `{processed_path}`
- Ready for model training

✅ Data preprocessing completed! The dataset is now ready for model training.
"""

        return summary
