# src/agents/supervisor.py
from typing import Dict, Any, List, Tuple, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.core.config import settings, AGENT_PROMPTS
from src.core.state import (
    ConversationState,
    AgentType,
    WorkflowStatus,
    ProblemType,
    add_message,
    set_pending_approval,
    update_state_timestamp,
)
from src.core.events import event_tracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that orchestrates the entire AutoML workflow.
    Routes tasks to appropriate specialist agents and manages state.
    """

    def __init__(self):
        super().__init__(
            name="Supervisor",
            agent_type=AgentType.SUPERVISOR,
            system_prompt=AGENT_PROMPTS["supervisor"],
        )

        # Define routing logic
        self.routing_rules = {
            "analyze": AgentType.ANALYSIS,
            "preprocess": AgentType.PREPROCESSING,
            "train": AgentType.MODEL,
            "optimize": AgentType.OPTIMIZATION,
        }

    def _get_default_prompt(self) -> str:
        """Get default system prompt"""
        return AGENT_PROMPTS["supervisor"]

    async def _execute(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Tuple[ConversationState, str]:
        """Execute supervisor logic"""

        # Record thought
        state = self._record_thought(
            state, "Analyzing user request and current workflow state"
        )

        # Get current context
        context = self._get_workflow_context(state)

        # Create decision prompt
        decision_prompt = self._create_decision_prompt(state, context)
        messages.append(HumanMessage(content=decision_prompt))

        # Get LLM decision
        response = await self._call_llm(
            messages, session_id=state["session_id"], user_id=state["user_id"]
        )

        # Parse decision
        decision = self._parse_decision(response.content)

        # Record decision
        state = self._record_thought(
            state,
            f"Decision made: {decision['action']}",
            decision=decision.get("reasoning", ""),
        )

        # Update state based on decision
        state = await self._execute_decision(state, decision)

        # Add response message
        state = add_message(
            state,
            "assistant",
            decision.get("user_message", "I'm processing your request..."),
            metadata={"decision": decision},
        )

        return state, decision.get("user_message", "")

    def _get_workflow_context(self, state: ConversationState) -> Dict[str, Any]:
        """Get current workflow context"""
        return {
            "has_dataset": state["dataset_id"] is not None,
            "dataset_analyzed": state["dataset_info"] is not None,
            "problem_type": state["problem_type"],
            "target_variable": state["target_variable"],
            "preprocessing_done": len(state["applied_transformations"]) > 0,
            "models_trained": len(state["models_trained"]),
            "optimization_done": state["best_params"] is not None,
            "current_status": state["status"],
            "errors": state["errors"][-5:] if state["errors"] else [],
            "mode": state["mode"],
        }

    def _create_decision_prompt(
        self, state: ConversationState, context: Dict[str, Any]
    ) -> str:
        """Create prompt for decision making"""
        user_message = state["messages"][-1]["content"] if state["messages"] else ""

        prompt = f"""
Based on the current workflow state and user request, decide the next action.

User Request: {user_message}

Current Context:
- Dataset uploaded: {context['has_dataset']}
- Dataset analyzed: {context['dataset_analyzed']}
- Problem type: {context['problem_type'] or 'Not determined'}
- Target variable: {context['target_variable'] or 'Not set'}
- Preprocessing completed: {context['preprocessing_done']}
- Models trained: {context['models_trained']}
- Optimization completed: {context['optimization_done']}
- Workflow mode: {context['mode']}

Recent errors: {context['errors']}

Decide the next action from these options:
1. Route to Analysis Agent - If dataset needs analysis or problem type identification
2. Route to Preprocessing Agent - If data needs cleaning or preparation
3. Route to Model Agent - If ready for model training
4. Route to Optimization Agent - If models need hyperparameter tuning
5. Request user input - If critical information is missing
6. Complete workflow - If all tasks are done
7. Handle error - If there's an error to address

Provide your decision in this format:
ACTION: [action name]
REASONING: [brief explanation]
NEXT_AGENT: [analysis/preprocessing/model/optimization/none]
USER_MESSAGE: [message to show the user]
REQUIRES_APPROVAL: [yes/no]
APPROVAL_MESSAGE: [if requires approval, what to ask]
"""

        return prompt

    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """Parse LLM decision from response"""
        decision = {
            "action": "",
            "reasoning": "",
            "next_agent": None,
            "user_message": "",
            "requires_approval": False,
            "approval_message": "",
        }

        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("ACTION:"):
                decision["action"] = line.replace("ACTION:", "").strip()
            elif line.startswith("REASONING:"):
                decision["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("NEXT_AGENT:"):
                agent_str = line.replace("NEXT_AGENT:", "").strip().lower()
                if agent_str in ["analysis", "preprocessing", "model", "optimization"]:
                    decision["next_agent"] = AgentType(agent_str)
            elif line.startswith("USER_MESSAGE:"):
                decision["user_message"] = line.replace("USER_MESSAGE:", "").strip()
            elif line.startswith("REQUIRES_APPROVAL:"):
                decision["requires_approval"] = (
                    line.replace("REQUIRES_APPROVAL:", "").strip().lower() == "yes"
                )
            elif line.startswith("APPROVAL_MESSAGE:"):
                decision["approval_message"] = line.replace(
                    "APPROVAL_MESSAGE:", ""
                ).strip()

        # Default user message if not provided
        if not decision["user_message"]:
            decision["user_message"] = f"Proceeding with {decision['action']}..."

        return decision

    async def _execute_decision(
        self, state: ConversationState, decision: Dict[str, Any]
    ) -> ConversationState:
        """Execute the supervisor's decision"""

        # Check if approval is required
        if decision["requires_approval"] and state["mode"] == "interactive":
            state = set_pending_approval(
                state,
                agent=self.agent_type,
                action=decision["action"],
                description=decision["approval_message"],
                options=["Approve", "Modify", "Skip"],
                default_option="Approve",
            )
            return state

        # Route to next agent
        if decision["next_agent"]:
            state["current_agent"] = decision["next_agent"]
            state["status"] = WorkflowStatus.RUNNING

            # Track routing decision
            await event_tracker.track_agent_action(
                session_id=state["session_id"],
                agent_name=self.name,
                action="route",
                decision=f"Routing to {decision['next_agent'].value}",
                output_data={"next_agent": decision["next_agent"].value},
                user_id=state["user_id"],
            )

        # Handle special actions
        elif decision["action"].lower() == "complete workflow":
            state["status"] = WorkflowStatus.COMPLETED
            state["current_agent"] = None

        elif decision["action"].lower() == "handle error":
            # TODO: Implement error recovery logic
            state["status"] = WorkflowStatus.FAILED
            state["current_agent"] = None

        return update_state_timestamp(state)

    def should_route_to_agent(
        self, state: ConversationState, agent_type: AgentType
    ) -> bool:
        """Determine if workflow should route to a specific agent"""
        context = self._get_workflow_context(state)

        if agent_type == AgentType.ANALYSIS:
            return context["has_dataset"] and not context["dataset_analyzed"]

        elif agent_type == AgentType.PREPROCESSING:
            return (
                context["dataset_analyzed"]
                and context["problem_type"] is not None
                and not context["preprocessing_done"]
            )

        elif agent_type == AgentType.MODEL:
            return context["preprocessing_done"] and context["models_trained"] == 0

        elif agent_type == AgentType.OPTIMIZATION:
            return context["models_trained"] > 0 and not context["optimization_done"]

        return False

    def get_next_agent(self, state: ConversationState) -> Optional[AgentType]:
        """Get the next agent in the workflow"""
        # Check each agent in order
        for agent_type in [
            AgentType.ANALYSIS,
            AgentType.PREPROCESSING,
            AgentType.MODEL,
            AgentType.OPTIMIZATION,
        ]:
            if self.should_route_to_agent(state, agent_type):
                return agent_type

        return None
