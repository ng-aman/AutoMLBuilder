# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.tools import BaseTool
from src.core.config import settings
from src.core.state import ConversationState, AgentType, add_agent_thought
from src.core.events import event_tracker
from src.utils.logger import get_logger
from src.utils.exceptions import AgentError, AgentTimeoutError, AgentExecutionError

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the system"""

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[ChatOpenAI] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = None,
        timeout_seconds: int = None,
    ):
        self.name = name
        self.agent_type = agent_type
        self.tools = tools or []
        self.system_prompt = system_prompt or self._get_default_prompt()
        self.max_retries = max_retries or settings.agent_max_retries
        self.timeout_seconds = timeout_seconds or settings.agent_timeout_seconds

        # Initialize LLM
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            api_key=settings.openai_api_key,
        )

        # Bind tools to LLM if available
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    @abstractmethod
    def _get_default_prompt(self) -> str:
        """Get default system prompt for the agent"""
        pass

    @abstractmethod
    async def _execute(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Tuple[ConversationState, str]:
        """Execute agent logic - to be implemented by subclasses"""
        pass

    async def __call__(
        self, state: ConversationState, user_message: Optional[str] = None
    ) -> ConversationState:
        """Main entry point for agent execution"""
        start_time = datetime.now(timezone.utc)

        # Update state with current agent
        state["current_agent"] = self.agent_type

        # Track agent start
        await event_tracker.track_agent_action(
            session_id=state["session_id"],
            agent_name=self.name,
            action="start",
            user_id=state["user_id"],
        )

        try:
            # Prepare messages
            messages = self._prepare_messages(state, user_message)

            # Execute with timeout and retries
            state, response = await self._execute_with_retry(state, messages)

            # Track success
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await event_tracker.track_agent_action(
                session_id=state["session_id"],
                agent_name=self.name,
                action="complete",
                output_data={"response": response},
                duration_ms=duration_ms,
                user_id=state["user_id"],
            )

            return state

        except Exception as e:
            # Track error
            await event_tracker.track_agent_action(
                session_id=state["session_id"],
                agent_name=self.name,
                action="error",
                error=str(e),
                user_id=state["user_id"],
            )

            # Re-raise the exception
            raise

    def _prepare_messages(
        self, state: ConversationState, user_message: Optional[str] = None
    ) -> List[BaseMessage]:
        """Prepare messages for LLM"""
        messages = []

        # Add system prompt
        system_prompt = self._format_system_prompt(state)
        messages.append(SystemMessage(content=system_prompt))

        # Add conversation history (last 10 messages)
        for msg in state["messages"][-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current user message if provided
        if user_message:
            messages.append(HumanMessage(content=user_message))

        return messages

    def _format_system_prompt(self, state: ConversationState) -> str:
        """Format system prompt with current context"""
        context = {
            "mode": state["mode"],
            "has_dataset": state["dataset_id"] is not None,
            "dataset_info": state["dataset_info"],
            "target_variable": state["target_variable"],
            "problem_type": state["problem_type"],
            "models_trained": len(state["models_trained"]),
            "current_status": state["status"],
        }

        return self.system_prompt.format(**context)

    async def _execute_with_retry(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Tuple[ConversationState, str]:
        """Execute with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute(state, messages), timeout=self.timeout_seconds
                )
                return result

            except asyncio.TimeoutError:
                last_error = AgentTimeoutError(
                    agent_name=self.name, timeout_seconds=self.timeout_seconds
                )
                logger.warning(
                    f"Agent timeout (attempt {attempt + 1}/{self.max_retries})",
                    agent=self.name,
                    timeout=self.timeout_seconds,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Agent execution failed (attempt {attempt + 1}/{self.max_retries})",
                    agent=self.name,
                    error=str(e),
                )

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        # All retries failed
        if isinstance(last_error, AgentError):
            raise last_error
        else:
            raise AgentExecutionError(agent_name=self.name, error=str(last_error))

    async def _call_llm(
        self, messages: List[BaseMessage], session_id: str, user_id: str
    ) -> AIMessage:
        """Call LLM and track the event"""
        start_time = datetime.now(timezone.utc)

        try:
            # Track LLM call start
            prompt = "\n".join([m.content for m in messages])

            # Make LLM call
            response = await self.llm.ainvoke(messages)

            # Calculate metrics
            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Track successful call
            await event_tracker.track_llm_call(
                session_id=session_id,
                model=settings.openai_model,
                prompt=prompt,
                response=response.content,
                duration_ms=duration_ms,
                user_id=user_id,
                agent=self.name,
            )

            return response

        except Exception as e:
            # Track error
            await event_tracker.track_llm_call(
                session_id=session_id,
                model=settings.openai_model,
                prompt=prompt,
                error=str(e),
                user_id=user_id,
                agent=self.name,
            )
            raise

    async def _use_tool(
        self, tool: BaseTool, tool_input: Any, session_id: str, user_id: str
    ) -> Any:
        """Use a tool and track the event"""
        start_time = datetime.now(timezone.utc)

        try:
            # Track tool call start
            parameters = (
                tool_input
                if isinstance(tool_input, dict)
                else {"input": str(tool_input)}
            )

            # Execute tool
            result = await tool.arun(tool_input)

            # Calculate metrics
            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Track successful call
            await event_tracker.track_tool_call(
                session_id=session_id,
                tool_name=tool.name,
                parameters=parameters,
                result=result,
                duration_ms=duration_ms,
                user_id=user_id,
                agent=self.name,
            )

            return result

        except Exception as e:
            # Track error
            await event_tracker.track_tool_call(
                session_id=session_id,
                tool_name=tool.name,
                parameters=parameters,
                error=str(e),
                user_id=user_id,
                agent=self.name,
            )
            raise

    def _record_thought(
        self, state: ConversationState, thought: str, decision: Optional[str] = None
    ) -> ConversationState:
        """Record agent thought in state"""
        return add_agent_thought(
            state=state, agent=self.agent_type, thought=thought, decision=decision
        )

    def _get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    async def _handle_tool_calls(
        self, response: AIMessage, state: ConversationState
    ) -> str:
        """Handle tool calls from LLM response"""
        if not response.tool_calls:
            return response.content

        tool_results = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool = self._get_tool_by_name(tool_name)
            if not tool:
                logger.error(f"Tool not found: {tool_name}")
                continue

            try:
                result = await self._use_tool(
                    tool=tool,
                    tool_input=tool_args,
                    session_id=state["session_id"],
                    user_id=state["user_id"],
                )
                tool_results.append(f"{tool_name}: {result}")
            except Exception as e:
                logger.error(f"Tool execution failed: {tool_name}", error=str(e))
                tool_results.append(f"{tool_name}: Error - {str(e)}")

        # Combine original response with tool results
        combined_response = response.content or ""
        if tool_results:
            combined_response += "\n\nTool Results:\n" + "\n".join(tool_results)

        return combined_response
