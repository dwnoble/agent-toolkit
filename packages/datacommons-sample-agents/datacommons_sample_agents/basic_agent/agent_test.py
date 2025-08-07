import pathlib
from google.adk.evaluation.agent_evaluator import AgentEvaluator
import pytest

parent_path = pathlib.Path(__file__).parent

@pytest.mark.asyncio
async def test_data_commons_agent():
    """Test the agent's basic ability via a session file."""
    await AgentEvaluator.evaluate(
        agent_module="datacommons_sample_agents.basic_agent",
        eval_dataset_file_path_or_dir=str(
            parent_path / "data/"
        )
    )