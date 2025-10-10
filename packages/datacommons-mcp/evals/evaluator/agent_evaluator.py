# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import json
import logging
import os
import statistics
import uuid
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Missing dependency 'pandas'. Please install it.") from e


from google.adk.agents.base_agent import BaseAgent
from google.adk.evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    EvalMetricResult,
    PrebuiltMetrics,
)
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_sets_manager import (
    convert_eval_set_to_pydanctic_schema,
)
from google.adk.utils.context_utils import Aclosing
from google.genai import types as genai_types
from pydantic import BaseModel

logger = logging.getLogger("google_adk." + __name__)


# Constants for default runs and evaluation criteria
NUM_RUNS = 2

TOOL_TRAJECTORY_SCORE_KEY = PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
RESPONSE_MATCH_SCORE_KEY = PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
SAFETY_V1_KEY = PrebuiltMetrics.SAFETY_V1.value

ALLOWED_CRITERIA = [
    TOOL_TRAJECTORY_SCORE_KEY,
    RESPONSE_EVALUATION_SCORE_KEY,
    RESPONSE_MATCH_SCORE_KEY,
    SAFETY_V1_KEY,
]

QUERY_COLUMN = "query"
REFERENCE_COLUMN = "reference"
EXPECTED_TOOL_USE_COLUMN = "expected_tool_use"


DEFAULT_CRITERIA = {
    TOOL_TRAJECTORY_SCORE_KEY: 1.0,  # 1-point scale; 1.0 is perfect.
    RESPONSE_MATCH_SCORE_KEY: 0.8,  # Rouge-1 text match; 0.8 is default.
}


def load_json(file_path: str) -> dict | list:
    with open(file_path) as f:
        return json.load(f)


class _EvalMetricResultWithInvocation(BaseModel):
    """EvalMetricResult along with both actual and expected invocation.

    This is class is intentionally marked as private and is created for
    convenience.
    """

    actual_invocation: Invocation
    expected_invocation: Invocation
    eval_metric_result: EvalMetricResult


class AgentEvaluator:
    """An evaluator for Agents, mainly intended for helping with test cases."""

    @staticmethod
    def find_config_for_test_file(test_file: str):
        """Find the test_config.json file in the same folder as the test file."""
        test_folder = os.path.dirname(test_file)
        config_path = os.path.join(test_folder, "test_config.json")
        if os.path.exists(config_path):
            config_data = load_json(config_path)
            if "criteria" in config_data and isinstance(config_data["criteria"], dict):
                return config_data["criteria"]
            raise ValueError(
                f"Invalid format for test_config.json at {config_path}. Expected a"
                " 'criteria' dictionary."
            )
        return DEFAULT_CRITERIA

    @staticmethod
    async def evaluate_eval_set(
        agent_module: str,
        eval_set: EvalSet,
        criteria: dict[str, float],
        num_runs: int = NUM_RUNS,
        agent_name: str | None = None,
    ):
        """Evaluates an agent using the given EvalSet.

        Returns a pandas DataFrame with the evaluation results.

        Args:
          agent_module: The path to python module that contains the definition of
            the agent. There is convention in place here, where the code is going to
            look for 'root_agent' in the loaded module.
          eval_set: The eval set.
          criteria: Evauation criterias, a dictionary of metric names to their
            respective thresholds.
          num_runs: Number of times all entries in the eval dataset should be
            assessed.
          agent_name: The name of the agent, if trying to evaluate something other
            than root agent. If left empty or none, then root agent is evaluated.
        """
        agent_for_eval = AgentEvaluator._get_agent_for_eval(
            module_name=agent_module, agent_name=agent_name
        )
        eval_metrics = [
            EvalMetric(metric_name=n, threshold=t) for n, t in criteria.items()
        ]

        # Step 1: Perform evals, basically inferencing and evaluation of metrics
        eval_results_by_eval_id = await AgentEvaluator._get_eval_results_by_eval_id(
            agent_for_eval=agent_for_eval,
            eval_set=eval_set,
            eval_metrics=eval_metrics,
            num_runs=num_runs,
        )

        # Step 2: Post-process the results
        for _, eval_results_per_eval_id in eval_results_by_eval_id.items():
            eval_metric_results = (
                AgentEvaluator._get_eval_metric_results_with_invocation(
                    eval_results_per_eval_id
                )
            )

        return AgentEvaluator._create_results_dataframe(eval_metric_results, num_runs)

    @staticmethod
    async def evaluate(
        agent_module: str,
        eval_dataset_path: str,
        num_runs: int = NUM_RUNS,
        agent_name: str | None = None,
    ) -> pd.DataFrame:
        """Evaluates an Agent and returns a DataFrame of results.

        Args:
          agent_module: The path to python module that contains the agent's definition.
          eval_dataset_file_path_or_dir: Path to a single .test.json file or a
            directory to be explored for all .test.json files.
          num_runs: Number of times to assess each entry in the eval dataset.
          agent_name: The name of the agent.
          initial_session_file: File with initial session state for all evals.

        Returns:
            A pandas DataFrame with evaluation results
        """

        # 1. Gather all test files from the given path
        criteria = AgentEvaluator.find_config_for_test_file(eval_dataset_path)
        eval_set = AgentEvaluator._get_eval_set_from_old_format(
            eval_dataset_path, criteria, {}
        )

        # 2. Capture the DataFrame returned by `evaluate_eval_set`
        result_df = await AgentEvaluator.evaluate_eval_set(
            agent_module=agent_module,
            eval_set=eval_set,
            criteria=criteria,
            num_runs=num_runs,
            agent_name=agent_name,
        )

        return result_df

    @staticmethod
    def _get_eval_set_from_old_format(
        eval_set_file: str,
        criteria: dict[str, float],
        initial_session: dict[str, Any],
    ) -> EvalSet:
        data = AgentEvaluator._load_dataset(eval_set_file)[0]
        AgentEvaluator._validate_input([data], criteria)
        eval_data = {
            "name": eval_set_file,
            "data": data,
            "initial_session": initial_session,
        }
        return convert_eval_set_to_pydanctic_schema(
            eval_set_id=str(uuid.uuid4()), eval_set_in_json_format=[eval_data]
        )

    @staticmethod
    def _get_initial_session(initial_session_file: str | None = None):
        initial_session = {}
        if initial_session_file:
            with open(initial_session_file) as f:
                initial_session = json.loads(f.read())
        return initial_session

    @staticmethod
    def _load_dataset(
        input_data: str | list[str] | list[dict] | list[list[dict]],
    ) -> list[list[dict]]:
        def load_json_file(file_path: str) -> list[dict]:
            data = load_json(file_path)
            if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
                raise ValueError(f"{file_path} must contain a list of dictionaries.")
            return data

        if isinstance(input_data, str):
            if os.path.isdir(input_data):
                test_files = []
                for root, _, files in os.walk(input_data):
                    for file in files:
                        if file.endswith(".test.json"):
                            test_files.append(os.path.join(root, file))
                return [load_json_file(f) for f in test_files]
            if os.path.isfile(input_data):
                return [load_json_file(input_data)]
            raise ValueError(f"Input path {input_data} is invalid.")
        if isinstance(input_data, list):
            if all(isinstance(i, str) and os.path.isfile(i) for i in input_data):
                return [load_json_file(i) for i in input_data]
            raise TypeError("Input list must contain valid file paths.")
        raise TypeError("Invalid input type for dataset loading.")

    @staticmethod
    def _validate_input(eval_dataset, criteria):
        """Validates that the evaluation criteria align with the provided dataset.

        For efficiency, we only use first row to validate input.
        """
        if not eval_dataset:
            raise ValueError("The evaluation dataset is None or empty.")

        for key in criteria:
            if key not in ALLOWED_CRITERIA:
                raise ValueError(
                    f"Invalid criteria key: {key}. Expected one of {ALLOWED_CRITERIA}."
                )

        if not eval_dataset:
            raise ValueError("The evaluation dataset is empty.")
        sample = eval_dataset[0]
        first_query = sample[0]

        if not isinstance(sample, list) and not isinstance(first_query, dict):
            raise ValueError(
                "Each evaluation dataset sample must be list of dictionary. But it's"
                f" {eval_dataset}"
            )

        if TOOL_TRAJECTORY_SCORE_KEY in criteria:
            if (
                QUERY_COLUMN not in first_query
                or EXPECTED_TOOL_USE_COLUMN not in first_query
            ):
                raise ValueError(
                    f"Samples for {TOOL_TRAJECTORY_SCORE_KEY} must include"
                    f" '{QUERY_COLUMN}' and '{EXPECTED_TOOL_USE_COLUMN}' keys. The"
                    f" sample is {sample}."
                )

        if RESPONSE_EVALUATION_SCORE_KEY in criteria:
            if QUERY_COLUMN not in first_query:
                raise ValueError(
                    f"Samples for {RESPONSE_EVALUATION_SCORE_KEY} must include"
                    f" '{QUERY_COLUMN}' key. The sample is {sample}."
                )

        if RESPONSE_MATCH_SCORE_KEY in criteria:
            if QUERY_COLUMN not in first_query or REFERENCE_COLUMN not in first_query:
                raise ValueError(
                    f"Samples for {RESPONSE_MATCH_SCORE_KEY} must include"
                    f" '{QUERY_COLUMN}' and '{REFERENCE_COLUMN}' keys. The sample is"
                    f" {sample}."
                )

    @staticmethod
    def _convert_content_to_text(content: genai_types.Content | None) -> str:
        if content and content.parts:
            return "\n".join([p.text for p in content.parts if p.text])

        return ""

    @staticmethod
    def _convert_tool_calls_to_text(
        intermediate_data: IntermediateData | None,
    ) -> str:
        if intermediate_data and intermediate_data.tool_uses:
            # TODO: Improve formatting for better readability
            # Current format: id='adk-f2b322b5-9d34-443b-9010-d02641f5c3b8' args={'query': 'Population', 'places': ['California, USA']} name='search_indicators'
            # Desired format: Tool: search_indicators\nArgs: {'query': 'Population', 'places': ['California, USA']}
            #
            # Suggested implementation:
            formatted_tools = []
            for tool in intermediate_data.tool_uses:
                formatted_tools.append(f"Tool: {tool.name}\nArgs: {tool.args}")
            return "\n\n".join(formatted_tools)

            # return "\n".join([str(t) for t in intermediate_data.tool_uses])

        return ""

    @staticmethod
    def _get_agent_for_eval(
        module_name: str, agent_name: str | None = None
    ) -> BaseAgent:
        module_path = f"{module_name}"
        agent_module = importlib.import_module(module_path)
        root_agent = agent_module.agent.root_agent

        agent_for_eval = root_agent
        if agent_name:
            agent_for_eval = root_agent.find_agent(agent_name)
            assert agent_for_eval, f"Sub-Agent `{agent_name}` not found."

        return agent_for_eval

    @staticmethod
    def _get_eval_sets_manager(app_name: str, eval_set: EvalSet) -> EvalSetsManager:
        eval_sets_manager = InMemoryEvalSetsManager()

        eval_sets_manager.create_eval_set(
            app_name=app_name, eval_set_id=eval_set.eval_set_id
        )
        for eval_case in eval_set.eval_cases:
            eval_sets_manager.add_eval_case(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                eval_case=eval_case,
            )

        return eval_sets_manager

    @staticmethod
    async def _get_eval_results_by_eval_id(
        agent_for_eval: BaseAgent,
        eval_set: EvalSet,
        eval_metrics: list[EvalMetric],
        num_runs: int,
    ) -> dict[str, list[EvalCaseResult]]:
        """Returns EvalCaseResults grouped by eval case id.

        The grouping happens because of the "num_runs" argument, where for any value
        greater than 1, we would have generated inferences num_runs times and so
        by extension we would have evaluated metrics on each of those inferences.
        """
        try:
            from google.adk.evaluation.base_eval_service import (
                EvaluateConfig,
                EvaluateRequest,
                InferenceConfig,
                InferenceRequest,
            )
            from google.adk.evaluation.local_eval_service import LocalEvalService
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

        # It is okay to pick up this dummy name.
        app_name = "test_app"
        eval_service = LocalEvalService(
            root_agent=agent_for_eval,
            eval_sets_manager=AgentEvaluator._get_eval_sets_manager(
                app_name=app_name, eval_set=eval_set
            ),
        )

        inference_requests = [
            InferenceRequest(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                inference_config=InferenceConfig(),
            )
        ] * num_runs  # Repeat inference request num_runs times.

        # Generate inferences
        inference_results = []
        for inference_request in inference_requests:
            async with Aclosing(
                eval_service.perform_inference(inference_request=inference_request)
            ) as agen:
                async for inference_result in agen:
                    inference_results.append(inference_result)

        # Evaluate metrics
        # As we perform more than one run for an eval case, we collect eval results
        # by eval id.
        eval_results_by_eval_id: dict[str, list[EvalCaseResult]] = {}
        evaluate_request = EvaluateRequest(
            inference_results=inference_results,
            evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
        )
        async with Aclosing(
            eval_service.evaluate(evaluate_request=evaluate_request)
        ) as agen:
            async for eval_result in agen:
                eval_id = eval_result.eval_id
                if eval_id not in eval_results_by_eval_id:
                    eval_results_by_eval_id[eval_id] = []

                eval_results_by_eval_id[eval_id].append(eval_result)

        return eval_results_by_eval_id

    @staticmethod
    def _get_eval_metric_results_with_invocation(
        eval_results_per_eval_id: list[EvalCaseResult],
    ) -> dict[str, list[_EvalMetricResultWithInvocation]]:
        """Returns _EvalMetricResultWithInvocation grouped by metric.

        EvalCaseResult contain results for each metric per invocation.

        This method flips it around and returns a structure that groups metric
        results per invocation by eval metric.

        This is a convenience function.
        """
        eval_metric_results: dict[str, list[_EvalMetricResultWithInvocation]] = {}

        # Go over the EvalCaseResult one by one, do note that at this stage all
        # EvalCaseResult belong to the same eval id.
        for eval_case_result in eval_results_per_eval_id:
            # For the given eval_case_result, we go over metric results for each
            # invocation. Do note that a single eval case can have more than one
            # invocation and for each invocation there could be more than on eval
            # metrics that were evaluated.
            for (
                eval_metrics_per_invocation
            ) in eval_case_result.eval_metric_result_per_invocation:
                # Go over each eval_metric_result for an invocation.
                for (
                    eval_metric_result
                ) in eval_metrics_per_invocation.eval_metric_results:
                    metric_name = eval_metric_result.metric_name
                    if metric_name not in eval_metric_results:
                        eval_metric_results[metric_name] = []

                    actual_invocation = eval_metrics_per_invocation.actual_invocation
                    expected_invocation = (
                        eval_metrics_per_invocation.expected_invocation
                    )

                    eval_metric_results[metric_name].append(
                        _EvalMetricResultWithInvocation(
                            actual_invocation=actual_invocation,
                            expected_invocation=expected_invocation,
                            eval_metric_result=eval_metric_result,
                        )
                    )
        return eval_metric_results

    @staticmethod
    def _create_results_dataframe(
        eval_metric_results: dict[
            str, list[_EvalMetricResultWithInvocation]
        ],  # Using Any for _EvalMetricResultWithInvocation
        num_runs: int,
    ) -> pd.DataFrame:
        """
        Processes evaluation results into a pandas DataFrame.

        Returns:
            A pandas DataFrame containing detailed results for each invocation,
            augmented with the average score and overall status for its corresponding metric.
        """
        all_results_data = []

        # 1. Iterate through each metric in the results (e.g., "ExactMatch", "Fidelity")
        for (
            metric_name,
            eval_metric_results_with_invocations,
        ) in eval_metric_results.items():
            if not eval_metric_results_with_invocations:
                continue

            threshold = eval_metric_results_with_invocations[
                0
            ].eval_metric_result.threshold

            # 2. Calculate metric-level scores and status first
            scores = [
                m.eval_metric_result.score
                for m in eval_metric_results_with_invocations
                if m.eval_metric_result.score is not None
            ]

            if scores:
                average_score = statistics.mean(scores)
                # Use .value if EvalStatus is an Enum
                overall_eval_status = (
                    "PASSED" if average_score >= threshold else "FAILED"
                )
            else:
                average_score = None  # Or float('nan')
                overall_eval_status = "NOT_EVALUATED"

            # 3. Iterate through each invocation result to build the detailed rows
            for invocation_idx, per_invocation_result in enumerate(
                eval_metric_results_with_invocations, 1
            ):
                # Use .value for enums to get the string representation
                eval_status = per_invocation_result.eval_metric_result.eval_status

                all_results_data.append(
                    {
                        "overall_eval_status": overall_eval_status,
                        "metric_name": metric_name,
                        "average_score": average_score,
                        "line_score": per_invocation_result.eval_metric_result.score,
                        "threshold": threshold,
                        "invocation_number": (invocation_idx + 1) % num_runs,
                        "prompt": AgentEvaluator._convert_content_to_text(
                            per_invocation_result.expected_invocation.user_content
                        ),
                        "line_eval_status": eval_status,
                        "expected_response": AgentEvaluator._convert_content_to_text(
                            per_invocation_result.expected_invocation.final_response
                        ),
                        "actual_response": AgentEvaluator._convert_content_to_text(
                            per_invocation_result.actual_invocation.final_response
                        ),
                        "expected_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
                            per_invocation_result.expected_invocation.intermediate_data
                        ),
                        "actual_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
                            per_invocation_result.actual_invocation.intermediate_data
                        ),
                    }
                )

        if not all_results_data:
            return pd.DataFrame()  # Return an empty DataFrame if no results

        # 4. Create the final DataFrame and return it
        return pd.DataFrame(all_results_data)
