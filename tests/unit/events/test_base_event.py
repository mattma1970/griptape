import time
import pytest
from griptape.events import (
    StartPromptEvent,
    FinishPromptEvent,
    StartTaskEvent,
    FinishTaskEvent,
    StartActionSubtaskEvent,
    FinishActionSubtaskEvent,
    CompletionChunkEvent,
    StartStructureRunEvent,
    FinishStructureRunEvent,
    BaseEvent,
)
from tests.mocks.mock_event import MockEvent


class TestBaseEvent:
    def test_timestamp(self):
        dt = time.time()

        assert MockEvent(timestamp=dt).timestamp == dt
        assert MockEvent().timestamp >= dt

    def test_to_dict(self):
        assert "timestamp" in MockEvent().to_dict()

    def test_start_prompt_event_from_dict(self):
        dict_value = {
            "type": "StartPromptEvent",
            "timestamp": 123.0,
            "token_count": 10,
        }

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, StartPromptEvent)
        assert event.timestamp == 123
        assert event.token_count == 10

    def test_finish_prompt_event_from_dict(self):
        dict_value = {
            "type": "FinishPromptEvent",
            "timestamp": 123.0,
            "token_count": 10,
        }

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, FinishPromptEvent)
        assert event.timestamp == 123
        assert event.token_count == 10

    def test_start_task_event_from_dict(self):
        dict_value = {
            "type": "StartTaskEvent",
            "timestamp": 123.0,
            "task_id": "foo",
            "task_parent_ids": ["bar"],
            "task_child_ids": ["baz"],
            "task_input": {"type": "TextArtifact", "value": "foo"},
            "task_output": {"type": "TextArtifact", "value": "bar"},
        }

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, StartTaskEvent)
        assert event.timestamp == 123
        assert event.task_id == "foo"
        assert event.task_parent_ids == ["bar"]
        assert event.task_child_ids == ["baz"]
        assert event.task_input.value == "foo"
        assert event.task_output.value == "bar"

    def test_start_subtask_event_from_dict(self):
        dict_value = {
            "type": "StartActionSubtaskEvent",
            "timestamp": 123.0,
            "task_id": "foo",
            "task_parent_ids": ["bar"],
            "task_child_ids": ["baz"],
            "task_input": {"type": "TextArtifact", "value": "foo"},
            "task_output": {"type": "TextArtifact", "value": "bar"},
            "subtask_parent_task_id": "foo",
            "subtask_thought": "bar",
            "subtask_action_type": "baz",
            "subtask_action_name": "qux",
            "subtask_action_input": {"value": "quux"},
        }

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, StartActionSubtaskEvent)
        assert event.timestamp == 123
        assert event.task_id == "foo"
        assert event.task_parent_ids == ["bar"]
        assert event.task_child_ids == ["baz"]
        assert event.task_input.value == "foo"
        assert event.task_output.value == "bar"
        assert event.subtask_thought == "bar"
        assert event.subtask_action_type == "baz"
        assert event.subtask_action_name == "qux"
        assert event.subtask_action_input is not None
        assert event.subtask_action_input["value"] == "quux"

    def test_finish_task_event_from_dict(self):
        dict_value = {
            "type": "FinishTaskEvent",
            "timestamp": 123.0,
            "task_id": "foo",
            "task_parent_ids": ["bar"],
            "task_child_ids": ["baz"],
            "task_input": {"type": "TextArtifact", "value": "foo"},
            "task_output": {"type": "TextArtifact", "value": "bar"},
        }

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, FinishTaskEvent)
        assert event.timestamp == 123
        assert event.task_id == "foo"
        assert event.task_parent_ids == ["bar"]
        assert event.task_child_ids == ["baz"]
        assert event.task_input.value == "foo"
        assert event.task_output.value == "bar"

    def test_finish_subtask_event_from_dict(self):
        dict_value = {
            "type": "FinishActionSubtaskEvent",
            "timestamp": 123.0,
            "task_id": "foo",
            "task_parent_ids": ["bar"],
            "task_child_ids": ["baz"],
            "task_input": {"type": "TextArtifact", "value": "foo"},
            "task_output": {"type": "TextArtifact", "value": "bar"},
            "subtask_parent_task_id": "foo",
            "subtask_thought": "bar",
            "subtask_action_type": "baz",
            "subtask_action_name": "qux",
            "subtask_action_input": {"value": "quux"},
        }

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, FinishActionSubtaskEvent)
        assert event.timestamp == 123
        assert event.task_id == "foo"
        assert event.task_parent_ids == ["bar"]
        assert event.task_child_ids == ["baz"]
        assert event.task_input.value == "foo"
        assert event.task_output.value == "bar"
        assert event.subtask_thought == "bar"
        assert event.subtask_action_type == "baz"
        assert event.subtask_action_name == "qux"
        assert event.subtask_action_input is not None
        assert event.subtask_action_input["value"] == "quux"

    def test_start_structure_run_event_from_dict(self):
        dict_value = {"type": "StartStructureRunEvent", "timestamp": 123.0}

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, StartStructureRunEvent)
        assert event.timestamp == 123

    def test_finish_structure_run_event_from_dict(self):
        dict_value = {"type": "FinishStructureRunEvent", "timestamp": 123.0}

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, FinishStructureRunEvent)
        assert event.timestamp == 123

    def test_completion_chunk_event_from_dict(self):
        dict_value = {
            "type": "CompletionChunkEvent",
            "timestamp": 123.0,
            "token": "foo",
        }

        event = BaseEvent.from_dict(dict_value)

        assert isinstance(event, CompletionChunkEvent)
        assert event.token == "foo"

    def test_unsupported_from_dict(self):
        dict_value = {"type": "foo", "value": "foobar"}
        with pytest.raises(ValueError):
            BaseEvent.from_dict(dict_value)
