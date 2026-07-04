import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.gates.basic_gates import chat_structure_gate
from corpus_studio.gates.models import GateScope, GateStatus, GateThresholds
from corpus_studio.gates.runner import run_chat_gates

runner = CliRunner()


def _row(*roles: str) -> dict:
    return {"messages": [{"role": role, "content": f"{role} says hi"} for role in roles]}


# --- pure chat_structure_gate -----------------------------------------------

def test_clean_conversation_passes():
    result = chat_structure_gate([_row("system", "user", "assistant", "user", "assistant")], GateThresholds())
    assert result.status == GateStatus.PASS
    assert "well-formed" in result.message


def test_assistant_first_flags():
    result = chat_structure_gate([_row("assistant", "user", "assistant")], GateThresholds())
    assert result.status == GateStatus.WARN
    assert "start with an assistant turn" in result.observed


def test_dangling_user_flags():
    result = chat_structure_gate([_row("user", "assistant", "user")], GateThresholds())
    assert "end on a user turn" in result.observed


def test_zero_assistant_flags():
    result = chat_structure_gate([_row("system", "user")], GateThresholds())
    assert "have no assistant turn" in result.observed


def test_zero_user_flags():
    result = chat_structure_gate([_row("system", "assistant")], GateThresholds())
    assert "have no user turn" in result.observed


def test_consecutive_same_role_flags_but_tool_is_ok():
    duplicated = chat_structure_gate([_row("user", "user", "assistant")], GateThresholds())
    assert "repeat a role back-to-back" in duplicated.observed
    # user -> assistant -> tool -> assistant has no ADJACENT equal roles.
    tool_ok = chat_structure_gate([_row("user", "assistant", "tool", "assistant")], GateThresholds())
    assert "repeat a role back-to-back" not in tool_ok.observed


def test_system_misplaced_flags():
    two_systems = chat_structure_gate([_row("system", "user", "system", "assistant")], GateThresholds())
    assert "misplace the system message" in two_systems.observed
    late_system = chat_structure_gate([_row("user", "system", "assistant")], GateThresholds())
    assert "misplace the system message" in late_system.observed


def test_turn_count_bounds():
    too_few = chat_structure_gate([_row("user")], GateThresholds(min_chat_turns=2))
    assert "have too few turns" in too_few.observed
    too_many = chat_structure_gate(
        [_row("user", "assistant", "user", "assistant")], GateThresholds(max_chat_turns=2)
    )
    assert "have too many turns" in too_many.observed


def test_block_flag_flips_only_training_breaking_faults():
    malformed = [_row("assistant", "user", "assistant")]  # assistant-first = training-breaking
    assert chat_structure_gate(malformed, GateThresholds()).status == GateStatus.WARN
    assert chat_structure_gate(malformed, GateThresholds(block_chat_malformed=True)).status == GateStatus.BLOCK

    stylistic = [_row("user", "user", "assistant")]  # consecutive-same-role only = stylistic
    assert chat_structure_gate(stylistic, GateThresholds(block_chat_malformed=True)).status == GateStatus.WARN


def test_affected_row_numbers_are_one_based():
    rows = [_row("user", "assistant"), _row("assistant", "user", "assistant")]  # row 2 = assistant-first
    result = chat_structure_gate(rows, GateThresholds())
    assert result.affected == ["2"]


def test_no_conversations_warns_not_pass():
    result = chat_structure_gate([{"instruction": "not a chat row"}], GateThresholds())
    assert result.status == GateStatus.WARN
    assert "No 'messages' conversations" in result.message


def test_malformed_individual_messages_do_not_crash():
    rows = [{"messages": [{"role": "user"}, "not-a-dict", {"content": "no role"}]}]
    result = chat_structure_gate(rows, GateThresholds())  # must not raise
    assert result.status in (GateStatus.WARN, GateStatus.BLOCK, GateStatus.PASS)


# --- runner + CLI ------------------------------------------------------------

def test_run_chat_gates_scope_and_status():
    report = run_chat_gates([_row("assistant", "user", "assistant")])  # structurally off
    assert report.scope == GateScope.CHAT_SUITE
    assert report.overall_status == GateStatus.WARN


def test_run_chat_gates_clean_passes():
    report = run_chat_gates([_row("system", "user", "assistant")])
    assert report.overall_status == GateStatus.PASS


def test_cli_chat_gate(tmp_path: Path):
    rows_path = tmp_path / "chat.jsonl"
    rows_path.write_text(json.dumps(_row("system", "user", "assistant")) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["chat-gate", str(rows_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["scope"] == "chat_suite"
    assert payload["overall_status"] == "pass"


def test_cli_chat_gate_saves_report_and_stays_advisory(tmp_path: Path):
    rows_path = tmp_path / "chat.jsonl"
    # A structurally-broken conversation with the block flag on -> BLOCK verdict, but exit stays 0.
    rows_path.write_text(json.dumps(_row("assistant", "user")) + "\n", encoding="utf-8")
    (tmp_path / "gate_thresholds.json").write_text('{"block_chat_malformed": true}', encoding="utf-8")

    result = runner.invoke(app, ["chat-gate", str(rows_path), "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output  # advisory: verdict is in the report, not the exit code
    payload = json.loads(result.output)
    assert payload["overall_status"] == "block"
    assert (tmp_path / "gate_reports").exists()


def test_cli_chat_gate_rejects_non_object_line(tmp_path: Path):
    rows_path = tmp_path / "chat.jsonl"
    rows_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    result = runner.invoke(app, ["chat-gate", str(rows_path)])
    assert result.exit_code == 1


def test_new_chat_thresholds_round_trip(tmp_path: Path):
    from corpus_studio.gates.models import GATE_THRESHOLDS_FILENAME, load_gate_thresholds

    (tmp_path / GATE_THRESHOLDS_FILENAME).write_text(
        '{"min_chat_turns": 4, "max_chat_turns": 20, "block_chat_malformed": true}', encoding="utf-8"
    )
    thresholds = load_gate_thresholds(tmp_path)
    assert thresholds.min_chat_turns == 4
    assert thresholds.max_chat_turns == 20
    assert thresholds.block_chat_malformed is True
