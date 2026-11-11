import pytest

from indonesian_hoax_checker.cli import TASKS, build_parser


def test_defaults():
    args = build_parser().parse_args(["Pemerintah menghentikan bantuan"])
    assert args.text == "Pemerintah menghentikan bantuan"
    assert args.task == "all"
    assert args.verbose is False


@pytest.mark.parametrize("task", TASKS)
def test_task_choices(task):
    assert build_parser().parse_args(["x", "--task", task]).task == task


def test_rejects_unknown_task():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--task", "summarize"])
