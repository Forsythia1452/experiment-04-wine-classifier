from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()


def test_app_runs_without_exception():
    app = run_app()
    assert not app.exception


def test_app_shows_four_summary_metrics():
    app = run_app()
    assert len(app.metric) == 4
    assert app.metric[0].label == "模型版本"


def test_app_has_prediction_and_evidence_tabs():
    app = run_app()
    assert [tab.label for tab in app.tabs] == ["单条预测", "模型证据"]


def test_app_has_all_thirteen_feature_inputs():
    app = run_app()
    assert len(app.number_input) == 13


def test_default_form_can_predict():
    app = run_app()
    app.button[0].click().run()
    assert not app.exception
    assert any("输入位于训练数据范围内" in item.value for item in app.success)


def test_app_displays_model_comparison_and_confusion_matrix():
    app = run_app()
    assert len(app.dataframe) >= 2
    assert any("只使用训练集交叉验证" in item.value for item in app.info)

