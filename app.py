from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import SchemaError, assess_distribution, predict_rows, validate_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "wine_pipeline.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
EVALUATION_PATH = ROOT / "models" / "evaluation.json"

LABEL_ZH = {
    "alcohol": "酒精含量", "malic_acid": "苹果酸", "ash": "灰分",
    "alcalinity_of_ash": "灰分碱度", "magnesium": "镁",
    "total_phenols": "总酚", "flavanoids": "黄酮类",
    "nonflavanoid_phenols": "非黄酮酚", "proanthocyanins": "原花青素",
    "color_intensity": "颜色强度", "hue": "色调",
    "od280/od315_of_diluted_wines": "稀释酒 OD 比值", "proline": "脯氨酸",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource
def load_artifacts():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256(MODEL_PATH)
    if actual_hash != meta["model_sha256"]:
        raise RuntimeError("模型文件哈希与元数据不一致，请重新训练")
    model = joblib.load(MODEL_PATH)
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    return model, meta, evaluation


st.set_page_config(page_title="WineLens 葡萄酒分类", page_icon="◈", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1180px; padding-top: 2.2rem;}
  [data-testid="stMetric"] {background:#f7f7fb;border:1px solid #e8e8f0;padding:14px 16px;border-radius:14px;}
  .hero {padding:18px 0 24px}.hero h1{font-size:2.6rem;margin:0;letter-spacing:-1px}.hero p{color:#667085;font-size:1.05rem}
  .result-card {background:linear-gradient(135deg,#332e78,#6256d9);color:white;padding:24px;border-radius:18px;margin:10px 0 18px}
  .result-card small{opacity:.78}.result-card h2{margin:.2rem 0 0;color:white}
  .soft-note{color:#667085;font-size:.88rem}.stButton>button{border-radius:10px;font-weight:700}
</style>
<div class="hero"><h1>WineLens 智能分类</h1><p>根据 13 项化学测量值识别 Wine 数据集中的栽培品种，并提示输入是否偏离训练分布。</p></div>
""", unsafe_allow_html=True)

try:
    model, meta, evaluation = load_artifacts()
except Exception as error:
    st.error(f"模型加载失败：{error}")
    st.stop()

top1, top2, top3, top4 = st.columns(4)
metrics = meta["final_metrics"]
top1.metric("模型版本", meta["model_version"])
top2.metric("最终 Accuracy", f"{metrics['accuracy']:.3f}")
top3.metric("Macro F1", f"{metrics['f1_macro']:.3f}")
top4.metric("测试样本", metrics["test_samples"])

tab_predict, tab_evidence = st.tabs(["单条预测", "模型证据"])

with tab_predict:
    st.markdown("### 输入化学测量值")
    st.caption("默认值为训练集各特征中位数。输入训练范围外的值时，系统会给出分布越界提示。")
    values = {}
    with st.form("predict_form"):
        columns = st.columns(3)
        for index, name in enumerate(meta["feature_names"]):
            stat = meta["training_stats"][name]
            span = max(stat["max"] - stat["min"], 0.01)
            with columns[index % 3]:
                values[name] = st.number_input(
                    LABEL_ZH.get(name, name),
                    min_value=float(stat["min"] - span), max_value=float(stat["max"] + span),
                    value=float(stat["median"]), format="%.4f",
                    help=f"训练范围 {stat['min']:.4g} - {stat['max']:.4g}",
                    key=f"feature_{name}",
                )
        submitted = st.form_submit_button("开始预测", type="primary", use_container_width=True)

    if submitted:
        frame = pd.DataFrame([values])
        try:
            clean = validate_frame(frame, meta)
            prediction = predict_rows(model, clean, meta).iloc[0]
            probabilities = model.predict_proba(clean)[0]
            drift = assess_distribution(clean.iloc[0], meta)
            st.markdown(
                f'<div class="result-card"><small>预测结果</small><h2>{prediction["prediction"]}</h2>'
                f'<div>最高类别概率 {prediction["confidence"]:.1%} · 模型 {meta["model_name"]}</div></div>',
                unsafe_allow_html=True,
            )
            probability_table = pd.DataFrame({
                "类别": meta["class_names"], "概率": probabilities,
            }).set_index("类别")
            st.bar_chart(probability_table, horizontal=True)
            if drift.level == "normal":
                st.success("输入位于训练数据范围内，未发现分布越界。")
            elif drift.level == "medium":
                st.warning("输入接近训练分布边缘。预测仅供课程演示。\n\n" + "\n\n".join(drift.details))
            else:
                st.error("多个特征超出训练分布，预测可靠性可能明显下降。\n\n" + "\n\n".join(drift.details))
        except SchemaError as error:
            st.error(str(error))

    st.markdown('<p class="soft-note">限制：本模型不能用于食品安全、商业定价或产地认证。概率是模型输出，不是事实确定性。</p>', unsafe_allow_html=True)

with tab_evidence:
    st.markdown("### 同协议模型比较")
    comparison = pd.DataFrame(evaluation["comparison"])
    display = comparison[["model", "cv_accuracy_mean", "cv_precision_macro_mean", "cv_recall_macro_mean", "cv_f1_macro_mean", "cv_f1_macro_std", "cv_fit_time_mean_seconds"]].copy()
    display.columns = ["模型", "Accuracy", "Precision", "Recall", "Macro F1", "F1 标准差", "平均拟合秒数"]
    st.dataframe(display.style.format({col: "{:.4f}" for col in display.columns[1:]}), hide_index=True, use_container_width=True)
    st.info(f"选择结果：{evaluation['selected_model']}。只使用训练集交叉验证选择模型，保留测试集仅做一次最终评价。")
    st.markdown("### 最终混淆矩阵")
    cm = pd.DataFrame(metrics["confusion_matrix"], index=[f"真实 {x}" for x in meta["class_names"]], columns=[f"预测 {x}" for x in meta["class_names"]])
    st.dataframe(cm, use_container_width=True)
    st.markdown("### 模型追踪")
    st.code(f"Git Commit: {meta['git_commit']}\nscikit-learn: {meta['sklearn_version']}\nSHA256: {meta['model_sha256']}\n训练时间: {meta['trained_at_utc']}")

st.divider()
st.caption("实验 4 · 侯宇晴 · 软件2304 · 23120506154")

