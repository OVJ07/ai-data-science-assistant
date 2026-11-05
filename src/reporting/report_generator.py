"""
src/reporting/report_generator.py
───────────────────────────────────
Generates a downloadable HTML report from actual execution results.
No hardcoded values — all content comes from the session state.
"""

import base64
import datetime
import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPORT_CSS = """
<style>
  :root {
    --bg: #1e1e2e;
    --surface: #2a2a3e;
    --accent: #4F8EF7;
    --text: #e0e0f0;
    --muted: #a0a0b8;
    --success: #2ECC71;
    --warning: #F39C12;
    --danger: #E74C3C;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; padding: 32px; line-height: 1.6; }
  h1 { color: var(--accent); font-size: 2rem; margin-bottom: 0.5rem; }
  h2 { color: var(--accent); font-size: 1.4rem; margin: 2rem 0 0.75rem; border-bottom: 1px solid #444; padding-bottom: 4px; }
  h3 { color: var(--text); font-size: 1.1rem; margin: 1rem 0 0.5rem; }
  .meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }
  .card { background: var(--surface); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
  th { background: #333355; color: var(--accent); padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border-bottom: 1px solid #333; }
  tr:hover td { background: #2e2e45; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; }
  .badge-success { background: #1d5e3a; color: var(--success); }
  .badge-warning { background: #5e4e1d; color: var(--warning); }
  .badge-danger  { background: #5e1d1d; color: var(--danger); }
  .code { background: #111122; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 12px; overflow-x: auto; }
  .warn-box { background: #3e2e00; border-left: 4px solid var(--warning); padding: 10px 16px; border-radius: 4px; margin: 8px 0; }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-top: 12px; }
  .metric-card { background: #333355; border-radius: 8px; padding: 16px; text-align: center; }
  .metric-val { font-size: 1.8rem; font-weight: bold; color: var(--accent); }
  .metric-label { color: var(--muted); font-size: 0.8rem; margin-top: 4px; }
  .shap-row-pos { color: var(--success); }
  .shap-row-neg { color: var(--danger); }
  .toc a { color: var(--accent); text-decoration: none; display: block; padding: 2px 0; }
  .toc a:hover { text-decoration: underline; }
</style>
"""


class ReportGenerator:
    """Generates comprehensive HTML reports from session execution data."""

    def generate(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a complete HTML report.

        Parameters
        ----------
        report_data : dict with keys matching sections of the report
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sections = []

        sections.append(self._header(report_data, timestamp))
        sections.append(self._section_dataset_profile(report_data))
        sections.append(self._section_data_quality(report_data))
        sections.append(self._section_target(report_data))
        sections.append(self._section_query_generation(report_data))
        sections.append(self._section_rag_retrieval(report_data))
        sections.append(self._section_recommendations(report_data))
        sections.append(self._section_training(report_data))
        sections.append(self._section_evaluation(report_data))
        sections.append(self._section_comparison(report_data))
        sections.append(self._section_xai(report_data))
        sections.append(self._section_llm_explanation(report_data))
        sections.append(self._section_limitations(report_data))
        sections.append(self._footer(timestamp))

        body = "\n".join(s for s in sections if s)
        return (
            f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n"
            f"<title>AI Data Science Assistant - Report</title>\n{REPORT_CSS}\n"
            f"</head>\n<body>\n{body}\n</body>\n</html>"
        )

    # ── Section builders ──────────────────────────────────────────────────────

    def _header(self, data: Dict, timestamp: str) -> str:
        dataset_name = data.get("dataset_name", "Unknown Dataset")
        return f"""
<h1>AI Data Science Assistant - Analysis Report</h1>
<div class='meta'>
  Dataset: <strong>{html.escape(dataset_name)}</strong> &nbsp;|&nbsp;
  Generated: {timestamp} &nbsp;|&nbsp;
  <span class='badge badge-success'>Actual Execution Results</span>
</div>
"""

    def _section_dataset_profile(self, data: Dict) -> str:
        profile = data.get("profile")
        if not profile:
            return ""
        p = profile.to_dict() if hasattr(profile, "to_dict") else profile
        rows_html = ""
        info = {
            "Rows": p.get("n_rows"),
            "Columns": p.get("n_cols"),
            "Duplicates": p.get("n_duplicates"),
            "Numerical Features": len(p.get("numerical_features", [])),
            "Categorical Features": len(p.get("categorical_features", [])),
            "Constant Features": len(p.get("constant_features", [])),
            "High-Cardinality Features": len(p.get("high_cardinality_features", [])),
            "ID-like Features": len(p.get("id_like_features", [])),
        }
        for k, v in info.items():
            rows_html += f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"

        warnings_html = ""
        for w in p.get("warnings", []):
            warnings_html += f"<div class='warn-box'>{html.escape(str(w))}</div>"

        return f"""
<h2 id='profile'>Dataset Profile</h2>
<div class='card'>
  <table><thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>{rows_html}</tbody></table>
  {warnings_html}
</div>"""

    def _section_data_quality(self, data: Dict) -> str:
        profile = data.get("profile")
        if not profile:
            return ""
        p = profile.to_dict() if hasattr(profile, "to_dict") else profile
        if not p.get("has_missing"):
            return "<h2 id='quality'>Data Quality</h2><div class='card'><p>No missing values detected.</p></div>"

        missing = getattr(profile, "missing_percentages", {})
        rows = "".join(
            f"<tr><td>{col}</td><td>{pct:.2f}%</td></tr>"
            for col, pct in sorted(missing.items(), key=lambda x: -x[1])
        )
        return f"""
<h2 id='quality'>Data Quality</h2>
<div class='card'>
  <table><thead><tr><th>Column</th><th>Missing %</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>"""

    def _section_target(self, data: Dict) -> str:
        profile = data.get("profile")
        if not profile:
            return ""
        p = profile.to_dict() if hasattr(profile, "to_dict") else profile
        target = p.get("target_column", "N/A")
        problem = p.get("problem_type", "N/A")
        imbalanced = "Yes" if p.get("is_imbalanced") else "No"
        badge = "badge-warning" if p.get("is_imbalanced") else "badge-success"
        return f"""
<h2 id='target'>Target and Problem Type</h2>
<div class='card'>
  <div class='metric-grid'>
    <div class='metric-card'><div class='metric-val'>{html.escape(str(target))}</div><div class='metric-label'>Target Column</div></div>
    <div class='metric-card'><div class='metric-val'>{html.escape(str(problem))}</div><div class='metric-label'>Problem Type</div></div>
    <div class='metric-card'><div class='metric-val'>{p.get("n_classes", "N/A")}</div><div class='metric-label'>Classes</div></div>
    <div class='metric-card'><div class='metric-val'><span class='badge {badge}'>{imbalanced}</span></div><div class='metric-label'>Imbalanced</div></div>
  </div>
</div>"""

    def _section_rag_retrieval(self, data: Dict) -> str:
        docs = data.get("retrieved_docs", [])
        if not docs:
            return ""
        rows = ""
        for doc in docs:
            title = getattr(doc, "title", "Unknown")
            score = getattr(doc, "relevance_score", 0)
            snippet = html.escape(getattr(doc, "snippet", "")[:200])
            rows += f"<tr><td>{html.escape(title)}</td><td>{score:.4f}</td><td>{snippet}</td></tr>"
        return f"""
<h2 id='rag'>RAG Retrieval Results</h2>
<div class='card'>
  <table><thead><tr><th>Document</th><th>Relevance Score</th><th>Snippet</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>"""

    def _section_query_generation(self, data: Dict) -> str:
        """Report the actual strategy rather than implying an LLM was used."""
        result = data.get("query_gen_result")
        if not result:
            return ""
        strategy = html.escape(str(getattr(result, "strategy", "deterministic")).replace("_", " "))
        provider = html.escape(str(getattr(result, "provider_name", "none")))
        model = html.escape(str(getattr(result, "model_name", "")))
        reason = html.escape(str(getattr(result, "selection_reason", "")))
        fallback = getattr(result, "fallback_reason", "") if getattr(result, "fallback_occurred", False) else ""
        fallback_html = f"<div class='warn-box'>Fallback: {html.escape(str(fallback))}</div>" if fallback else ""
        return f"""
<h2 id='query-generation'>Query-generation strategy</h2>
<div class='card'>
  <p><strong>Strategy:</strong> {strategy}</p>
  <p><strong>Provider/model:</strong> {provider}{' / ' + model if model else ''}</p>
  <p><strong>Selection reason:</strong> {reason}</p>
  <p><strong>Queries generated:</strong> {getattr(result, 'total_queries', 0)}</p>
  {fallback_html}
</div>"""

    def _section_recommendations(self, data: Dict) -> str:
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        cards = ""
        for rec in recs:
            technique = html.escape(getattr(rec, "technique", ""))
            trigger = html.escape(getattr(rec, "trigger", ""))
            reason = html.escape(getattr(rec, "reason", ""))
            limit = html.escape(getattr(rec, "limitations", ""))
            cards += f"""
<div class='card'>
  <h3>{technique}</h3>
  <p><strong>Trigger:</strong> {trigger}</p>
  <p><strong>Reason:</strong> {reason}</p>
  {"<p><strong>Limitations:</strong> " + limit + "</p>" if limit else ""}
</div>"""
        return f"<h2 id='recs'>Preprocessing Recommendations</h2>{cards}"

    def _section_training(self, data: Dict) -> str:
        result = data.get("training_result")
        if not result:
            return ""
        best = result.best_params or {}
        params_html = "".join(
            f"<tr><td>{html.escape(k)}</td><td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in best.items()
        )
        cv_score = f"{result.best_cv_score:.4f}" if result.best_cv_score is not None else "N/A"
        cv_std = f"+/-{result.cv_score_std:.4f}" if result.cv_score_std is not None else ""
        return f"""
<h2 id='training'>Model Training and Hyperparameter Search</h2>
<div class='card'>
  <div class='metric-grid'>
    <div class='metric-card'><div class='metric-val'>{html.escape(result.model_name)}</div><div class='metric-label'>Model</div></div>
    <div class='metric-card'><div class='metric-val'>{cv_score} {cv_std}</div><div class='metric-label'>Best CV Score ({result.cv_scoring})</div></div>
    <div class='metric-card'><div class='metric-val'>{result.n_param_combinations}</div><div class='metric-label'>Configurations Tried</div></div>
    <div class='metric-card'><div class='metric-val'>{"RAG-Guided" if result.is_rag_guided else "Baseline"}</div><div class='metric-label'>Strategy</div></div>
  </div>
  {"<h3>Best Hyperparameters</h3><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>" + params_html + "</tbody></table>" if params_html else ""}
</div>"""

    def _section_evaluation(self, data: Dict) -> str:
        result = data.get("training_result")
        if not result or not result.test_metrics:
            return ""
        metrics_html = ""
        for key, val in result.test_metrics.items():
            if key == "confusion_matrix" or val is None or isinstance(val, (list, tuple)):
                continue
            metrics_html += f"""
<div class='metric-card'>
  <div class='metric-val'>{val:.4f}</div>
  <div class='metric-label'>{key.replace("_", " ").title()}</div>
</div>"""
        cm = result.test_metrics.get("confusion_matrix")
        cm_html = ""
        if cm:
            class_names = result.test_metrics.get("class_names", [str(i) for i in range(len(cm))])
            header = "".join(f"<th>{html.escape(str(c))}</th>" for c in class_names)
            cm_rows = ""
            for i, row in enumerate(cm):
                cells = "".join(f"<td><strong>{v}</strong></td>" for v in row)
                label = html.escape(str(class_names[i] if i < len(class_names) else i))
                cm_rows += f"<tr><td>{label}</td>{cells}</tr>"
            cm_html = f"""
<h3>Confusion Matrix</h3>
<table><thead><tr><th>Actual / Predicted</th>{header}</tr></thead><tbody>{cm_rows}</tbody></table>"""

        return f"""
<h2 id='eval'>Model Evaluation (Test Set)</h2>
<div class='card'>
  <div class='metric-grid'>{metrics_html}</div>
  {cm_html}
</div>"""

    def _section_comparison(self, data: Dict) -> str:
        comp = data.get("comparison_result")
        if not comp:
            return ""
        b = comp.baseline_result
        r = comp.rag_result
        if not (b and r):
            return ""

        def fmt_metrics(metrics):
            return "".join(
                f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v:.4f}</td></tr>"
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None and k != "confusion_matrix"
            )

        def details(label, training):
            cv = "N/A" if training.best_cv_score is None else f"{training.best_cv_score:.4f}"
            std = "N/A" if training.cv_score_std is None else f"{training.cv_score_std:.4f}"
            params = html.escape(str(training.best_params or "Default estimator parameters"))
            return (
                f"<p><strong>CV ({html.escape(training.cv_scoring)}):</strong> {cv} ± {std}</p>"
                f"<p><strong>Configurations evaluated:</strong> {training.n_param_combinations}</p>"
                f"<p><strong>Best parameters:</strong> <code>{params}</code></p>"
            )

        agreement = "Unavailable"
        if getattr(comp, "prediction_agreement", None) is not None:
            agreement = f"{comp.prediction_agreement:.1%}; identical arrays: {'yes' if comp.predictions_identical else 'no'}"
        elif getattr(comp, "max_prediction_difference", None) is not None:
            agreement = f"max absolute difference {comp.max_prediction_difference:.6g}; identical arrays: {'yes' if comp.predictions_identical else 'no'}"

        return f"""
<h2 id='comparison'>Baseline vs RAG-Guided Comparison</h2>
<div class='card'>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
    <div>
      <h3>Baseline</h3>
      {details("Baseline", b)}
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{fmt_metrics(b.test_metrics)}</tbody></table>
    </div>
    <div>
      <h3>RAG-Guided</h3>
      {details("RAG-guided", r)}
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{fmt_metrics(r.test_metrics)}</tbody></table>
    </div>
  </div>
  <p><strong>Same held-out observations:</strong> {'yes' if getattr(comp, 'test_indices_match', False) else 'no'}</p>
  <p><strong>Prediction agreement:</strong> {agreement}</p>
  <div class='warn-box' style='margin-top:16px'>{html.escape(comp.honest_assessment)}</div>
</div>"""

    def _section_xai(self, data: Dict) -> str:
        xai = data.get("xai_result")
        if not xai or not xai.global_importance:
            return ""
        rows = ""
        for feat, val in sorted(xai.global_importance.items(), key=lambda x: -x[1])[:15]:
            rows += f"<tr><td>{html.escape(feat)}</td><td>{val:.4f}</td></tr>"

        local_html = ""
        if xai.local_shap:
            local_rows = ""
            for feat, val in sorted(xai.local_shap.items(), key=lambda x: -abs(x[1]))[:10]:
                css_class = "shap-row-pos" if val > 0 else "shap-row-neg"
                direction = "increases" if val > 0 else "decreases"
                local_rows += f"<tr class='{css_class}'><td>{html.escape(feat)}</td><td>{val:+.4f}</td><td>{direction}</td></tr>"
            local_html = f"""
<h3>Local Explanation (Observation #{xai.observation_index})</h3>
<p>Prediction: <strong>{xai.prediction}</strong>
{"| P=" + f"{xai.prediction_probability:.4f}" if xai.prediction_probability else ""}</p>
<table><thead><tr><th>Feature</th><th>SHAP Value</th><th>Direction</th></tr></thead>
<tbody>{local_rows}</tbody></table>"""

        return f"""
<h2 id='xai'>Explainable AI (SHAP)</h2>
<div class='card'>
  <p><strong>Method:</strong> {html.escape(xai.shap_method)}</p>
  <h3>Global Feature Importance (Mean |SHAP|)</h3>
  <table><thead><tr><th>Feature</th><th>Mean |SHAP|</th></tr></thead>
  <tbody>{rows}</tbody></table>
  {local_html}
</div>"""

    def _section_llm_explanation(self, data: Dict) -> str:
        explanation = data.get("llm_explanation")
        if not explanation:
            return ""
        provider = data.get("llm_provider", "Unknown")
        return f"""
<h2 id='llm'>RAG-Grounded Explanation</h2>
<div class='card'>
  <p><strong>Provider:</strong> {html.escape(provider)}</p>
  <div class='code'>{html.escape(str(explanation))}</div>
</div>"""

    def _section_limitations(self, data: Dict) -> str:
        return """
<h2 id='limits'>Limitations</h2>
<div class='card'>
  <ul style='padding-left:20px;line-height:2'>
    <li>All metrics are computed on a single held-out test split. Results may vary with different random seeds.</li>
    <li>SHAP values reflect model associations, not causal relationships.</li>
    <li>RAG retrieval quality depends on the relevance of knowledge-base documents to the dataset.</li>
    <li>Hyperparameter search is limited to a predefined grid; optimal configurations may exist outside this grid.</li>
    <li>Class imbalance handling via class_weight does not create new minority samples.</li>
    <li>This report is generated automatically and should be reviewed by a domain expert.</li>
  </ul>
</div>"""

    def _footer(self, timestamp: str) -> str:
        return f"""
<hr style='margin-top:40px;border-color:#444'>
<p style='color:#666;font-size:0.8rem;text-align:center;padding:16px'>
  AI Data Science Assistant - Generated {timestamp} - All results from actual execution
</p>"""
