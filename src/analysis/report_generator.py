from __future__ import annotations

from src.config.schemas import ExperimentSummary
from src.config.settings import AppSettings
from src.utils.io import write_json


def generate_analysis_report(settings: AppSettings, aggregates: dict) -> ExperimentSummary:
    overall_best_pipeline = "rag" if (aggregates.get("pairwise_faithfulness_gain") or 0.0) >= 0 else "baseline"
    report = ExperimentSummary(
        experiment_id=settings.experiment_id,
        course_name=settings.dataset.course_name,
        overall_best_pipeline=overall_best_pipeline,
        faithfulness_gain=aggregates["faithfulness_gain"],
        answer_correctness_delta=aggregates.get("answer_correctness_delta"),
        factscore_init_score_delta=aggregates.get("factscore_init_score_delta"),
        unsupported_claim_rate_delta=aggregates.get("unsupported_claim_rate_delta"),
        answer_relevance_delta=aggregates["answer_relevance_delta"],
        factscore_score_delta=aggregates.get("factscore_score_delta"),
        adversarial_refusal_rate_baseline=aggregates["adversarial_refusal_rate_baseline"],
        adversarial_refusal_rate_rag=aggregates["adversarial_refusal_rate_rag"],
        question_type_breakdown={"rows": aggregates["question_type_breakdown"]},
        notes=[
            "Non-RAG baseline never receives retrieval contexts during generation.",
            (
                "Baseline evaluation contexts come from "
                f"{settings.evaluation.reference_context_strategy} and are used only during evaluation."
            ),
            "Manual review candidates are exported separately to support spot-checking low-faithfulness rows.",
            (
                "FActScore evaluation is "
                + ("enabled." if settings.evaluation.factscore_enabled else "disabled.")
            ),
        ],
    )
    write_json(settings.analysis_report_path, report.model_dump(mode="json"))
    return report
