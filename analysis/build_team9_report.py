"""Build Team 9's original consolidated PathSnap technical report PDF.

Only archived team experiments, reproducible local re-evaluations, and supplied
demo screenshots are used. No text or results from another team's report enter
this document.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image,
                                KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "results"
FIG = RESULTS / "figures"
EVIDENCE = ROOT / "analysis" / "evidence"
OUT = ROOT / "docs" / "PathSnap_Team9_Technical_Report.pdf"

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

NAVY = colors.HexColor("#102a43")
TEAL = colors.HexColor("#087e8b")
INK = colors.HexColor("#172432")
MUTED = colors.HexColor("#526477")
PALE = colors.HexColor("#eef5f7")
LINE = colors.HexColor("#d7e2e7")

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 48
TOP = 48
BOTTOM = 48
WIDTH = PAGE_W - LEFT - RIGHT

STYLES = {
    "title": ParagraphStyle("Title", fontName="Arial-Bold", fontSize=29,
                            leading=34, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("Subtitle", fontName="Arial", fontSize=14,
                               leading=20, textColor=MUTED, spaceAfter=18),
    "h1": ParagraphStyle("H1", fontName="Arial-Bold", fontSize=16,
                         leading=20, textColor=INK, spaceBefore=20,
                         spaceAfter=9, keepWithNext=True),
    "h2": ParagraphStyle("H2", fontName="Arial-Bold", fontSize=11.5,
                         leading=15, textColor=INK, spaceBefore=12,
                         spaceAfter=5, keepWithNext=True),
    "body": ParagraphStyle("Body", fontName="Arial", fontSize=9.2,
                           leading=13.2, textColor=INK, spaceAfter=7),
    "small": ParagraphStyle("Small", fontName="Arial", fontSize=8,
                            leading=11, textColor=MUTED, spaceAfter=5),
    "caption": ParagraphStyle("Caption", fontName="Arial", fontSize=8,
                              leading=11, textColor=MUTED, spaceBefore=3,
                              spaceAfter=10),
    "table": ParagraphStyle("TableCell", fontName="Arial", fontSize=7.3,
                            leading=9.8, textColor=INK),
    "tablehead": ParagraphStyle("TableHead", fontName="Arial-Bold", fontSize=7.3,
                                leading=9.8, textColor=colors.white),
    "meta": ParagraphStyle("Meta", fontName="Arial", fontSize=9,
                           leading=13, textColor=MUTED, spaceAfter=4),
}


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def heading(story: list, number: str, text: str, level: int = 1):
    story.append(p(f"{number} {text}", "h1" if level == 1 else "h2"))


def para(story: list, text: str):
    story.append(p(text))


def tab(story: list, rows: list[list[str]], widths: list[float], *, keep=False):
    assert abs(sum(widths) - WIDTH) < 1
    prepared = []
    for idx, row in enumerate(rows):
        prepared.append([p(str(cell), "tablehead" if idx == 0 else "table") for cell in row])
    t = Table(prepared, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), .35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    parts = [t, Spacer(1, 8)]
    if keep:
        story.append(KeepTogether(parts))
    else:
        story.extend(parts)


def figure(story: list, path: Path, caption: str, max_height: float = 270):
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(WIDTH / w, max_height / h)
    story.append(KeepTogether([
        Image(str(path), width=w * scale, height=h * scale),
        p(caption, "caption"),
    ]))


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=LEFT,
                         rightMargin=RIGHT, topMargin=TOP, bottomMargin=BOTTOM,
                         title="PathSnap Technical Report Team 9",
                         author="PathSnap Team 9")
        frame = Frame(LEFT, BOTTOM, WIDTH, PAGE_H - TOP - BOTTOM,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="Report", frames=frame,
                                           onPage=self.draw_page))

    @staticmethod
    def draw_page(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(.6)
            canvas.line(LEFT, PAGE_H - 32, PAGE_W - RIGHT, PAGE_H - 32)
            canvas.setFont("Arial-Bold", 8)
            canvas.setFillColor(NAVY)
            canvas.drawString(LEFT, PAGE_H - 26, "PATHSNAP  |  TEAM 9")
            canvas.setFont("Arial", 8)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 26, "Technical report")
            canvas.line(LEFT, 34, PAGE_W - RIGHT, 34)
            canvas.drawString(LEFT, 22, "18 November 2026")
            canvas.drawRightString(PAGE_W - RIGHT, 22, str(doc.page))
        canvas.restoreState()


def load_results():
    xgb = json.loads((RESULTS / "xgboost_verified.json").read_text())
    mlp = json.loads((RESULTS / "mlp_verified.json").read_text())
    mlp_run = json.loads((ROOT / "analysis" / "mlp_shared" / "metrics.json").read_text())
    tuning = ROOT / "outputs" / "xgboost" / "hyperparameter_tuning"
    xgb_test = json.loads((tuning / "student_test_metrics.json").read_text())
    cv = pd.read_csv(tuning / "cv_results.csv")
    cv_mean = cv.loc[cv.fold == "Mean"].iloc[0]
    others = {name: json.loads((RESULTS / f"{name}.json").read_text())
              for name in ("random_forest", "svm", "knn")}
    ablation = pd.read_csv(RESULTS / "hmm_ablation.csv", dtype={"trip_id": str})
    return xgb, mlp, mlp_run, xgb_test, cv_mean, others, ablation


def pct(x):
    return f"{x * 100:.2f}%"


def build() -> None:
    xgb, mlp, mlp_run, xgb_test, cv_mean, others, ablation = load_results()
    data_summary = json.loads((ROOT / "analysis" / "eda" / "dataset_summary.json").read_text())
    story: list = []

    # Cover
    story.append(Spacer(1, 42))
    story.append(p("PathSnap Technical Report", "title"))
    story.append(p("GPS map matching with XGBoost and a directed HMM decoder", "subtitle"))
    story.append(p("Team 9  |  Machine Learning Hackathon  |  18 November 2026", "meta"))
    story.append(p("Repository: <link href='https://github.com/shibishetty77/ML-HACKATHON' color='#087e8b'>github.com/shibishetty77/ML-HACKATHON</link>", "meta"))
    story.append(Spacer(1, 12))
    story.append(p("Team 9 members", "h2"))
    tab(story, [
        ["Student ID", "Name"],
        ["PES2UG24CS440", "SANDEEP H"],
        ["PES2UG24CS448", "SATVIK DAS"],
        ["PES2UG24CS467", "SHIBI R SHETTY"],
        ["PES2UG25CS818", "PRAJWAL G T"],
    ], [WIDTH * .42, WIDTH * .58])
    story.append(Spacer(1, 12))
    para(story, "This report records Team 9's five candidate-scoring model families, the common-data comparison, Bayesian tuning of the selected XGBoost model, directed HMM route reconstruction, geographical visualisation, and the two-laptop evaluator demo. All numerical claims below come from archived training outputs or the reproducible scripts listed in Section 9.")
    para(story, f"<b>Main result.</b> Tuned XGBoost reaches {pct(xgb['conditional_top1'])} conditional Top-1 and {pct(xgb['end_to_end_top1'])} end-to-end Top-1 on the 10 held-out student-test trips. In a separate 10-trip evaluator ablation, route arc F1 rises from {ablation.classifier_arc_f1.mean():.3f} for independent XGBoost choices to {ablation.hmm_arc_f1.mean():.3f} after HMM decoding. These are different evaluation units and are not interchangeable.")
    story.append(Spacer(1, 18))
    tab(story, [
        ["What is measured", "Evaluation partition", "Key boundary"],
        ["Candidate Macro F1 and Top-1", "80 train trips in 5 folds; 10 student-test trips", "Weak pointwise route alignment; candidate CSV omits reference misses"],
        ["Route arc F1 and continuity", "10 evaluator trips", "Reference .route is an arc sequence; no pointwise labels"],
        ["Live API and map", "Two laptops on one LAN", "Server runs XGBoost + HMM; client displays returned route"],
    ], [130, 155, WIDTH - 285])
    story.append(Spacer(1, 15))
    para(story, "The five-model table is a fixed-configuration re-evaluation on the same four-feature candidate tables and trip folds. It keeps the original notebook experiments visible, while preventing different sampling and feature pipelines from being presented as one fair ranking.")
    story.append(PageBreak())

    # 1. Dataset
    heading(story, "2.2", "Dataset understanding and exploratory analysis")
    para(story, "Each trip supplies a time-ordered GPS track, directed road nodes and arcs, and an ordered reference arc sequence. The training split contains 80 trips; student-test contains 10. Evaluator trips are reserved for the final route demo and are never used for fitting or hyperparameter search. The route sequence has no one-to-one timestamp labels, so supervised point labels are generated by the documented monotone alignment in the shared preprocessing code.")
    train = data_summary["train"]
    test = data_summary["student_test"]
    tab(story, [
        ["Raw-data measure", "Train", "Student-test"],
        ["Trips / GPS observations", f"{train['trips']} / {train['observations']:,}", f"{test['trips']} / {test['observations']:,}"],
        ["Trip-local road nodes / directed arcs", f"{train['nodes']:,} / {train['arcs']:,}", f"{test['nodes']:,} / {test['arcs']:,}"],
        ["Missing or nonfinite raw values", str(train["missing_or_nonfinite_values"]), str(test["missing_or_nonfinite_values"])],
        ["Time gaps greater than 10 s", f"{train['gaps_gt_10s']:,}", f"{test['gaps_gt_10s']:,}"],
        ["Zero-length directed arcs", str(train["zero_length_arcs"]), str(test["zero_length_arcs"])],
    ], [WIDTH * .46, WIDTH * .27, WIDTH * .27])
    para(story, "The raw files have no nonfinite coordinates or timestamps, but time gaps and zero-length road arcs require explicit handling. Trip lengths vary widely: the training median is 1,529 observations and the maximum is 19,313. A 30-second gap resets motion features and the decoder; the first point and near-stationary moves receive defined speed/heading fallbacks. These choices avoid carrying a stale direction across a break.")
    figure(story, ROOT / "analysis" / "eda" / "eda_overview.png",
           "Figure 1. Team 9 raw-data audit: trip size, road-network size and sampling intervals for train and student-test. Source: analysis/eda/dataset_summary.json and the supplied raw ZIP.", 270)
    heading(story, "2.2.1", "Candidate generation and features", level=2)
    para(story, "The common training pipeline searches directed road arcs within 50 m of each GPS point and retains up to 10 candidates. It projects the point onto each arc and computes four model inputs. Trip ID, observation ID, candidate ID, reference arc ID and binary label remain metadata; they do not enter the classifier. Tree models use the raw numeric features. MLP, SVM and KNN fit a scaler on training rows only for each fold.")
    tab(story, [
        ["Feature", "Meaning and reason"],
        ["distance_to_segment_m", "Point-to-arc distance; a close road is more plausible when GPS noise is modest."],
        ["projection_fraction", "Clamped position along the directed arc; distinguishes endpoint and interior matches."],
        ["observation_speed_mps", "Motion from consecutive GPS points and elapsed time; helps reject implausible movement."],
        ["heading_difference_rad", "Absolute wrapped difference between vehicle movement and arc bearing; uses direction."],
    ], [165, WIDTH - 165])
    para(story, "The common candidate CSVs have no missing values in these four numeric inputs. Of 184,319 reference-aligned training observations, 179,002 retain the reference candidate; the corresponding student-test counts are 16,985 and 16,424. The candidate tables contain only reference-hit groups. We therefore distinguish conditional Top-1 (among included groups) from end-to-end Top-1 (all aligned observations, counting 561 student-test reference misses as wrong). All pointwise results depend on weak route alignment rather than independently recorded road truth.")
    figure(story, ROOT / "analysis" / "eda" / "trajectory_student_test_00000000.png",
           "Figure 2. Student-test trip 00000000: GPS observations against local directed road geometry and supplied reference route. This is exploratory geometry, not a model score.", 230)

    # 2. Comparison
    heading(story, "3.", "Five-model experimentation and comparison")
    para(story, "The common comparison fixes the four inputs, the 80-trip training split, the five archived trip-grouped folds, the 10 student-test trips, and the metric code. Candidate Macro F1 uses a 0.5 probability threshold, except linear SVM uses its zero decision boundary. Top-1 chooses the highest score within each observation. The continuity column is classifier-only: consecutive covered GPS rows within 30 seconds must remain on one arc or move to a directly adjacent directed arc. It is not the later HMM continuity measure.")
    heading(story, "3.1", "Overall model comparison", level=2)
    configs = [
        ["XGBoost", "Boosted trees", "550 trees; depth 5; learning rate 0.0282"],
        ["MLP", "Neural network", "4-32-16-1; ReLU; Adam; alpha 0.001"],
        ["Random Forest", "Bagged trees", "120 trees; depth 18; min leaf 2"],
        ["Linear SVM", "Margin classifier", "LinearSVC; C 1; balanced classes"],
        ["KNN", "Instance based", "k 13; Manhattan; distance weighted"],
    ]
    tab(story, [["Model", "Family", "Fixed configuration used in common comparison"], *configs],
        [105, 105, WIDTH - 210])
    rows = [["Model", "CV F1", "Test Top-1", "Continuity", "Test F1", "Log loss", "Infer ms"]]
    records = [
        ("XGBoost", float(cv_mean.macro_f1), xgb, xgb_test["inference_time_s"]),
        ("MLP", mlp_run["cross_validation_mean_and_sd"]["macro_f1"]["mean"],
         mlp, mlp_run["student_test_inference_seconds"]),
    ]
    for name, cv_f1, metrics, timing in records:
        rows.append([name, f"{cv_f1:.4f}", pct(metrics["conditional_top1"]),
                     pct(metrics["classifier_route_continuity"]),
                     f"{metrics['candidate_macro_f1']:.4f}",
                     f"{metrics['log_loss']:.4f}", f"{timing * 1000:.1f}"])
    for name, key in [("Random Forest", "random_forest"), ("Linear SVM", "svm"), ("KNN", "knn")]:
        run = others[key]
        m = run["student_test"]
        rows.append([name, f"{run['cv_macro_f1_mean']:.4f}", pct(m["conditional_top1"]),
                     pct(m["classifier_route_continuity"]),
                     f"{m['candidate_macro_f1']:.4f}",
                     f"{m['log_loss']:.4f}" if m["log_loss"] is not None else "N/A - no probability",
                     f"{m['inference_seconds'] * 1000:.1f}"])
    tab(story, rows, [84, 53, 70, 75, 57, 74, WIDTH - 413])
    para(story, "Top-1 is conditional on the reference road appearing in the archived candidate table; the common student-test end-to-end Top-1 values are XGBoost 80.11%, MLP 79.89%, Random Forest 80.06%, SVM 73.94% and KNN 77.09%. Candidate F1 scores rows, not whole GPS observations. SVM has no log loss because the linear margin model emits no probability. Timing covers only the model call; XGBoost and MLP timings come from archived runs in different environments, so the timing column is descriptive rather than a controlled speed ranking.")
    para(story, "<b>Model choice.</b> Tuned XGBoost has the highest mean CV Macro F1 (0.8559), highest student-test candidate Macro F1 (0.8759), and lowest reported student-test log loss (0.1581). Random Forest's conditional Top-1 is only 0.0005 lower, but its F1 and log loss are worse in this fixed run. XGBoost is the model deployed in the server.")

    # 3. Individual models
    heading(story, "3.2", "Individual model findings", level=2)
    heading(story, "3.2.1", "XGBoost", level=2)
    para(story, "Boosted trees can combine road distance, projected position, speed and heading without scaling. The selected binary-logistic model uses 550 trees, depth 5, learning rate 0.0282, subsampling 0.743 and column sampling 0.982. It attains 0.8559 mean CV Macro F1 and 0.8759 student-test Macro F1, with 82.85% conditional Top-1 and 0.1581 log loss. Its strength is nonlinear ranking with fast batched scoring; its limitation is that candidate retrieval and weak labels bound the result. The held-out score exceeds the CV mean, so this run shows no obvious generalization collapse, but a train-versus-validation learning curve was not archived and overfitting cannot be ruled out from this comparison alone.")
    heading(story, "3.2.2", "Multilayer Perceptron", level=2)
    para(story, "The comparable MLP is a 4-input, 32- and 16-unit ReLU network with Adam, learning rate 0.001, alpha 0.001, batch size 2,048 and training-fold scaling. Binary cross-entropy is optimized through the classifier's log-loss objective. It records 0.8536 mean CV Macro F1, 0.8752 student-test Macro F1, 82.62% conditional Top-1 and 0.1597 log loss. Its benefit is smooth nonlinear scoring with a small model file. It is sensitive to scaling and feature drift. Recorded validation loss falls and then flattens across 60 epochs; there is no late sustained rise in these folds, although fold-to-fold performance varies.")
    figure(story, FIG / "mlp_learning_curves.png",
           "Figure 3. Actual per-epoch MLP training objective and held-out-trip validation log loss for the common four-feature run. No artificial curves are used for the non-epoch models.", 220)
    heading(story, "3.2.3", "Random Forest", level=2)
    para(story, "Random Forest averages independent decision trees, which is suitable for nonlinear interactions and noisy coordinates. The common re-evaluation uses 120 Gini trees, depth 18, minimum leaf 2, square-root feature sampling and balanced bootstrap class weights. It reaches 0.8413 mean CV Macro F1, 0.8544 student-test Macro F1, 82.80% conditional Top-1 and 0.2119 log loss. Its Top-1 is close to XGBoost, while the probability loss and inference time are less favorable in this run. Depth and leaf limits constrain complexity; the archived common run does not contain same-split training scores, so a precise overfitting diagnosis is not established.")
    heading(story, "3.2.4", "Linear SVM", level=2)
    para(story, "A linear SVM tests whether the four geometric inputs are nearly linearly separable. The common run uses scaled features, C=1, balanced classes, and the default squared-hinge objective. Its 0.7239 mean CV Macro F1 and 0.7297 student-test Macro F1 are well below the nonlinear models; conditional Top-1 is 76.47%. CV and student-test F1 are similar, which argues against a large split-specific collapse. The low absolute F1 relative to nonlinear models suggests the linear boundary is too simple for this representation, but this is an interpretation rather than a train-loss diagnosis. Margin outputs support ranking but do not define a log loss without calibration.")
    heading(story, "3.2.5", "K nearest neighbors", level=2)
    para(story, "KNN compares a candidate with nearby examples in scaled feature space. The common run uses 13 neighbors, Manhattan distance and inverse-distance weighting, matching the original notebook's chosen neighbor rule while using all shared training rows. It attains 0.8400 mean CV Macro F1, 0.8587 student-test Macro F1, 79.72% conditional Top-1 and 0.2888 log loss. It needs little fit computation but stores the training set and spends 1.47 s on the student-test model call here. CV and student-test scores do not show a generalization collapse, yet coarse neighborhood probabilities and the higher log loss limit use as HMM emissions.")
    para(story, "<b>Other recorded configurations.</b> The original KNN notebook used its own generated candidates and a balanced 32,000-row sample; the original Random Forest notebook used 15 features and its saved execution stopped before a complete CV; the SVM notebook also tried an RBF variant on a 15,000-row fold sample. A separate 16-feature MLP investigation used a 75 m, eight-candidate pipeline and therefore belongs to a different data contract. These experiments are archived as context and are not mixed into the common table.")
    heading(story, "3.3", "Model visualisations", level=2)
    figure(story, FIG / "cv_macro_f1.png",
           "Figure 4. Candidate Macro F1 on five frozen trip folds. Bars show mean and error bars show sample standard deviation. The SVM result reproduces its original notebook score under the common data contract.", 225)
    figure(story, FIG / "student_test_top1_f1.png",
           "Figure 5. Student-test conditional observation Top-1 and candidate-row Macro F1. The metrics have different units and are shown together only to expose the model trade-off.", 225)
    figure(story, FIG / "loss_continuity.png",
           "Figure 6. Student-test probability loss and classifier-only direct route continuity. SVM is omitted from log loss because its saved form produces margins rather than probabilities.", 225)

    # 4. Tuning
    heading(story, "4.", "Bayesian hyperparameter optimisation")
    para(story, "Optuna's seeded TPE sampler searched XGBoost configurations using five-fold mean observation Top-1 on the 80 training trips as its primary objective. The SQLite study records 25 completed and six failed/interrupted trials. The highest raw primary score occurred at trial 30; the archived selection rule allowed solutions within 0.0005 of that peak and used Macro F1 and log loss as secondary criteria, selecting trial 17 for deployment. Student-test trips were evaluated after this selection.")
    tab(story, [
        ["Parameter", "Search range", "Selected value"],
        ["Trees / depth", "200-800 by 50 / 3-8", "550 / 5"],
        ["Learning rate", "0.02-0.15, log scale", "0.02817"],
        ["Minimum child weight", "1-10", "6"],
        ["Row / column sampling", "0.65-1.0 / 0.60-1.0", "0.7428 / 0.9823"],
        ["Gamma", "0-2", "1.1601"],
        ["L1 / L2 regularization", "1e-8 to 10 / 0.1 to 20, log scales", "7.45e-8 / 0.3624"],
        ["Histogram bins / positive weight", "128, 256, 512 / 1, 2, 3, 5", "128 / 1"],
    ], [145, 190, WIDTH - 335])
    comparison = pd.read_csv(ROOT / "outputs" / "xgboost" / "hyperparameter_tuning" /
                             "baseline_vs_tuned.csv").set_index("Metric")
    tune_rows = [["Measure", "Baseline", "Tuned", "Tuned - baseline"]]
    for label, metric in [
        ("CV Macro F1", "CV Macro F1"),
        ("Student-test Top-1", "Student-Test Top-1 Accuracy (Conditional)"),
        ("Student-test Macro F1", "Student-Test Macro F1"),
        ("Student-test log loss", "Student-Test Log Loss"),
    ]:
        row = comparison.loc[metric]
        tune_rows.append([label, f"{row['Baseline']:.6f}", f"{row['Tuned']:.6f}",
                          f"{row['Absolute_Diff']:+.6f}"])
    tune_rows.append(["Student-test inference", "118 ms", "116 ms", "-2 ms (archived single-run timing)"])
    tab(story, tune_rows, [145, 94, 94, WIDTH - 333])
    para(story, "Tuning adds about 0.00010 CV Macro F1, but the held-out student-test Top-1 and Macro F1 both fall by about 0.0003 and 0.0002; log loss rises by 0.00007. These are very small changes, so the study does not show a held-out accuracy improvement. The archived inference calls differ by about 2 ms and are not repeated hardware-controlled benchmarks. The baseline weights are not present in the submitted checkout; the baseline column is traceable to the archived comparison CSV and training notebook, while the tuned model JSON is shipped and loaded by the API.")
    figure(story, ROOT / "outputs" / "xgboost" / "hyperparameter_tuning" / "visualisations" /
           "optimization_history.png",
           "Figure 7. Archived Optuna optimization history over completed trials; trial 17 is the selected deployment configuration, while trial 30 had the highest raw primary score.", 225)

    # 5. HMM
    heading(story, "5.", "HMM and Viterbi route reconstruction")
    para(story, "At every GPS observation, candidate directed arcs are hidden states. The XGBoost positive-class score provides the emission likelihood; the decoder adds its clipped log probability to a dynamic-programming path score. A transition is possible only if the same arc can be continued or a directed road path joins the previous exit node to the next entry node within the travel bound. Its log weight penalizes disagreement between road distance and observed GPS displacement and is normalized across feasible successor states. The decoder keeps up to eight candidate states per observation.")
    para(story, "Nonpositive or greater-than-30-second time gaps make a transition invalid. A GPS row without road candidates flushes the current Viterbi chain. If no directed connection exists, the decoder starts a new chunk rather than drawing a fabricated bridge. The route geometry includes accepted connector arcs and is returned as separate polylines. Reference `.route` is read only after scoring and decoding for offline arc-overlap evaluation.")
    figure(story, FIG / "hmm_ablation.png",
           "Figure 8. Ten evaluator trips: unique predicted arc set F1 against each supplied reference route, before and after deployed HMM/Viterbi decoding. The reference sequence has no pointwise observation labels.", 225)
    tab(story, [
        ["Evaluator measure", "Independent classifier", "XGBoost + HMM"],
        ["Mean unique route arc F1 (10 trips)", f"{ablation.classifier_arc_f1.mean():.3f}", f"{ablation.hmm_arc_f1.mean():.3f}"],
        ["Mean feasible adjacent transitions", pct(ablation.classifier_transition_feasibility.mean()),
         pct(ablation.hmm_transition_feasibility.mean())],
        ["Trips with higher arc F1", "-", f"{int((ablation.hmm_minus_classifier_arc_f1 > 0).sum())} / {len(ablation)}"],
    ], [235, 125, WIDTH - 360], keep=True)
    para(story, "The mean route arc F1 gain is 0.122 across these evaluator trips. Trip 00000023 has the smallest gain (about 0.026), so the decoder sometimes adds little to an already coherent independent path. No trip-level arc F1 decrease appears in this ten-trip set. The decoder still leaves gaps on trips 00000015 and 00000073; it does not force a continuous path when evidence or graph connectivity is insufficient. These arc-set results do not prove a pointwise Top-1 improvement on evaluator data.")

    # 6. Map
    heading(story, "6.", "Geographical visualisation and live route map")
    para(story, "The interactive Folium file for evaluator trip 00000030 shows original GPS rows 220-319, nearby directed road arcs, the supplied reference route and the deployed XGBoost plus HMM result as toggleable layers. It is available at analysis/results/folium_trip_00000030.html and is generated by analysis/folium_map.py. The reference is loaded only after prediction. OpenStreetMap background tiles require internet; the trip geometry comes from the evaluator ZIP.")
    para(story, "Figure 9 is the team's actual frontend laptop capture from the two-laptop demo. Its Leaflet route explorer displays server-returned GPS observations and the predicted route for trip 00000030. The live UI keeps ground truth on the server laptop; the Folium HTML is a separate interactive file.")
    figure(story, EVIDENCE / "frontend_route_explorer.png",
           "Figure 9. Team-supplied live Route Explorer screenshot: blue GPS observations and cyan XGBoost plus HMM route on the frontend laptop for evaluator trip 00000030. The offline Folium file is a separate repository artifact.", 310)

    # 7. Demo
    heading(story, "7.", "API UI and two-laptop deployment")
    para(story, "The server laptop runs FastAPI, loads the tuned XGBoost JSON once, reads evaluator trips, scores candidates and decodes with HMM. The frontend laptop runs Next.js, requests `/health` and `/api/v1/predict`, and displays returned metrics and map geometry. Both laptops join the same Wi-Fi or hotspot. The frontend's server connection field uses the server laptop's current LAN address, not its own localhost.")
    tab(story, [
        ["Laptop", "Role", "Start command from repository"],
        ["Server laptop", "Model, route decoder, API and logs", "powershell -ExecutionPolicy Bypass -File backend\\start_backend.ps1"],
        ["Frontend laptop", "Evaluator UI and interactive route", "cd frontend; npm.cmd ci; npm.cmd run dev"],
    ], [90, 175, WIDTH - 265])
    para(story, "For the captured demo, the frontend connected to `http://192.168.1.9:8000`. The server log shows a `POST /api/v1/predict` returning HTTP 200 for trips 00000015 and 00000030. Trip 00000015 was partial (6,854 observations, 55,290 candidates, 29 unresolved steps); trip 00000030 succeeded (654 observations, 15,805 candidates, zero unresolved steps). Private LAN addresses are examples from that session and must be replaced when the network changes.")
    figure(story, EVIDENCE / "frontend_overview.png",
           "Figure 10. Live frontend laptop: online server status, loaded XGBoost weights, active HMM decoder, 10 evaluator trips and trip 00000015 result.", 250)
    figure(story, EVIDENCE / "server_api_log.png",
           "Figure 11. Server laptop terminal during the two-laptop demo. The POST requests and per-trip start/finish log lines demonstrate backend execution and HTTP 200 responses.", 250)

    # 8. Limitations
    story.append(PageBreak())
    heading(story, "8.", "Interpretation and limitations")
    para(story, "<b>Reference alignment.</b> The supplied `.route` sequence is an ordered set of road arcs, not a timestamped road label per GPS sample. Candidate labels in the training table are weak monotone alignments. The report measures agreement with those labels; it does not claim independently verified physical road truth.")
    para(story, "<b>Candidate coverage.</b> The common candidate tables omit observations where the reference arc was absent, even when other candidates existed. Candidate-row metrics and conditional Top-1 therefore describe only reference-hit groups. End-to-end Top-1 explicitly counts aligned reference misses. The 50 m radius also limits what any classifier can recover.")
    para(story, "<b>Deployment candidate shift.</b> The API uses the same four feature names and 50 m road search, but its spatial index can return more than the training pipeline's ten candidates per observation. The student-test classifier scores therefore do not exactly characterize the live candidate distribution; evaluator arc results are reported separately.")
    para(story, "<b>Protocol and timing.</b> The original model notebooks used different feature sets, sampling rates and/or candidate generation. The common re-evaluation fixes those elements for the five-model table, but inference times still come from different local run environments. The baseline XGBoost model weights were not supplied; the tuning comparison is based on saved metrics. Evaluator route arc F1 and student-test candidate F1 are distinct metrics and are never combined.")
    para(story, "<b>AI assistance.</b> AI tools assisted with backend integration, report drafting, audit scripts, visualisation and reproducibility checks. The reported measurements come from executed code and saved outputs. Team members should review and explain the data assumptions, model choices and metrics, and disclose assistance in the form required by the event policy.")

    # 9. Reproduction and source index
    heading(story, "9.", "Reproducibility and evidence index")
    para(story, "The public repository contains the final API/UI implementation, five original model notebooks, the tuned XGBoost artifact, the shared candidate preprocessing, and the report-specific re-evaluation scripts. The following paths identify the main evidence used in this report. Raw evaluator and student trip ZIPs are supplied externally to the scripts and are not training artifacts.")
    tab(story, [
        ["Purpose", "Repository path"],
        ["Shared data, folds and preprocessing", "Hyperparameter_shared_pipeline/"],
        ["Tuned XGBoost parameters, CV and test predictions", "outputs/xgboost/hyperparameter_tuning/"],
        ["Four-feature MLP trainer and saved run", "analysis/mlp_shared/"],
        ["Common SVM, RF and KNN runs", "analysis/report_benchmark.py; analysis/results/*.json"],
        ["HMM ablation and Folium generator", "analysis/hmm_ablation.py; analysis/folium_map.py"],
        ["EDA audit and demo screenshots", "analysis/eda/; analysis/evidence/"],
        ["Production backend and evaluator frontend", "backend/app/; frontend/"],
    ], [170, WIDTH - 170])
    para(story, "To repeat the report-specific checks, run the commands in `analysis/README.md`, then build this PDF with `py -3.11 analysis/build_team9_report.py`. The server and frontend setup commands are in the repository README.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ReportDoc(OUT).build(story)
    print(OUT)


if __name__ == "__main__":
    build()
