Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ReportDir = Join-Path $Root "report"
$FiguresDir = Join-Path $Root "outputs\figures"
$ShapDir = Join-Path $Root "outputs\shap_results"
$TablesDir = Join-Path $Root "outputs\tables"

$metrics = Import-Csv (Join-Path $TablesDir "final_metrics_with_gmean_ks.csv")
$defaultMetrics = @($metrics | Where-Object { $_.operating_point -eq "default_050" })[0]
$costMetrics = @($metrics | Where-Object { $_.operating_point -eq "cost_sensitive_015" })[0]
$shapTop = Import-Csv (Join-Path $TablesDir "shap_top_features.csv") | Select-Object -First 5
$rhoRows = Import-Csv (Join-Path $TablesDir "shap_stability_spearman.csv")
$meanRho = [math]::Round((($rhoRows | ForEach-Object { [double]$_.spearman_rho } | Measure-Object -Average).Average), 3)

function AddDocLine($selection, [string]$text, [int]$size = 11, [bool]$bold = $false) {
    $selection.Font.Size = $size
    $selection.Font.Bold = if ($bold) { 1 } else { 0 }
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function AddDocHeading($selection, [string]$text) {
    AddDocLine $selection $text 15 $true
}

function AddDocImage($selection, [string]$path, [string]$caption) {
    if (-not (Test-Path $path)) { return }
    AddDocLine $selection $caption 10 $true
    $shape = $selection.InlineShapes.AddPicture($path)
    if ($shape.Width -gt 420) { $shape.Width = 420 }
    $selection.TypeParagraph()
}

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Add()
    $selection = $word.Selection

    AddDocLine $selection "Explainable Credit Card Default Prediction" 20 $true
    AddDocLine $selection "Final Report Summary" 14 $true
    AddDocLine $selection "This report summarizes the completed machine learning and explainable AI project using the UCI Default of Credit Card Clients / Taiwan dataset."

    AddDocHeading $selection "1. Objective"
    AddDocLine $selection "The objective is to predict next-month credit-card default and explain model decisions using SHAP and LIME. The project also evaluates calibration, threshold cost, SHAP stability, and faithfulness."

    AddDocHeading $selection "2. Dataset and Cleaning"
    AddDocLine $selection "The dataset contains 30,000 customers, 23 predictors, and a binary default target. Missing values and duplicates were checked. EDUCATION values 0, 5, and 6 were merged into Others; MARRIAGE value 0 was merged into Others. Negative BILL_AMT values were preserved."
    AddDocImage $selection (Join-Path $FiguresDir "target_class_distribution.png") "Figure 1. Target distribution"
    AddDocImage $selection (Join-Path $FiguresDir "default_rate_by_delay_count.png") "Figure 2. Default rate by delayed payment count"

    AddDocHeading $selection "3. Modelling"
    AddDocLine $selection "The project compared Logistic Regression, balanced Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM, and SMOTENC variants. A compact RandomizedSearchCV was added for XGBoost. The final model is tuned XGBoost with no resampling."

    AddDocHeading $selection "4. Final Results"
    AddDocLine $selection ("Threshold 0.50: Accuracy={0}, Precision={1}, Recall={2}, F1={3}, ROC-AUC={4}, PR-AUC={5}, Brier={6}, G-mean={7}, KS={8}." -f $defaultMetrics.accuracy, $defaultMetrics.precision, $defaultMetrics.recall, $defaultMetrics.f1, $defaultMetrics.roc_auc, $defaultMetrics.pr_auc, $defaultMetrics.brier_score, $defaultMetrics.g_mean, $defaultMetrics.ks_statistic)
    AddDocLine $selection ("Cost-sensitive threshold 0.15: Precision={0}, Recall={1}, F1={2}, G-mean={3}, Expected cost={4}." -f $costMetrics.precision, $costMetrics.recall, $costMetrics.f1, $costMetrics.g_mean, $costMetrics.expected_cost_fn5_fp1)
    AddDocImage $selection (Join-Path $FiguresDir "roc_curves.png") "Figure 3. ROC curve"
    AddDocImage $selection (Join-Path $FiguresDir "precision_recall_curves.png") "Figure 4. Precision-recall curve"
    AddDocImage $selection (Join-Path $FiguresDir "calibration_curve.png") "Figure 5. Calibration curve"

    AddDocHeading $selection "5. Explainability"
    AddDocLine $selection ("Top SHAP features: {0}." -f (($shapTop | ForEach-Object { $_.feature }) -join ", "))
    AddDocLine $selection "The strongest signals are recent repayment status and delay behavior, especially PAY_0, max_delay, recent_delay, and delay_count."
    AddDocImage $selection (Join-Path $ShapDir "shap_bar.png") "Figure 6. SHAP feature importance"
    AddDocImage $selection (Join-Path $ShapDir "shap_waterfall_correct_default_high_risk.png") "Figure 7. Local SHAP explanation"
    AddDocImage $selection (Join-Path $FiguresDir "lime_correct_default_high_risk.png") "Figure 8. Local LIME explanation"

    AddDocHeading $selection "6. Reliability"
    AddDocLine $selection ("SHAP stability was tested across 10 random seeds. Mean pairwise Spearman rho was approximately {0}. Faithfulness was tested by perturbing top SHAP-ranked variables and measuring AUC/probability shifts." -f $meanRho)
    AddDocImage $selection (Join-Path $FiguresDir "shap_stability_top5_frequency.png") "Figure 9. SHAP top-5 stability frequency"
    AddDocImage $selection (Join-Path $FiguresDir "probability_shift_after_perturbation.png") "Figure 10. Faithfulness probability shift"

    AddDocHeading $selection "7. Remaining Limitations"
    AddDocLine $selection "The project uses one public dataset, so external validation is needed. Hyperparameter tuning is compact rather than exhaustive. SHAP stability uses 10 seeds rather than 100. Fairness analysis is not deeply developed. The result is an academic prototype, not a production credit-decision system."

    AddDocHeading $selection "8. Conclusion"
    AddDocLine $selection "The project satisfies the roadmap: prediction, imbalance-aware evaluation, threshold optimization, calibration, SHAP/LIME explanation, SHAP stability, and faithfulness testing. The final tuned XGBoost model gives strong literature-consistent performance and stable, financially meaningful explanations."

    $docxPath = Join-Path $ReportDir "final_report.docx"
    $pdfPath = Join-Path $ReportDir "final_report.pdf"
    $doc.SaveAs2($docxPath)
    $doc.SaveAs2($pdfPath, 17)
}
finally {
    if ($doc -ne $null) { $doc.Close($false) | Out-Null; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null }
    if ($word -ne $null) { $word.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

function AddSlide($presentation, [string]$title, [string[]]$bullets, [string]$image = "") {
    $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
    $titleBox = $slide.Shapes.AddTextbox(1, 30, 20, 660, 45)
    $titleBox.TextFrame.TextRange.Text = $title
    $titleBox.TextFrame.TextRange.Font.Size = 28
    $titleBox.TextFrame.TextRange.Font.Bold = 1
    $bodyBox = $slide.Shapes.AddTextbox(1, 40, 90, 380, 340)
    $bodyBox.TextFrame.TextRange.Text = (($bullets | ForEach-Object { "- " + $_ }) -join "`r`n")
    $bodyBox.TextFrame.TextRange.Font.Size = 16
    if ($image -ne "" -and (Test-Path $image)) {
        $slide.Shapes.AddPicture($image, $false, $true, 445, 95, 280, 250) | Out-Null
    }
}

$ppt = $null
$presentation = $null
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $presentation = $ppt.Presentations.Add()
    AddSlide $presentation "Explainable Credit Card Default Prediction" @("UCI Taiwan dataset", "Tuned XGBoost final model", "SHAP/LIME + stability + faithfulness") (Join-Path $FiguresDir "target_class_distribution.png")
    AddSlide $presentation "Dataset and Cleaning" @("30,000 rows, 23 predictors", "6,636 default cases", "EDUCATION/MARRIAGE anomalies merged", "No missing values") (Join-Path $FiguresDir "pay0_default_rate.png")
    AddSlide $presentation "Feature Engineering" @("Delay count, max delay, recent delay", "Bill and payment aggregates", "Payment-to-bill ratio", "Utilization proxy") (Join-Path $FiguresDir "default_rate_by_delay_count.png")
    AddSlide $presentation "Model Selection" @("Compared LR, RF, HistGB, XGBoost, LightGBM", "SMOTENC tested", "RandomizedSearchCV tuned XGBoost", "Final: tuned XGBoost") ""
    AddSlide $presentation "Final Performance" @(("ROC-AUC: {0}" -f $defaultMetrics.roc_auc), ("PR-AUC: {0}" -f $defaultMetrics.pr_auc), ("Recall at 0.50: {0}" -f $defaultMetrics.recall), ("Recall at 0.15: {0}" -f $costMetrics.recall)) (Join-Path $FiguresDir "roc_curves.png")
    AddSlide $presentation "Threshold and Calibration" @("Default 0.50 threshold is not cost-optimal", "FN cost = 5, FP cost = 1", ("Best expected cost: {0}" -f $costMetrics.expected_cost_fn5_fp1), "Calibration checked with Brier score") (Join-Path $FiguresDir "calibration_curve.png")
    AddSlide $presentation "SHAP Results" @("PAY_0 is the top feature", "Delay behavior dominates risk", "LIMIT_BAL and utilization_proxy matter", "Findings are financially meaningful") (Join-Path $ShapDir "shap_bar.png")
    AddSlide $presentation "Local Explanations" @("SHAP waterfalls generated", "LIME cases generated", "High-risk, low-risk, false-negative, and borderline examples", "Supports case-level interpretation") (Join-Path $FiguresDir "lime_correct_default_high_risk.png")
    AddSlide $presentation "Reliability" @(("Mean SHAP Spearman rho: {0}" -f $meanRho), "Top features stable across seeds", "Faithfulness tested by perturbation", "PAY_0 perturbation causes large AUC drop") (Join-Path $FiguresDir "shap_stability_top5_frequency.png")
    AddSlide $presentation "Limitations and Conclusion" @("Single dataset; external validation needed", "Compact tuning, not exhaustive", "10-seed stability, not 100", "Roadmap goals satisfied") (Join-Path $FiguresDir "probability_shift_after_perturbation.png")
    $presentation.SaveAs((Join-Path $ReportDir "presentation_slides.pptx"))
}
finally {
    if ($presentation -ne $null) { $presentation.Close(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null }
    if ($ppt -ne $null) { $ppt.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output "Generated final_report.docx, final_report.pdf, and presentation_slides.pptx"

