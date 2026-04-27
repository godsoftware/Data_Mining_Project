Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ReportDir = Join-Path $Root "report"
$TablesDir = Join-Path $Root "outputs\tables"
$FiguresDir = Join-Path $Root "outputs\figures"
$ShapDir = Join-Path $Root "outputs\shap_results"

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

function CsvRows($name, $limit = 0) {
    $path = Join-Path $TablesDir $name
    $rows = Import-Csv $path
    if ($limit -gt 0) { return @($rows | Select-Object -First $limit) }
    return @($rows)
}

function JsonObject($name) {
    $path = Join-Path $TablesDir $name
    return (Get-Content $path -Raw | ConvertFrom-Json)
}

function Add-WordText($selection, [string]$text, [int]$size = 11, [bool]$bold = $false) {
    $selection.Font.Size = $size
    $selection.Font.Bold = if ($bold) { 1 } else { 0 }
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function Add-WordHeading($selection, [string]$text, [int]$level = 1) {
    $headingStyle = -1 - $level
    $selection.Style = $headingStyle
    $selection.TypeText($text)
    $selection.TypeParagraph()
    $selection.Style = -1
}

function Add-WordCsvTable($doc, $selection, [string]$title, [string]$csvName, [int]$limit = 8) {
    Add-WordHeading $selection $title 3
    $rows = CsvRows $csvName $limit
    if ($rows.Count -eq 0) { return }
    $headers = @($rows[0].PSObject.Properties.Name)
    $table = $doc.Tables.Add($selection.Range, $rows.Count + 1, $headers.Count)
    $table.Borders.Enable = 1
    for ($c = 0; $c -lt $headers.Count; $c++) {
        $table.Cell(1, $c + 1).Range.Text = $headers[$c]
        $table.Cell(1, $c + 1).Range.Bold = 1
    }
    for ($r = 0; $r -lt $rows.Count; $r++) {
        for ($c = 0; $c -lt $headers.Count; $c++) {
            $value = [string]$rows[$r].$($headers[$c])
            if ($value.Length -gt 80) { $value = $value.Substring(0, 77) + "..." }
            $table.Cell($r + 2, $c + 1).Range.Text = $value
        }
    }
    $selection.EndKey(6) | Out-Null
    $selection.TypeParagraph()
}

function Add-WordImage($selection, [string]$path, [string]$caption) {
    if (-not (Test-Path $path)) { return }
    Add-WordText $selection $caption 10 $true
    $shape = $selection.InlineShapes.AddPicture($path)
    if ($shape.Width -gt 430) { $shape.Width = 430 }
    $selection.TypeParagraph()
}

$metadata = JsonObject "final_model_metadata.json"
$finalMetrics = CsvRows "final_metrics_with_gmean_ks.csv"
$defaultMetrics = @($finalMetrics | Where-Object { $_.operating_point -eq "default_050" })[0]
$costMetrics = @($finalMetrics | Where-Object { $_.operating_point -eq "cost_sensitive_015" })[0]
$shapTop = CsvRows "shap_top_features.csv" 5
$stability = CsvRows "shap_stability_spearman.csv"
$meanRho = [math]::Round((($stability | ForEach-Object { [double]$_.spearman_rho } | Measure-Object -Average).Average), 3)

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Add()
    $selection = $word.Selection

    Add-WordText $selection "Explainable Credit Card Default Prediction Using Machine Learning, Imbalance Handling, Calibration, and SHAP Stability Analysis" 18 $true
    Add-WordText $selection "Final Report Draft" 14 $true
    Add-WordText $selection "Dataset: UCI Default of Credit Card Clients / Taiwan" 11 $false

    Add-WordHeading $selection "Abstract" 1
    Add-WordText $selection "This project develops an explainable credit-card default prediction framework using the UCI Taiwan dataset. The workflow combines data cleaning, payment-behavior feature engineering, imbalance-aware model comparison, compact hyperparameter tuning, calibration, cost-sensitive threshold optimization, SHAP, LIME, SHAP stability, and faithfulness testing. The final selected model is a tuned XGBoost classifier because it achieved the strongest cost-sensitive threshold result while preserving literature-consistent ROC-AUC and PR-AUC performance."

    Add-WordHeading $selection "1. Introduction and Problem Definition" 1
    Add-WordText $selection "The goal is to predict whether a credit-card client will default next month and to explain the model's decisions in a financially meaningful way. The project does not optimize accuracy alone; it also evaluates probability calibration, business threshold behavior, and explanation reliability."

    Add-WordHeading $selection "2. Dataset Description" 1
    Add-WordText $selection "The dataset contains 30,000 Taiwanese credit-card accounts, 23 predictors, and a binary target. The target distribution is imbalanced: 23,364 non-default and 6,636 default observations. EDUCATION values 0, 5, and 6 were merged into Others, and MARRIAGE value 0 was merged into Others. Negative BILL_AMT values were preserved because they can represent overpayment or credit balance."
    Add-WordCsvTable $doc $selection "Dataset Variable Description" "dataset_variable_description.csv" 12

    Add-WordHeading $selection "3. Literature and Research Gap" 1
    Add-WordText $selection "Prior studies usually focus on predictive performance, imbalance handling, or explainability separately. This project combines these components with calibration, threshold optimization, SHAP/LIME explanations, stability testing, and faithfulness analysis in a single framework."
    Add-WordCsvTable $doc $selection "Reviewed Papers Summary" "reviewed_papers_summary.csv" 8

    Add-WordHeading $selection "4. Methodology" 1
    Add-WordText $selection "The methodology follows a leakage-safe pipeline: stratified train-test split, preprocessing fitted on training data only, categorical encoding, scaling, feature engineering, model comparison, threshold analysis, calibration, SHAP/LIME explanation, stability analysis, and faithfulness tests."
    Add-WordCsvTable $doc $selection "Feature Engineering List" "feature_engineering_list.csv" 20
    Add-WordCsvTable $doc $selection "Hyperparameter Search Space" "hyperparameter_search_space.csv" 10

    Add-WordHeading $selection "5. Exploratory Data Analysis" 1
    Add-WordImage $selection (Join-Path $FiguresDir "target_class_distribution.png") "Figure 1. Target class distribution"
    Add-WordImage $selection (Join-Path $FiguresDir "pay0_default_rate.png") "Figure 2. Default rate by PAY_0"
    Add-WordImage $selection (Join-Path $FiguresDir "default_rate_by_delay_count.png") "Figure 3. Default rate by delayed payment count"
    Add-WordImage $selection (Join-Path $FiguresDir "correlation_heatmap.png") "Figure 4. Correlation heatmap"

    Add-WordHeading $selection "6. Model Training and Selection" 1
    Add-WordText $selection "The models tested include Logistic Regression, balanced Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, XGBoost, LightGBM, and SMOTENC variants. A compact RandomizedSearchCV was then applied to XGBoost. The tuned XGBoost model was selected as final because it reduced the FN-weighted expected cost."
    Add-WordCsvTable $doc $selection "Enhanced Model Comparison" "enhanced_model_comparison.csv" 10
    Add-WordCsvTable $doc $selection "Hyperparameter Tuning Results" "hyperparameter_tuning_results.csv" 8

    Add-WordHeading $selection "7. Final Evaluation" 1
    Add-WordText $selection ("At threshold 0.50, the tuned final model achieved ROC-AUC {0}, PR-AUC {1}, precision {2}, recall {3}, F1 {4}, and Brier score {5}." -f $defaultMetrics.roc_auc, $defaultMetrics.pr_auc, $defaultMetrics.precision, $defaultMetrics.recall, $defaultMetrics.f1, $defaultMetrics.brier_score)
    Add-WordText $selection ("At the cost-sensitive threshold 0.15, recall increased to {0}, G-mean increased to {1}, and expected cost decreased to {2}." -f $costMetrics.recall, $costMetrics.g_mean, $costMetrics.expected_cost_fn5_fp1)
    Add-WordCsvTable $doc $selection "Final Metrics with G-Mean and KS" "final_metrics_with_gmean_ks.csv" 4
    Add-WordCsvTable $doc $selection "Threshold Comparison" "threshold_comparison.csv" 8
    Add-WordCsvTable $doc $selection "Calibration Results" "calibration_results.csv" 5
    Add-WordImage $selection (Join-Path $FiguresDir "roc_curves.png") "Figure 5. ROC curve"
    Add-WordImage $selection (Join-Path $FiguresDir "precision_recall_curves.png") "Figure 6. Precision-recall curve"
    Add-WordImage $selection (Join-Path $FiguresDir "calibration_curve.png") "Figure 7. Calibration curve"
    Add-WordImage $selection (Join-Path $FiguresDir "confusion_matrix_threshold_05.png") "Figure 8. Confusion matrix at threshold 0.50"

    Add-WordHeading $selection "8. Explainable AI Analysis" 1
    Add-WordText $selection ("The top SHAP features are {0}. This confirms that recent payment delay behavior is the dominant default-risk signal." -f (($shapTop | ForEach-Object { $_.feature }) -join ", "))
    Add-WordCsvTable $doc $selection "SHAP Top Features" "shap_top_features.csv" 10
    Add-WordImage $selection (Join-Path $ShapDir "shap_bar.png") "Figure 9. SHAP feature importance bar plot"
    Add-WordImage $selection (Join-Path $ShapDir "shap_beeswarm.png") "Figure 10. SHAP beeswarm plot"
    Add-WordCsvTable $doc $selection "Local Explanation Cases" "local_explanation_cases.csv" 8
    Add-WordImage $selection (Join-Path $ShapDir "shap_waterfall_correct_default_high_risk.png") "Figure 11. Local SHAP waterfall for a correctly predicted default customer"
    Add-WordImage $selection (Join-Path $FiguresDir "lime_correct_default_high_risk.png") "Figure 12. LIME explanation for a correctly predicted default customer"

    Add-WordHeading $selection "9. Stability and Faithfulness" 1
    Add-WordText $selection ("The SHAP stability analysis retrained the final XGBoost configuration across 10 random seeds. The mean pairwise Spearman rank correlation was approximately {0}, indicating stable explanations for the strongest features." -f $meanRho)
    Add-WordCsvTable $doc $selection "SHAP Top-5 Frequency" "shap_top5_frequency.csv" 8
    Add-WordCsvTable $doc $selection "Faithfulness Test Results" "faithfulness_test_results.csv" 10
    Add-WordImage $selection (Join-Path $FiguresDir "shap_stability_top5_frequency.png") "Figure 13. SHAP top-5 stability frequency"
    Add-WordImage $selection (Join-Path $FiguresDir "probability_shift_after_perturbation.png") "Figure 14. Probability shift after feature perturbation"

    Add-WordHeading $selection "10. Remaining Limitations" 1
    Add-WordText $selection "First, the project uses one public dataset from Taiwan, so external validity to other countries or time periods is limited. Second, hyperparameter tuning was compact rather than exhaustive. Third, SHAP stability used 10 repeated seeds, whereas recent literature uses up to 100 models. Fourth, sensitive fairness evaluation by demographic group was not deeply investigated. Fifth, the model should not be used as an automated credit decision system without governance, bias review, and external validation."

    Add-WordHeading $selection "11. Conclusion" 1
    Add-WordText $selection "The project satisfies the proposed aim: it predicts credit-card default, explains model behavior with SHAP and LIME, evaluates threshold and calibration behavior, and tests explanation stability and faithfulness. The tuned XGBoost model is selected as the final model because it provides strong discrimination, improved FN-weighted cost, and stable explanations centered on recent repayment behavior."

    Add-WordHeading $selection "References" 1
    Add-WordText $selection "Yeh, I-C. and Lien, C-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications."
    Add-WordText $selection "Bhandary, R. and Ghosh, B. K. (2025). Credit Card Default Prediction: An Empirical Analysis on Predictive Performance Using Statistical and Machine Learning Methods. Journal of Risk and Financial Management."
    Add-WordText $selection "Lin, L. and Wang, Y. (2025). SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default Model. Risks."
    Add-WordText $selection "Jin, L., Wu, Z., and Zhao, J. (2022). Prediction of Credit Card Defaulters Based on SMOTE-XGBoost Model. WHICEB Proceedings."

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

function Add-PptSlide($presentation, [string]$title, [string[]]$bullets, [string]$imagePath = "") {
    $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
    $titleBox = $slide.Shapes.AddTextbox(1, 30, 20, 660, 45)
    $titleBox.TextFrame.TextRange.Text = $title
    $titleBox.TextFrame.TextRange.Font.Size = 28
    $titleBox.TextFrame.TextRange.Font.Bold = 1

    $bodyBox = $slide.Shapes.AddTextbox(1, 40, 85, 390, 360)
    $bodyBox.TextFrame.TextRange.Font.Size = 16
    $bodyBox.TextFrame.TextRange.Text = (($bullets | ForEach-Object { "- " + $_ }) -join "`r`n")

    if ($imagePath -ne "" -and (Test-Path $imagePath)) {
        $shape = $slide.Shapes.AddPicture($imagePath, $false, $true, 445, 95, 280, 260)
    }
}

$ppt = $null
$presentation = $null
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $presentation = $ppt.Presentations.Add()

    Add-PptSlide $presentation "Explainable Credit Card Default Prediction" @(
        "UCI Default of Credit Card Clients / Taiwan",
        "Tuned XGBoost + threshold optimization + calibration",
        "SHAP, LIME, stability, and faithfulness analysis"
    ) (Join-Path $FiguresDir "target_class_distribution.png")

    Add-PptSlide $presentation "Dataset and Cleaning" @(
        "30,000 rows, 23 predictors, binary default target",
        "Default class count: 6,636",
        "EDUCATION and MARRIAGE anomalies merged into Others",
        "Negative bill amounts preserved as valid credit-balance cases"
    ) (Join-Path $FiguresDir "pay0_default_rate.png")

    Add-PptSlide $presentation "Feature Engineering" @(
        "Delay features: delay_count, max_delay, recent_delay",
        "Bill/payment aggregates: totals, averages, trends",
        "Risk ratios: payment_to_bill_ratio and utilization_proxy",
        "No target-derived leakage features"
    ) (Join-Path $FiguresDir "default_rate_by_delay_count.png")

    Add-PptSlide $presentation "Model Selection" @(
        "Compared LR, balanced LR, RF, HistGB, XGBoost, LightGBM",
        "SMOTENC variants tested inside training pipeline",
        "Compact RandomizedSearchCV applied to XGBoost",
        "Final model: tuned XGBoost, no resampling"
    ) ""

    Add-PptSlide $presentation "Final Performance" @(
        ("ROC-AUC: {0}" -f $defaultMetrics.roc_auc),
        ("PR-AUC: {0}" -f $defaultMetrics.pr_auc),
        ("Threshold 0.50 recall: {0}" -f $defaultMetrics.recall),
        ("Cost-sensitive threshold 0.15 recall: {0}" -f $costMetrics.recall)
    ) (Join-Path $FiguresDir "roc_curves.png")

    Add-PptSlide $presentation "Threshold and Calibration" @(
        "0.50 threshold is not business-optimal",
        "FN cost = 5 and FP cost = 1 used for cost analysis",
        ("Best cost-sensitive expected cost: {0}" -f $costMetrics.expected_cost_fn5_fp1),
        "Calibration checked with Brier score and calibration curve"
    ) (Join-Path $FiguresDir "calibration_curve.png")

    Add-PptSlide $presentation "Global Explainability" @(
        "PAY_0 is the strongest SHAP feature",
        "Recent delay and max delay are also dominant",
        "LIMIT_BAL and utilization_proxy capture credit exposure",
        "Findings align with payment-behavior intuition"
    ) (Join-Path $ShapDir "shap_bar.png")

    Add-PptSlide $presentation "Local Explainability" @(
        "Local SHAP waterfalls generated for representative customers",
        "LIME examples generated for high-risk, low-risk, false-negative, and borderline cases",
        "Local explanations support case-level interpretation"
    ) (Join-Path $FiguresDir "lime_correct_default_high_risk.png")

    Add-PptSlide $presentation "Reliability Tests" @(
        ("Mean SHAP ranking Spearman rho: {0}" -f $meanRho),
        "Top drivers remained stable across repeated seeds",
        "Feature perturbation caused the largest probability shifts for top SHAP drivers",
        "Supports explanation reliability"
    ) (Join-Path $FiguresDir "shap_stability_top5_frequency.png")

    Add-PptSlide $presentation "Limitations and Conclusion" @(
        "Single public dataset; external validation is still needed",
        "Compact tuning, not exhaustive optimization",
        "10-seed stability, not 100-seed industrial validation",
        "Project meets prediction, XAI, calibration, threshold, stability, and faithfulness goals"
    ) (Join-Path $FiguresDir "probability_shift_after_perturbation.png")

    $pptxPath = Join-Path $ReportDir "presentation_slides.pptx"
    $presentation.SaveAs($pptxPath)
}
finally {
    if ($presentation -ne $null) { $presentation.Close(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null }
    if ($ppt -ne $null) { $ppt.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output "Generated final_report.docx, final_report.pdf, and presentation_slides.pptx"
