param(
    [string]$Configuration = "Debug"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$exePath = Join-Path $repoRoot "apps\desktop\CorpusStudio.Desktop\bin\$Configuration\net8.0-windows\CorpusStudio.Desktop.exe"
if (-not (Test-Path $exePath)) {
    throw "Desktop executable not found at $exePath. Run dotnet build apps\desktop\CorpusStudio.Desktop.sln first."
}

function Wait-Until([scriptblock]$Block, [int]$TimeoutSeconds = 20, [string]$Message = "Timed out waiting for condition") {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $result = & $Block
        if ($null -ne $result -and $result -ne $false) {
            return $result
        }

        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)

    throw $Message
}

function Condition-ByAutomationId([string]$AutomationId) {
    New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty,
        $AutomationId
    )
}

function Condition-ByName([string]$Name) {
    New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty,
        $Name
    )
}

function Condition-ByProcessId([int]$ProcessId) {
    New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty,
        $ProcessId
    )
}

function Condition-ByControlType($ControlType) {
    New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        $ControlType
    )
}

function Wait-Window([int]$ProcessId) {
    Wait-Until -TimeoutSeconds 20 -Message "Window for process $ProcessId was not found" -Block {
        [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
            [System.Windows.Automation.TreeScope]::Children,
            (Condition-ByProcessId $ProcessId)
        )
    }
}

function Wait-WindowByName([int]$ProcessId, [string]$Name) {
    Wait-Until -TimeoutSeconds 20 -Message "Window named $Name for process $ProcessId was not found" -Block {
        $windows = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Children,
            (Condition-ByProcessId $ProcessId)
        )

        foreach ($window in $windows) {
            if ($window.Current.Name -eq $Name `
                -and $window.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window) {
                return $window
            }
        }

        return $null
    }
}

function Find-Descendant($Root, [string]$AutomationId, [int]$TimeoutSeconds = 10) {
    Wait-Until -TimeoutSeconds $TimeoutSeconds -Message "Element $AutomationId was not found" -Block {
        $Root.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            (Condition-ByAutomationId $AutomationId)
        )
    }
}

function Find-ByName($Root, [string]$Name, [int]$TimeoutSeconds = 10) {
    Wait-Until -TimeoutSeconds $TimeoutSeconds -Message "Element named $Name was not found" -Block {
        $Root.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            (Condition-ByName $Name)
        )
    }
}

function Invoke-Element($Element) {
    $Element.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
}

function Set-ElementValue($Element, [string]$Value) {
    $Element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern).SetValue($Value)
}

function Get-ElementValue($Element) {
    $Element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern).Current.Value
}

function Get-ElementText($Element) {
    try {
        return Get-ElementValue $Element
    }
    catch {
        return $Element.Current.Name
    }
}

function Get-ElementSelectionName($Element) {
    try {
        $selectionPattern = $Element.GetCurrentPattern([System.Windows.Automation.SelectionPattern]::Pattern)
        $selection = $selectionPattern.Current.GetSelection()
        if ($selection.Count -gt 0) {
            return $selection[0].Current.Name
        }
    }
    catch {
        return $Element.Current.Name
    }

    return $Element.Current.Name
}

function Select-Tab($Root, [string]$Name) {
    $tab = Find-ByName $Root $Name 10
    $tab.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
}

function Select-ListItemContaining($Root, [string]$AutomationId, [string]$Text) {
    $list = Find-Descendant $Root $AutomationId 10
    $item = Wait-Until -TimeoutSeconds 10 -Message "List item containing $Text was not found" -Block {
        $candidates = $list.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants,
            [System.Windows.Automation.Condition]::TrueCondition
        )

        foreach ($candidate in $candidates) {
            if (-not $candidate.Current.Name.Contains($Text)) {
                continue
            }

            $selectable = $candidate
            while ($null -ne $selectable) {
                try {
                    $selectable.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern) | Out-Null
                    return $selectable
                }
                catch {
                    $selectable = [System.Windows.Automation.TreeWalker]::ControlViewWalker.GetParent($selectable)
                }
            }
        }

        return $null
    }

    $item.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
    return $item
}

function Select-ComboBoxItem($Root, [string]$AutomationId, [string]$Text) {
    $comboBox = Find-Descendant $Root $AutomationId 10
    $expandPattern = $comboBox.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern)
    $expandPattern.Expand()
    try {
        $item = Wait-Until -TimeoutSeconds 10 -Message "ComboBox item $Text was not found" -Block {
            foreach ($searchRoot in @($comboBox, $Root)) {
                $candidates = $searchRoot.FindAll(
                    [System.Windows.Automation.TreeScope]::Descendants,
                    [System.Windows.Automation.Condition]::TrueCondition
                )

                foreach ($candidate in $candidates) {
                    if ($candidate.Current.Name -ne $Text) {
                        continue
                    }

                    try {
                        $candidate.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern) | Out-Null
                        return $candidate
                    }
                    catch {
                        continue
                    }
                }
            }

            return $null
        }

        $item.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
        return $item
    }
    finally {
        try {
            $expandPattern.Collapse()
        }
        catch {
        }
    }
}

function Select-ComboBoxItemContaining($Root, [string]$AutomationId, [string]$Text) {
    $comboBox = Find-Descendant $Root $AutomationId 10
    $expandPattern = $comboBox.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern)
    $expandPattern.Expand()
    try {
        $item = Wait-Until -TimeoutSeconds 10 -Message "ComboBox item containing $Text was not found" -Block {
            foreach ($searchRoot in @($comboBox, $Root)) {
                $candidates = $searchRoot.FindAll(
                    [System.Windows.Automation.TreeScope]::Descendants,
                    [System.Windows.Automation.Condition]::TrueCondition
                )

                foreach ($candidate in $candidates) {
                    if (-not $candidate.Current.Name.Contains($Text)) {
                        continue
                    }

                    try {
                        $candidate.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern) | Out-Null
                        return $candidate
                    }
                    catch {
                        continue
                    }
                }
            }

            return $null
        }

        $item.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
        return $item
    }
    finally {
        try {
            $expandPattern.Collapse()
        }
        catch {
        }
    }
}

function Wait-And-ClickButton([int]$ProcessId, [string]$Name) {
    $ok = Wait-Until -TimeoutSeconds 10 -Message "OK button was not found" -Block {
        $windows = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Children,
            (Condition-ByProcessId $ProcessId)
        )

        foreach ($window in $windows) {
            $button = $window.FindFirst(
                [System.Windows.Automation.TreeScope]::Descendants,
                (Condition-ByName $Name)
            )
            if ($null -ne $button) {
                return $button
            }
        }

        return $null
    }

    Invoke-Element $ok
}

function Wait-And-ClickOk([int]$ProcessId) {
    Wait-And-ClickButton $ProcessId "OK"
}

function Start-CorpusStudio([string]$DataDir, [string]$ExportDir) {
    $env:CORPUS_STUDIO_DATA_DIR = $DataDir
    $env:CORPUS_STUDIO_EXPORT_DIR = $ExportDir
    Start-Process -FilePath $exePath -WorkingDirectory (Split-Path $exePath) -PassThru
}

function Stop-CorpusStudio($Process) {
    if ($Process -and -not $Process.HasExited) {
        $Process.CloseMainWindow() | Out-Null
        if (-not $Process.WaitForExit(5000)) {
            $Process.Kill()
        }
    }
}

$oldData = $env:CORPUS_STUDIO_DATA_DIR
$oldExport = $env:CORPUS_STUDIO_EXPORT_DIR
$stamp = Get-Date -Format "yyyyMMddHHmmss"
$projectName = "Smoke Examples $stamp"
$projectId = "smoke_examples_$stamp"
$dataDir = Join-Path $env:TEMP "CorpusStudioUiData_$stamp"
$exportDir = Join-Path $env:TEMP "CorpusStudioUiExport_$stamp"
$importPath = Join-Path $env:TEMP "CorpusStudioImport_$stamp.jsonl"
New-Item -ItemType Directory -Force -Path $dataDir, $exportDir | Out-Null

$process = $null
try {
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id

    Invoke-Element (Find-Descendant $main "NewDatasetProjectButton")
    Find-Descendant $main "ProjectNameTextBox" 10 | Out-Null
    Set-ElementValue (Find-Descendant $main "ProjectNameTextBox") $projectName
    Set-ElementValue (Find-Descendant $main "ProjectIdTextBox") $projectId
    Invoke-Element (Find-Descendant $main "CreateProjectButton")
    Wait-And-ClickOk $process.Id

    $invalidDraft = @{
        instruction = "Explain invalid tags."
        output = "Tags should be a list."
        tags = "bad"
    } | ConvertTo-Json -Depth 5
    Set-ElementValue (Find-Descendant $main "DraftTextBox") $invalidDraft
    Invoke-Element (Find-Descendant $main "ValidateButton")
    $validationSummary = Find-Descendant $main "ValidationSummaryTextBox"
    Wait-Until -TimeoutSeconds 10 -Message "Validation summary did not report expected list issue" -Block {
        $value = Get-ElementValue $validationSummary
        if ($value -like "*Expected list.*" -and $value -like "*[tags]*") {
            return $value
        }

        return $null
    } | Out-Null
    Select-ListItemContaining $main "ValidationIssuesListBox" "Expected list." | Out-Null
    Wait-Until -TimeoutSeconds 10 -Message "Draft editor was not focused after selecting validation issue" -Block {
        $focused = [System.Windows.Automation.AutomationElement]::FocusedElement
        if ($null -ne $focused -and $focused.Current.AutomationId -eq "DraftTextBox") {
            return $true
        }

        return $null
    } | Out-Null

    $rows = for ($index = 0; $index -lt 20; $index++) {
        [ordered]@{
            instruction = "Explain item $index."
            input = ""
            output = "Item $index explanation."
        }
    }
    $draftJson = $rows | ConvertTo-Json -Depth 5
    Set-ElementValue (Find-Descendant $main "DraftTextBox") $draftJson

    $projectPath = Join-Path $dataDir $projectId
    $importRows = for ($index = 20; $index -lt 22; $index++) {
        [ordered]@{
            instruction = "Explain imported item $index."
            input = ""
            output = "Imported item $index explanation."
        }
    }
    $importRows = @($importRows) + [ordered]@{
        instruction = "Rejected import row."
        input = ""
    }
    $importLines = $importRows | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 5 }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($importPath, $importLines, $utf8NoBom)

    Invoke-Element (Find-Descendant $main "SaveExampleButton")
    Wait-And-ClickOk $process.Id
    Invoke-Element (Find-Descendant $main "ImportDatasetButton")
    Start-Sleep -Milliseconds 500
    [System.Windows.Forms.SendKeys]::SendWait($importPath)
    [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
    $validationSummary = Find-Descendant $main "ValidationSummaryTextBox"
    $importPreview = Wait-Until -TimeoutSeconds 15 -Message "Import preview did not report accepted rows" -Block {
        $value = Get-ElementValue $validationSummary
        if ($value -like "*Accepted rows: 2*" `
            -and $value -like "*Rejected rows: 1*" `
            -and $value -like "*Missing required field: output*") {
            return $value
        }

        return $null
    }
    Wait-And-ClickButton $process.Id "Yes"
    Wait-And-ClickOk $process.Id

    $quarantineDir = Join-Path $projectPath "import_quarantine"
    $quarantineFile = Wait-Until -TimeoutSeconds 10 -Message "Import quarantine file was not written" -Block {
        if (-not (Test-Path $quarantineDir)) {
            return $null
        }

        $files = @(Get-ChildItem -Path $quarantineDir -Filter "*_rejected.jsonl")
        if ($files.Count -eq 1) {
            return $files[0].FullName
        }

        return $null
    }
    $quarantineText = Get-Content -Raw -Path $quarantineFile
    if ($quarantineText -notlike "*Rejected import row.*" `
        -or $quarantineText -notlike "*Missing required field: output*") {
        throw "Import quarantine file did not include the rejected row and error at $quarantineFile"
    }
    Select-Tab $main "Quarantine"
    $quarantineDetail = Find-Descendant $main "ImportQuarantineDetailTextBox"
    $quarantineReviewText = Wait-Until -TimeoutSeconds 10 -Message "Quarantine review did not show rejected row details" -Block {
        $value = Get-ElementValue $quarantineDetail
        if ($value -like "*Rejected import row.*" `
            -and $value -like "*Missing required field: output*") {
            return $value
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "RetryQuarantineItemButton")
    Wait-Until -TimeoutSeconds 10 -Message "Retrying quarantine row did not load it into the draft editor" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*Rejected import row.*") {
            return $value
        }

        return $null
    } | Out-Null

    Invoke-Element (Find-Descendant $main "RunQualityButton")
    $qualitySummary = Find-Descendant $main "QualitySummaryTextBox"
    $firstQuality = Wait-Until -TimeoutSeconds 10 -Message "Quality summary did not report imported examples" -Block {
        $value = Get-ElementValue $qualitySummary
        if ($value -like "*Examples: 22*" `
            -and $value -like "*Exact duplicates: 0*" `
            -and $value -like "*Normalized duplicates: 0*" `
            -and $value -like "*Low-information rows: 0*" `
            -and $value -like "*Synthetic pattern warnings: 0*") {
            return $value
        }

        return $null
    }
    $qualityHistory = Find-Descendant $main "QualityHistoryTextBox"
    $firstQualityHistory = Wait-Until -TimeoutSeconds 10 -Message "Quality history did not record quality run" -Block {
        $value = Get-ElementValue $qualityHistory
        if ($value -like "*Recent quality history:*" `
            -and $value -like "*Examples: 22*" `
            -and $value -like "*Issues:*") {
            return $value
        }

        return $null
    }

    Select-Tab $main "Examples"
    $detail = Find-Descendant $main "SelectedExampleTextBox"
    $firstJson = Wait-Until -TimeoutSeconds 10 -Message "Saved example JSON did not populate" -Block {
        $value = Get-ElementValue $detail
        if ($value -like "*`"instruction`"*" -and $value -like "*Explain item 0*") {
            return $value
        }

        return $null
    }

    $examplesPath = Join-Path $projectPath "examples.jsonl"
    if (-not (Test-Path $examplesPath)) {
        throw "examples.jsonl was not written at $examplesPath"
    }
    $exampleCount = (Get-Content -Path $examplesPath).Count
    if ($exampleCount -ne 22) {
        throw "Expected 22 examples, found $exampleCount in $examplesPath"
    }
    $qualityHistoryPath = Join-Path $projectPath "quality_history.jsonl"
    if (-not (Test-Path $qualityHistoryPath)) {
        throw "quality_history.jsonl was not written at $qualityHistoryPath"
    }
    if (@(Get-Content -Path $qualityHistoryPath).Count -lt 3) {
        throw "Expected at least 3 quality history entries in $qualityHistoryPath"
    }

    Stop-CorpusStudio $process
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id
    $qualitySummary = Find-Descendant $main "QualitySummaryTextBox"
    $reloadedQuality = Wait-Until -TimeoutSeconds 10 -Message "Reloaded quality summary did not report imported examples" -Block {
        $value = Get-ElementValue $qualitySummary
        if ($value -like "*Examples: 22*" `
            -and $value -like "*Exact duplicates: 0*" `
            -and $value -like "*Normalized duplicates: 0*" `
            -and $value -like "*Low-information rows: 0*" `
            -and $value -like "*Synthetic pattern warnings: 0*") {
            return $value
        }

        return $null
    }

    Select-Tab $main "Examples"
    $detail = Find-Descendant $main "SelectedExampleTextBox"
    $reloadedJson = Wait-Until -TimeoutSeconds 10 -Message "Reloaded example JSON did not populate" -Block {
        $value = Get-ElementValue $detail
        if ($value -like "*`"instruction`"*" -and $value -like "*Explain item 0*") {
            return $value
        }

        return $null
    }

    Select-Tab $main "Splits"
    Set-ElementValue (Find-Descendant $main "SplitTrainPercentTextBox") "80"
    Set-ElementValue (Find-Descendant $main "SplitValidationPercentTextBox") "10"
    Set-ElementValue (Find-Descendant $main "SplitSeedTextBox") "123"
    Invoke-Element (Find-Descendant $main "GenerateSplitsButton")
    $splitSummary = Find-Descendant $main "SplitSummaryTextBox"
    $splitText = Wait-Until -TimeoutSeconds 10 -Message "Split summary did not report expected counts" -Block {
        $value = Get-ElementValue $splitSummary
        if ($value -like "*Train: 17*" `
            -and $value -like "*Validation: 2*" `
            -and $value -like "*Test: 3*" `
            -and $value -like "*Ratios: train 80%, validation 10%, test 10%*" `
            -and $value -like "*Seed: 123*") {
            return $value
        }

        return $null
    }
    $splitDir = Join-Path (Join-Path $exportDir $projectId) "splits"
    $expectedSplitCounts = @{
        "train.jsonl" = 17
        "validation.jsonl" = 2
        "test.jsonl" = 3
    }
    foreach ($splitFile in $expectedSplitCounts.Keys) {
        $path = Join-Path $splitDir $splitFile
        if (-not (Test-Path $path)) {
            throw "$splitFile was not written at $path"
        }

        $splitLineCount = @(Get-Content -Path $path).Count
        if ($splitLineCount -ne $expectedSplitCounts[$splitFile]) {
            throw "Expected $($expectedSplitCounts[$splitFile]) rows in $path, found $splitLineCount"
        }
    }

    Set-ElementValue (Find-Descendant $main "SplitTrainPercentTextBox") "96"
    Set-ElementValue (Find-Descendant $main "SplitValidationPercentTextBox") "1"
    Invoke-Element (Find-Descendant $main "GenerateSplitsButton")
    $splitWarningText = Wait-Until -TimeoutSeconds 10 -Message "Split summary did not report tiny split warnings" -Block {
        $value = Get-ElementValue $splitSummary
        if ($value -like "*Train: 21*" `
            -and $value -like "*Validation: 0*" `
            -and $value -like "*Test: 1*" `
            -and $value -like "*Warnings:*" `
            -and $value -like "*Validation split has no rows*" `
            -and $value -like "*Test split has only 1 row*") {
            return $value
        }

        return $null
    }

    $projectJson = Get-Content -Raw -Path (Join-Path $projectPath "project.json") | ConvertFrom-Json
    if ($projectJson.split_settings.train_ratio -ne 0.96 `
        -or $projectJson.split_settings.validation_ratio -ne 0.01 `
        -or $projectJson.split_settings.seed -ne 123) {
        throw "Split settings were not persisted to project.json"
    }

    $evaluationFailedSavedRow = (Get-Content -Path $examplesPath)[1]
    $evaluationFailedSavedRowObject = $evaluationFailedSavedRow | ConvertFrom-Json
    $evaluationFailedSavedRowInstruction = [string]$evaluationFailedSavedRowObject.instruction
    $evaluationFailedSavedRowOutput = [string]$evaluationFailedSavedRowObject.output

    $evaluationDir = Join-Path (Join-Path $exportDir $projectId) "evaluation"
    New-Item -ItemType Directory -Force -Path $evaluationDir | Out-Null
    $evaluationReportPath = Join-Path $evaluationDir "20260630120000_evaluation_report.json"
    $evaluationComparisonReportPath = Join-Path $evaluationDir "20260630120500_evaluation_report.json"
    $evaluationReportPayload = [ordered]@{
        dataset = "smoke_eval_dataset"
        model = "fake-eval-model"
        examples_tested = 2
        average_score = 57.5
        failed_examples = 1
        weak_tags = @("loops")
        tag_summary = @(
            [ordered]@{
                tag = "recursion"
                examples = 1
                failed_examples = 1
                average_score = 40.0
            },
            [ordered]@{
                tag = "loops"
                examples = 1
                failed_examples = 0
                average_score = 75.0
            }
        )
        failure_reason_summary = @(
            [ordered]@{
                reason = "Weak explanation."
                failed_examples = 1
            }
        )
        score_band_summary = @(
            [ordered]@{
                band = "0-49"
                examples = 1
                failed_examples = 1
                average_score = 40.0
            },
            [ordered]@{
                band = "70-84"
                examples = 1
                failed_examples = 0
                average_score = 75.0
            }
        )
        manually_scored_examples = 0
        average_manual_score = $null
        run_settings = [ordered]@{
            dataset_path = (Join-Path $projectPath "examples.jsonl")
            schema_id = "instruction"
            backend = "unsupported-regression"
            base_url = "http://localhost:11434"
            model = "fake-eval-model"
            limit = 2
            score_threshold = 75.0
            timeout_seconds = 44
        }
        results = @(
            [ordered]@{
                example_id = "row-1"
                prompt = "Explain loops."
                expected_output = "A loop repeats a block."
                model_output = "A loop repeats work."
                score = 75.0
                passed = $true
                tags = @("loops")
                notes = $null
                manual_score = $null
                manual_notes = $null
            },
            [ordered]@{
                example_id = "row-2"
                prompt = "Explain recursion."
                expected_output = "Recursion calls the same routine with a smaller problem."
                model_output = "Recursion is when code repeats somehow."
                score = 40.0
                passed = $false
                tags = @("recursion")
                notes = "Weak explanation."
                manual_score = $null
                manual_notes = $null
            }
        )
    } | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText($evaluationReportPath, $evaluationReportPayload, $utf8NoBom)
    [System.IO.File]::SetLastWriteTime($evaluationReportPath, [datetime]"2026-06-30T12:00:00")

    $evaluationComparisonReportPayload = [ordered]@{
        dataset = "smoke_eval_dataset"
        model = "fake-eval-model-v2"
        examples_tested = 2
        average_score = 80.0
        failed_examples = 0
        weak_tags = @()
        tag_summary = @(
            [ordered]@{
                tag = "loops"
                examples = 1
                failed_examples = 0
                average_score = 75.0
            },
            [ordered]@{
                tag = "recursion"
                examples = 1
                failed_examples = 0
                average_score = 85.0
            }
        )
        failure_reason_summary = @()
        score_band_summary = @(
            [ordered]@{
                band = "70-84"
                examples = 1
                failed_examples = 0
                average_score = 75.0
            },
            [ordered]@{
                band = "85-100"
                examples = 1
                failed_examples = 0
                average_score = 85.0
            }
        )
        manually_scored_examples = 0
        average_manual_score = $null
        run_settings = [ordered]@{
            dataset_path = (Join-Path $projectPath "examples.jsonl")
            schema_id = "instruction"
            backend = "unsupported-regression"
            base_url = "http://localhost:11434"
            model = "fake-eval-model-v2"
            limit = 2
            score_threshold = 75.0
            timeout_seconds = 44
        }
        results = @(
            [ordered]@{
                example_id = "row-1"
                prompt = "Explain loops."
                expected_output = "A loop repeats a block."
                model_output = "A loop repeats a block of instructions."
                score = 75.0
                passed = $true
                tags = @("loops")
                notes = $null
                manual_score = $null
                manual_notes = $null
            },
            [ordered]@{
                example_id = "row-2"
                prompt = "Explain recursion."
                expected_output = "Recursion calls the same routine with a smaller problem."
                model_output = "Recursion is when a routine calls itself on a smaller problem."
                score = 85.0
                passed = $true
                tags = @("recursion")
                notes = $null
                manual_score = $null
                manual_notes = $null
            }
        )
    } | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText($evaluationComparisonReportPath, $evaluationComparisonReportPayload, $utf8NoBom)
    [System.IO.File]::SetLastWriteTime($evaluationComparisonReportPath, [datetime]"2026-06-30T12:05:00")

    $aiAssistQueuePath = Join-Path $projectPath "ai_assist_reviews.jsonl"
    $aiAssistQueueViewsPath = Join-Path $projectPath "ai_assist_queue_views.json"
    $aiAssistRewriteBatchesPath = Join-Path $projectPath "ai_assist_rewrite_batches.json"
    $aiAssistSuggestedJsonl = @{
        instruction = "Explain queued AI Assist."
        input = ""
        output = "Queued AI Assist suggestions stay review-only until a human uses them."
    } | ConvertTo-Json -Compress -Depth 5
    $aiAssistQueueItems = @(
        [ordered]@{
            review_id = "assist_accept"
            created_at = "2026-06-30T12:00:00Z"
            decided_at = $null
            review_state = "review_required"
            schema_id = "instruction"
            action = "draft-example"
            model = "fake-ai-assist-model"
            prompt_template_id = "ai_assist_review_v0.1"
            source_draft = "{`"instruction`":`"Draft source`",`"output`":`"Draft output`"}"
            model_output = "Suggested a clearer draft."
            suggested_jsonl = "$aiAssistSuggestedJsonl`n"
            warnings = @()
            validation_errors = @()
        },
        [ordered]@{
            review_id = "assist_reject"
            created_at = "2026-06-30T12:01:00Z"
            decided_at = $null
            review_state = "review_required"
            schema_id = "instruction"
            action = "review"
            model = "fake-ai-review-model"
            prompt_template_id = "ai_assist_review_v0.1"
            source_draft = "{`"instruction`":`"Weak source`",`"output`":`"Weak output`"}"
            model_output = "This review should be rejected."
            suggested_jsonl = ""
            warnings = @("review-only warning")
            validation_errors = @()
        },
        [ordered]@{
            review_id = "assist_bulk"
            created_at = "2026-06-30T12:02:00Z"
            decided_at = $null
            review_state = "review_required"
            schema_id = "instruction"
            action = "review"
            model = "fake-ai-bulk-model"
            prompt_template_id = "ai_assist_review_v0.1"
            source_draft = "{`"instruction`":`"Bulk source`",`"output`":`"Bulk output`"}"
            model_output = "This review should be bulk rejected."
            suggested_jsonl = ""
            warnings = @("bulk warning")
            validation_errors = @()
        }
    )
    $aiAssistQueueLines = $aiAssistQueueItems | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 6 }
    [System.IO.File]::WriteAllLines($aiAssistQueuePath, $aiAssistQueueLines, $utf8NoBom)

    Stop-CorpusStudio $process
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id
    Select-Tab $main "Splits"
    Wait-Until -TimeoutSeconds 10 -Message "Saved split settings did not reload into the split controls" -Block {
        $trainValue = Get-ElementValue (Find-Descendant $main "SplitTrainPercentTextBox")
        $validationValue = Get-ElementValue (Find-Descendant $main "SplitValidationPercentTextBox")
        $seedValue = Get-ElementValue (Find-Descendant $main "SplitSeedTextBox")
        if ($trainValue -eq "96" -and $validationValue -eq "1" -and $seedValue -eq "123") {
            return "$trainValue/$validationValue/$seedValue"
        }

        return $null
    } | Out-Null

    Select-Tab $main "Evaluation"
    $evaluationComparisonHistoryText = Wait-Until -TimeoutSeconds 10 -Message "Newest evaluation report did not load by default" -Block {
        $summaryValue = Get-ElementValue (Find-Descendant $main "EvaluationSummaryTextBox")
        $reportValue = Get-ElementValue (Find-Descendant $main "EvaluationReportTextBox")
        if ($summaryValue -like "*Dataset: smoke_eval_dataset*" `
            -and $summaryValue -like "*Model: fake-eval-model-v2*" `
            -and $summaryValue -like "*Tag summary:*loops: 1 ex, 0 failed, avg 75*" `
            -and $summaryValue -like "*recursion: 1 ex, 0 failed, avg 85*" `
            -and $summaryValue -like "*Failure reasons: none*" `
            -and $summaryValue -like "*Score bands:*70-84: 1 ex, 0 failed, avg 75*" `
            -and $summaryValue -like "*85-100: 1 ex, 0 failed, avg 85*" `
            -and $reportValue -like "*smoke_eval_dataset*" `
            -and $reportValue -like "*fake-eval-model-v2*" `
            -and $reportValue -like "*tag_summary*" `
            -and $reportValue -like "*score_band_summary*" `
            -and $reportValue -like "*row-2*") {
            return "$summaryValue`n$reportValue"
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "CompareEvaluationReportsButton")
    $evaluationComparisonText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation report comparison did not summarize saved report deltas" -Block {
        $value = Get-ElementValue (Find-Descendant $main "EvaluationComparisonTextBox")
        if ($value -like "*Selected report:*" `
            -and $value -like "*fake-eval-model-v2 | 2 ex | avg 80*" `
            -and $value -like "*Compared with:*" `
            -and $value -like "*fake-eval-model | 2 ex | avg 57.5*" `
            -and $value -like "*Average score: 80 (+22.5)*" `
            -and $value -like "*Failed examples: 0 (-1; improved)*" `
            -and $value -like "*now passing: 1*" `
            -and $value -like "*Weak tags cleared: loops*" `
            -and $value -like "*row-2: 40 -> 85 (+45)*") {
            return $value
        }

        return $null
    }
    Select-ListItemContaining $main "EvaluationReportHistoryListBox" "fake-eval-model | 2 ex" | Out-Null
    $evaluationReportHistoryText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation report history did not reload saved report" -Block {
        $summaryValue = Get-ElementValue (Find-Descendant $main "EvaluationSummaryTextBox")
        $reportValue = Get-ElementValue (Find-Descendant $main "EvaluationReportTextBox")
        if ($summaryValue -like "*Dataset: smoke_eval_dataset*" `
            -and $summaryValue -like "*Model: fake-eval-model*" `
            -and $summaryValue -like "*Tag summary:*recursion: 1 ex, 1 failed, avg 40*" `
            -and $summaryValue -like "*loops: 1 ex, 0 failed, avg 75*" `
            -and $summaryValue -like "*Failure reasons: Weak explanation.: 1*" `
            -and $summaryValue -like "*Score bands:*0-49: 1 ex, 1 failed, avg 40*" `
            -and $summaryValue -like "*70-84: 1 ex, 0 failed, avg 75*" `
            -and $reportValue -like "*smoke_eval_dataset*" `
            -and $reportValue -like "*fake-eval-model*" `
            -and $reportValue -like "*failure_reason_summary*" `
            -and $reportValue -like "*row-1*") {
            return "$summaryValue`n$reportValue"
        }

        return $null
    }
    Select-ComboBoxItem $main "EvaluationResultFilterComboBox" "Failed" | Out-Null
    $evaluationFailureQueueText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation failure queue did not show only failed examples" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "EvaluationResultsSummaryTextBlock")
        $detailValue = Get-ElementValue (Find-Descendant $main "EvaluationExampleDetailTextBox")
        if ($summaryValue -like "*Results: 1 failed, 1 passed, 0 manually reviewed. Filter: Failed, showing 1 of 2.*" `
            -and $detailValue -like "*Example: row-2*" `
            -and $detailValue -like "*Passed: no*") {
            return "$summaryValue`n$detailValue"
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "EditEvaluationFailureButton")
    $evaluationFailureEditDraftText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation failed-row edit did not load the saved row into Writing Studio" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value.Contains($evaluationFailedSavedRowInstruction) `
            -and $value.Contains($evaluationFailedSavedRowOutput)) {
            return $value
        }

        return $null
    }
    Select-Tab $main "Evaluation"
    Select-ComboBoxItem $main "EvaluationResultFilterComboBox" "Failed" | Out-Null
    Select-ListItemContaining $main "EvaluationResultsListBox" "row-2" | Out-Null
    Invoke-Element (Find-Descendant $main "PrepareEvaluationFailureButton")
    $evaluationFailurePreparedText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation failure triage did not prepare AI Assist controls" -Block {
        $actionValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistActionComboBox")
        $instructionValue = Get-ElementValue (Find-Descendant $main "AiAssistInstructionTextBox")
        if ($actionValue -eq "rewrite-output" `
            -and $instructionValue -like "*Triage failed evaluation example row-2*" `
            -and $instructionValue -like "*Auto score: 40*" `
            -and $instructionValue -like "*Human review remains required*") {
            return "$actionValue`n$instructionValue"
        }

        return $null
    }
    Select-Tab $main "Writing Studio"
    $evaluationFailureDraftText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation failure triage did not load failed example into the draft editor" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*Explain recursion.*" `
            -and $value -like "*Recursion calls the same routine*" `
            -and $value -like "*`"tags`"*") {
            return $value
        }

        return $null
    }
    Select-Tab $main "Evaluation"
    Select-ComboBoxItem $main "EvaluationResultFilterComboBox" "All" | Out-Null
    Wait-Until -TimeoutSeconds 10 -Message "Evaluation result queue did not reset to all examples" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "EvaluationResultsSummaryTextBlock")
        if ($summaryValue -like "*Filter: All, showing 2 of 2*") {
            return $summaryValue
        }

        return $null
    } | Out-Null
    Select-ListItemContaining $main "EvaluationResultsListBox" "row-1" | Out-Null
    Set-ElementValue (Find-Descendant $main "EvaluationManualScoreTextBox") "92"
    Set-ElementValue (Find-Descendant $main "EvaluationManualNotesTextBox") "Manual review accepted with minor wording gap."
    Invoke-Element (Find-Descendant $main "SaveEvaluationReviewButton")
    $evaluationManualReviewText = Wait-Until -TimeoutSeconds 10 -Message "Manual evaluation review was not saved into report JSON" -Block {
        $summaryValue = Get-ElementValue (Find-Descendant $main "EvaluationSummaryTextBox")
        $reportValue = Get-ElementValue (Find-Descendant $main "EvaluationReportTextBox")
        if ($summaryValue -like "*Manual scores: 1 example(s), average 92*" `
            -and $reportValue -like "*manual_score*" `
            -and $reportValue -like "*92*" `
            -and $reportValue -like "*Manual review accepted*") {
            return "$summaryValue`n$reportValue"
        }

        return $null
    }
    $savedEvaluationReport = Get-Content -Raw -Path $evaluationReportPath
    if ($savedEvaluationReport -notlike "*manual_score*" `
        -or $savedEvaluationReport -notlike "*92*" `
        -or $savedEvaluationReport -notlike "*Manual review accepted with minor wording gap*") {
        throw "Manual evaluation review was not persisted to $evaluationReportPath"
    }
    Wait-Until -TimeoutSeconds 10 -Message "Evaluation controls did not load default local backend settings" -Block {
        $backendValue = Get-ElementValue (Find-Descendant $main "EvaluationBackendTextBox")
        $modelValue = Get-ElementValue (Find-Descendant $main "EvaluationModelTextBox")
        $baseUrlValue = Get-ElementValue (Find-Descendant $main "EvaluationBaseUrlTextBox")
        if ($backendValue -eq "ollama" `
            -and $modelValue -eq "qwen2.5-coder:7b" `
            -and $baseUrlValue -eq "http://localhost:11434") {
            return "$backendValue/$modelValue/$baseUrlValue"
        }

        return $null
    } | Out-Null
    Select-ListItemContaining $main "EvaluationReportHistoryListBox" "fake-eval-model-v2" | Out-Null
    Invoke-Element (Find-Descendant $main "RerunEvaluationReportButton")
    $evaluationRegressionRerunText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation regression rerun did not use saved run settings" -Block {
        $summaryValue = Get-ElementValue (Find-Descendant $main "EvaluationSummaryTextBox")
        $backendValue = Get-ElementValue (Find-Descendant $main "EvaluationBackendTextBox")
        $modelValue = Get-ElementValue (Find-Descendant $main "EvaluationModelTextBox")
        $limitValue = Get-ElementValue (Find-Descendant $main "EvaluationLimitTextBox")
        $thresholdValue = Get-ElementValue (Find-Descendant $main "EvaluationScoreThresholdTextBox")
        $timeoutValue = Get-ElementValue (Find-Descendant $main "EvaluationTimeoutTextBox")
        if ($summaryValue -like "*Evaluation could not run.*" `
            -and $summaryValue -like "*Unsupported model backend*" `
            -and $backendValue -eq "unsupported-regression" `
            -and $modelValue -eq "fake-eval-model-v2" `
            -and $limitValue -eq "2" `
            -and $thresholdValue -eq "75" `
            -and $timeoutValue -eq "44") {
            return "$summaryValue`n$backendValue/$modelValue/$limitValue/$thresholdValue/$timeoutValue"
        }

        return $null
    }
    Set-ElementValue (Find-Descendant $main "EvaluationBackendTextBox") "unsupported"
    Invoke-Element (Find-Descendant $main "RefreshEvaluationModelsButton")
    $evaluationModelRefreshErrorText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation model refresh did not report unsupported backend" -Block {
        $value = Get-ElementText (Find-Descendant $main "EvaluationModelListSummaryTextBlock")
        if ($value -like "*Evaluation model refresh failed.*" `
            -and $value -like "*Unsupported model backend*") {
            return $value
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "CheckEvaluationBackendButton")
    $evaluationSummary = Find-Descendant $main "EvaluationSummaryTextBox"
    $evaluationHealthText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation backend health check did not report unsupported backend" -Block {
        $value = Get-ElementValue $evaluationSummary
        if ($value -like "*Evaluation could not run.*" `
            -and $value -like "*Unsupported model backend*") {
            return $value
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "RunEvaluationButton")
    $evaluationPreflightText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation run did not perform pre-run backend validation" -Block {
        $value = Get-ElementValue $evaluationSummary
        if ($value -like "*Evaluation could not run.*" `
            -and $value -like "*Unsupported model backend*") {
            return $value
        }

        return $null
    }
    Set-ElementValue (Find-Descendant $main "EvaluationBackendTextBox") "ollama"
    Set-ElementValue (Find-Descendant $main "EvaluationScoreThresholdTextBox") "101"
    Invoke-Element (Find-Descendant $main "RunEvaluationButton")
    $evaluationValidationText = Wait-Until -TimeoutSeconds 10 -Message "Evaluation tab did not report invalid score threshold before network use" -Block {
        $value = Get-ElementValue $evaluationSummary
        if ($value -like "*Evaluation could not run.*" `
            -and $value -like "*score threshold must be a number from 0 to 100*") {
            return $value
        }

        return $null
    }
    $evaluationReportText = Get-ElementValue (Find-Descendant $main "EvaluationReportTextBox")
    if ($evaluationReportText -notlike "*No evaluation report was produced.*") {
        throw "Evaluation invalid-input path unexpectedly produced a report"
    }
    Set-ElementValue (Find-Descendant $main "EvaluationScoreThresholdTextBox") "70"

    Select-Tab $main "AI Assist"
    Wait-Until -TimeoutSeconds 10 -Message "AI Assist controls did not load default local backend settings" -Block {
        $backendValue = Get-ElementValue (Find-Descendant $main "AiAssistBackendTextBox")
        $modelValue = Get-ElementValue (Find-Descendant $main "AiAssistModelTextBox")
        $baseUrlValue = Get-ElementValue (Find-Descendant $main "AiAssistBaseUrlTextBox")
        if ($backendValue -eq "ollama" `
            -and $modelValue -eq "qwen2.5-coder:7b" `
            -and $baseUrlValue -eq "http://localhost:11434") {
            return "$backendValue/$modelValue/$baseUrlValue"
        }

        return $null
    } | Out-Null

    Select-Tab $main "Evaluation"
    Set-ElementValue (Find-Descendant $main "EvaluationBackendTextBox") "openai-compatible"
    Set-ElementValue (Find-Descendant $main "EvaluationModelTextBox") "smoke-eval-model"
    Set-ElementValue (Find-Descendant $main "EvaluationBaseUrlTextBox") "http://localhost:1234/v1"
    Set-ElementValue (Find-Descendant $main "EvaluationTimeoutTextBox") "33"
    Select-Tab $main "AI Assist"
    Set-ElementValue (Find-Descendant $main "AiAssistBackendTextBox") "openai-compatible"
    Set-ElementValue (Find-Descendant $main "AiAssistModelTextBox") "smoke-assist-model"
    Set-ElementValue (Find-Descendant $main "AiAssistBaseUrlTextBox") "http://localhost:1234/v1"
    Set-ElementValue (Find-Descendant $main "AiAssistTimeoutTextBox") "44"
    Select-Tab $main "Settings"
    Invoke-Element (Find-Descendant $main "SaveLabSettingsButton")
    $labSettingsSavedText = Wait-Until -TimeoutSeconds 10 -Message "Lab settings save did not update settings summary" -Block {
        $value = Get-ElementText (Find-Descendant $main "LabSettingsSummaryTextBlock")
        if ($value -like "*Saved lab backend settings to project metadata:*") {
            return $value
        }

        return $null
    }
    $projectJson = Get-Content -Raw -Path (Join-Path $projectPath "project.json") | ConvertFrom-Json
    if ($projectJson.lab_settings.evaluation.backend -ne "openai-compatible" `
        -or $projectJson.lab_settings.evaluation.model -ne "smoke-eval-model" `
        -or $projectJson.lab_settings.evaluation.base_url -ne "http://localhost:1234/v1" `
        -or $projectJson.lab_settings.evaluation.timeout_seconds -ne 33 `
        -or $projectJson.lab_settings.ai_assist.backend -ne "openai-compatible" `
        -or $projectJson.lab_settings.ai_assist.model -ne "smoke-assist-model" `
        -or $projectJson.lab_settings.ai_assist.base_url -ne "http://localhost:1234/v1" `
        -or $projectJson.lab_settings.ai_assist.timeout_seconds -ne 44) {
        throw "Lab backend settings were not persisted to project.json"
    }

    Stop-CorpusStudio $process
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id
    Select-Tab $main "Evaluation"
    $labSettingsReloadedText = Wait-Until -TimeoutSeconds 10 -Message "Saved lab settings did not reload into Evaluation controls" -Block {
        $backendValue = Get-ElementValue (Find-Descendant $main "EvaluationBackendTextBox")
        $modelValue = Get-ElementValue (Find-Descendant $main "EvaluationModelTextBox")
        $baseUrlValue = Get-ElementValue (Find-Descendant $main "EvaluationBaseUrlTextBox")
        $timeoutValue = Get-ElementValue (Find-Descendant $main "EvaluationTimeoutTextBox")
        if ($backendValue -eq "openai-compatible" `
            -and $modelValue -eq "smoke-eval-model" `
            -and $baseUrlValue -eq "http://localhost:1234/v1" `
            -and $timeoutValue -eq "33") {
            return "$backendValue/$modelValue/$baseUrlValue/$timeoutValue"
        }

        return $null
    }
    Select-Tab $main "AI Assist"
    Wait-Until -TimeoutSeconds 10 -Message "Saved lab settings did not reload into AI Assist controls" -Block {
        $backendValue = Get-ElementValue (Find-Descendant $main "AiAssistBackendTextBox")
        $modelValue = Get-ElementValue (Find-Descendant $main "AiAssistModelTextBox")
        $baseUrlValue = Get-ElementValue (Find-Descendant $main "AiAssistBaseUrlTextBox")
        $timeoutValue = Get-ElementValue (Find-Descendant $main "AiAssistTimeoutTextBox")
        if ($backendValue -eq "openai-compatible" `
            -and $modelValue -eq "smoke-assist-model" `
            -and $baseUrlValue -eq "http://localhost:1234/v1" `
            -and $timeoutValue -eq "44") {
            return "$backendValue/$modelValue/$baseUrlValue/$timeoutValue"
        }

        return $null
    } | Out-Null
    Select-Tab $main "Evaluation"
    Set-ElementValue (Find-Descendant $main "EvaluationBackendTextBox") "ollama"
    Set-ElementValue (Find-Descendant $main "EvaluationModelTextBox") "qwen2.5-coder:7b"
    Set-ElementValue (Find-Descendant $main "EvaluationBaseUrlTextBox") "http://localhost:11434"
    Set-ElementValue (Find-Descendant $main "EvaluationTimeoutTextBox") "120"
    Select-Tab $main "AI Assist"
    Set-ElementValue (Find-Descendant $main "AiAssistBackendTextBox") "ollama"
    Set-ElementValue (Find-Descendant $main "AiAssistModelTextBox") "qwen2.5-coder:7b"
    Set-ElementValue (Find-Descendant $main "AiAssistBaseUrlTextBox") "http://localhost:11434"
    Set-ElementValue (Find-Descendant $main "AiAssistTimeoutTextBox") "120"

    Select-ListItemContaining $main "AiAssistReviewQueueListBox" "fake-ai-assist-model" | Out-Null
    $aiAssistQueueLoadedText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist queue item did not load into review pane" -Block {
        $summaryValue = Get-ElementValue (Find-Descendant $main "AiAssistSummaryTextBox")
        $reviewValue = Get-ElementValue (Find-Descendant $main "AiAssistReviewTextBox")
        $sourceValue = Get-ElementValue (Find-Descendant $main "AiAssistSourceDraftTextBox")
        $suggestionValue = Get-ElementValue (Find-Descendant $main "AiAssistSuggestedJsonlTextBox")
        $diffValue = Get-ElementValue (Find-Descendant $main "AiAssistDiffSummaryTextBox")
        if ($summaryValue -like "*Review state: review_required*" `
            -and $reviewValue -like "*Suggested a clearer draft*" `
            -and $reviewValue -like "*Queued AI Assist suggestions stay review-only*" `
            -and $sourceValue -like "*Draft source*" `
            -and $suggestionValue -like "*Queued AI Assist suggestions stay review-only*" `
            -and $diffValue -like "*Source lines:*") {
            return "$summaryValue`n$reviewValue`n$sourceValue`n$suggestionValue`n$diffValue"
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "UseAiAssistSuggestionButton")
    Wait-Until -TimeoutSeconds 10 -Message "Using AI Assist suggestion did not load it into the draft editor" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*Queued AI Assist suggestions stay review-only*") {
            return $value
        }

        return $null
    } | Out-Null
    $aiAssistQueueText = Get-Content -Raw -Path $aiAssistQueuePath
    if ($aiAssistQueueText -notlike "*assist_accept*" `
        -or $aiAssistQueueText -notlike "*accepted*") {
        throw "AI Assist accept state was not persisted to $aiAssistQueuePath"
    }
    Select-Tab $main "AI Assist"
    Select-ListItemContaining $main "AiAssistReviewQueueListBox" "fake-ai-review-model" | Out-Null
    Invoke-Element (Find-Descendant $main "RejectAiAssistReviewButton")
    $aiAssistRejectedText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist reject state did not update the queue summary" -Block {
        $summaryValue = Get-ElementValue (Find-Descendant $main "AiAssistSummaryTextBox")
        $queueValue = Get-ElementValue (Find-Descendant $main "AiAssistReviewTextBox")
        $sourceValue = Get-ElementValue (Find-Descendant $main "AiAssistSourceDraftTextBox")
        $suggestionValue = Get-ElementValue (Find-Descendant $main "AiAssistSuggestedJsonlTextBox")
        if ($summaryValue -like "*Review state: rejected*" `
            -and $queueValue -like "*This review should be rejected*" `
            -and $sourceValue -like "*Weak source*" `
            -and $suggestionValue -like "*No suggested JSONL*") {
            return "$summaryValue`n$queueValue`n$sourceValue`n$suggestionValue"
        }

        return $null
    }
    $aiAssistQueueText = Get-Content -Raw -Path $aiAssistQueuePath
    if ($aiAssistQueueText -notlike "*assist_reject*" `
        -or $aiAssistQueueText -notlike "*rejected*") {
        throw "AI Assist reject state was not persisted to $aiAssistQueuePath"
    }
    Set-ElementValue (Find-Descendant $main "AiAssistQueueSearchTextBox") "bulk"
    Select-ComboBoxItem $main "AiAssistQueueSortComboBox" "Model" | Out-Null
    Select-ComboBoxItem $main "AiAssistQueueFilterComboBox" "Pending" | Out-Null
    $aiAssistQueueSearchSortText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist queue search/sort did not narrow visible reviews" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistQueueSummaryTextBlock")
        $searchValue = Get-ElementValue (Find-Descendant $main "AiAssistQueueSearchTextBox")
        $sortValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistQueueSortComboBox")
        if ($summaryValue -like "*Filter: Pending, search: bulk, sort: Model, showing 1 of 3*" `
            -and $searchValue -eq "bulk" `
            -and $sortValue -eq "Model") {
            return "$summaryValue`n$searchValue`n$sortValue"
        }

        return $null
    }
    Set-ElementValue (Find-Descendant $main "AiAssistQueueViewNameTextBox") "Bulk Pending View"
    Invoke-Element (Find-Descendant $main "SaveAiAssistQueueViewButton")
    $aiAssistQueueViewSavedText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist queue view save did not update the queue summary" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistQueueSummaryTextBlock")
        if ($summaryValue -like "*AI Assist queue view saved: Bulk Pending View.*") {
            return $summaryValue
        }

        return $null
    }
    if (-not (Test-Path $aiAssistQueueViewsPath)) {
        throw "AI Assist queue view file was not created at $aiAssistQueueViewsPath"
    }
    $aiAssistQueueViewsText = Get-Content -Raw -Path $aiAssistQueueViewsPath
    if ($aiAssistQueueViewsText -notlike '*"name": "Bulk Pending View"*' `
        -or $aiAssistQueueViewsText -notlike '*"filter": "Pending"*' `
        -or $aiAssistQueueViewsText -notlike '*"search": "bulk"*' `
        -or $aiAssistQueueViewsText -notlike '*"sort": "Model"*') {
        throw "AI Assist queue view state was not persisted to $aiAssistQueueViewsPath"
    }
    Select-ComboBoxItem $main "AiAssistQueueFilterComboBox" "All" | Out-Null
    Select-ComboBoxItem $main "AiAssistQueueSortComboBox" "Newest" | Out-Null
    Set-ElementValue (Find-Descendant $main "AiAssistQueueSearchTextBox") ""
    Invoke-Element (Find-Descendant $main "LoadAiAssistQueueViewButton")
    $aiAssistQueueViewLoadedText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist queue view load did not restore queue controls" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistQueueSummaryTextBlock")
        $filterValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistQueueFilterComboBox")
        $searchValue = Get-ElementValue (Find-Descendant $main "AiAssistQueueSearchTextBox")
        $sortValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistQueueSortComboBox")
        if ($summaryValue -like "*AI Assist queue view loaded: Bulk Pending View.*" `
            -and $filterValue -eq "Pending" `
            -and $searchValue -eq "bulk" `
            -and $sortValue -eq "Model") {
            return "$summaryValue`n$filterValue`n$searchValue`n$sortValue"
        }

        return $null
    }
    Select-ListItemContaining $main "AiAssistReviewQueueListBox" "fake-ai-bulk-model" | Out-Null
    Invoke-Element (Find-Descendant $main "BulkRejectAiAssistReviewsButton")
    $aiAssistBulkRejectedText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist bulk reject did not update the queue summary" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistQueueSummaryTextBlock")
        $queueFilterValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistQueueFilterComboBox")
        $queueValue = Get-ElementValue (Find-Descendant $main "AiAssistReviewTextBox")
        if ($summaryValue -like "*AI Assist bulk triage marked 1 review(s) rejected.*" `
            -and $summaryValue -like "*Undo steps available: 1.*" `
            -and $queueFilterValue -eq "Pending" `
            -and $queueValue -like "*No AI Assist reviews match the current queue controls*") {
            return "$summaryValue`n$queueFilterValue`n$queueValue"
        }

        return $null
    }
    $aiAssistQueueText = Get-Content -Raw -Path $aiAssistQueuePath
    if ($aiAssistQueueText -notlike "*assist_bulk*" `
        -or $aiAssistQueueText -notlike "*fake-ai-bulk-model*" `
        -or $aiAssistQueueText -notlike "*rejected*") {
        throw "AI Assist bulk reject state was not persisted to $aiAssistQueuePath"
    }
    Select-ComboBoxItem $main "AiAssistQueueFilterComboBox" "All" | Out-Null
    Select-ComboBoxItem $main "AiAssistQueueSortComboBox" "Newest" | Out-Null
    Set-ElementValue (Find-Descendant $main "AiAssistQueueSearchTextBox") ""
    Invoke-Element (Find-Descendant $main "BulkAcceptAiAssistReviewsButton")
    $aiAssistBulkAcceptedText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist bulk accept did not add a second undo step" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistQueueSummaryTextBlock")
        if ($summaryValue -like "*AI Assist bulk triage marked 3 review(s) accepted.*" `
            -and $summaryValue -like "*Undo steps available: 2.*") {
            return $summaryValue
        }

        return $null
    }
    $aiAssistQueueText = Get-Content -Raw -Path $aiAssistQueuePath
    if ($aiAssistQueueText -notlike '*"review_id":"assist_bulk"*' `
        -or $aiAssistQueueText -notlike '*"review_state":"accepted"*') {
        throw "AI Assist second bulk action state was not persisted to $aiAssistQueuePath"
    }
    Invoke-Element (Find-Descendant $main "UndoBulkAiAssistReviewsButton")
    $aiAssistBulkUndoOnceText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist first bulk undo did not restore the previous accepted/rejected mix" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistQueueSummaryTextBlock")
        if ($summaryValue -like "*AI Assist bulk triage undo restored 3 review(s).*" `
            -and $summaryValue -like "*Undo steps remaining: 1.*") {
            return $summaryValue
        }

        return $null
    }
    $aiAssistQueueText = Get-Content -Raw -Path $aiAssistQueuePath
    if ($aiAssistQueueText -notlike '*"review_id":"assist_bulk"*' `
        -or $aiAssistQueueText -notlike '*"model":"fake-ai-bulk-model"*' `
        -or $aiAssistQueueText -notlike '*"review_state":"rejected"*') {
        throw "AI Assist first bulk undo state was not persisted to $aiAssistQueuePath"
    }
    Invoke-Element (Find-Descendant $main "UndoBulkAiAssistReviewsButton")
    $aiAssistBulkUndoText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist second bulk undo did not restore the pending queue item" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistQueueSummaryTextBlock")
        if ($summaryValue -like "*AI Assist bulk triage undo restored 1 review(s).*" `
            -and $summaryValue -like "*Undo steps remaining: 0.*") {
            return $summaryValue
        }

        return $null
    }
    $aiAssistQueueText = Get-Content -Raw -Path $aiAssistQueuePath
    if ($aiAssistQueueText -notlike '*"review_id":"assist_bulk"*' `
        -or $aiAssistQueueText -notlike '*"model":"fake-ai-bulk-model"*' `
        -or $aiAssistQueueText -notlike '*"review_state":"review_required"*') {
        throw "AI Assist second bulk undo state was not persisted to $aiAssistQueuePath"
    }
    Select-ComboBoxItem $main "AiAssistQueueFilterComboBox" "Pending" | Out-Null
    Set-ElementValue (Find-Descendant $main "AiAssistQueueSearchTextBox") "bulk"
    Wait-Until -TimeoutSeconds 10 -Message "AI Assist second bulk undo did not restore the visible pending item" -Block {
        $queueValue = Get-ElementValue (Find-Descendant $main "AiAssistReviewTextBox")
        if ($queueValue -like "*This review should be bulk rejected*") {
            return $queueValue
        }

        return $null
    } | Out-Null
    Select-ComboBoxItem $main "AiAssistQueueFilterComboBox" "All" | Out-Null
    Set-ElementValue (Find-Descendant $main "AiAssistQueueSearchTextBox") ""
    Set-ElementValue (Find-Descendant $main "AiAssistBackendTextBox") "unsupported"
    Invoke-Element (Find-Descendant $main "RefreshAiAssistModelsButton")
    $aiAssistModelRefreshErrorText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist model refresh did not report unsupported backend" -Block {
        $value = Get-ElementText (Find-Descendant $main "AiAssistModelListSummaryTextBlock")
        if ($value -like "*AI Assist model refresh failed.*" `
            -and $value -like "*Unsupported model backend*") {
            return $value
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "CheckAiAssistBackendButton")
    $aiAssistSummary = Find-Descendant $main "AiAssistSummaryTextBox"
    $aiAssistHealthText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist backend health check did not report unsupported backend" -Block {
        $value = Get-ElementValue $aiAssistSummary
        if ($value -like "*AI Assist could not run.*" `
            -and $value -like "*Unsupported model backend*") {
            return $value
        }

        return $null
    }
    Set-ElementValue (Find-Descendant $main "AiAssistBackendTextBox") "ollama"
    Set-ElementValue (Find-Descendant $main "AiAssistTimeoutTextBox") "0"
    Invoke-Element (Find-Descendant $main "RunAiAssistButton")
    $aiAssistValidationText = Wait-Until -TimeoutSeconds 10 -Message "AI Assist tab did not report invalid timeout before network use" -Block {
        $value = Get-ElementValue $aiAssistSummary
        if ($value -like "*AI Assist could not run.*" `
            -and $value -like "*timeout must be a positive whole number*") {
            return $value
        }

        return $null
    }
    $aiAssistReviewText = Get-ElementValue (Find-Descendant $main "AiAssistReviewTextBox")
    if ($aiAssistReviewText -notlike "*No AI Assist suggestion was produced.*") {
        throw "AI Assist invalid-input path unexpectedly produced a suggestion"
    }
    Set-ElementValue (Find-Descendant $main "AiAssistTimeoutTextBox") "120"

    Select-Tab $main "Training"
    Wait-Until -TimeoutSeconds 10 -Message "Training controls did not load default config settings" -Block {
        $targetValue = Get-ElementValue (Find-Descendant $main "TrainingTargetTextBox")
        $baseModelValue = Get-ElementValue (Find-Descendant $main "TrainingBaseModelTextBox")
        $formatValue = Get-ElementValue (Find-Descendant $main "TrainingFormatTextBox")
        if ($targetValue -eq "axolotl_yaml" `
            -and $baseModelValue -eq "Qwen/Qwen2.5-Coder-7B-Instruct" `
            -and $formatValue -eq "instruction") {
            return "$targetValue/$baseModelValue/$formatValue"
        }

        return $null
    } | Out-Null
    Set-ElementValue (Find-Descendant $main "TrainingSequenceLenTextBox") "0"
    Invoke-Element (Find-Descendant $main "GenerateTrainingConfigButton")
    $trainingSummary = Find-Descendant $main "TrainingSummaryTextBox"
    $trainingValidationText = Wait-Until -TimeoutSeconds 10 -Message "Training tab did not report invalid sequence length" -Block {
        $value = Get-ElementValue $trainingSummary
        if ($value -like "*Training config could not be generated.*" `
            -and $value -like "*sequence length must be a positive whole number*") {
            return $value
        }

        return $null
    }
    $trainingPreview = Get-ElementValue (Find-Descendant $main "TrainingConfigPreviewTextBox")
    if ($trainingPreview -notlike "*No training config was generated.*") {
        throw "Training invalid-input path unexpectedly produced a config"
    }
    Set-ElementValue (Find-Descendant $main "TrainingSequenceLenTextBox") "4096"
    Invoke-Element (Find-Descendant $main "GenerateTrainingConfigButton")
    $trainingGeneratedText = Wait-Until -TimeoutSeconds 10 -Message "Training config summary did not report generated config" -Block {
        $value = Get-ElementValue $trainingSummary
        if ($value -like "*Target: axolotl_yaml*" `
            -and $value -like "*Training launcher: not implemented*" `
            -and $value -like "*Config:*") {
            return $value
        }

        return $null
    }
    $trainingConfigPreviewText = Wait-Until -TimeoutSeconds 10 -Message "Training config preview did not show rendered config" -Block {
        $value = Get-ElementValue (Find-Descendant $main "TrainingConfigPreviewTextBox")
        if ($value -like "*base_model*" `
            -and $value -like "*Qwen/Qwen2.5-Coder-7B-Instruct*" `
            -and $value -like "*dataset_path*" `
            -and $value -like "*train.jsonl*") {
            return $value
        }

        return $null
    }
    $trainingDir = Join-Path (Join-Path $exportDir $projectId) "training"
    $trainingConfigFile = Wait-Until -TimeoutSeconds 10 -Message "Training config file was not written" -Block {
        if (-not (Test-Path $trainingDir)) {
            return $null
        }

        $files = @(Get-ChildItem -Path $trainingDir -Filter "*_axolotl_yaml_config.yaml")
        if ($files.Count -eq 1) {
            return $files[0].FullName
        }

        return $null
    }
    $trainingConfigText = Get-Content -Raw -Path $trainingConfigFile
    if ($trainingConfigText -notlike "*base_model*" `
        -or $trainingConfigText -notlike "*Qwen/Qwen2.5-Coder-7B-Instruct*" `
        -or $trainingConfigText -notlike "*dataset_path*" `
        -or $trainingConfigText -notlike "*train.jsonl*") {
        throw "Training config file did not include expected model and dataset path at $trainingConfigFile"
    }

    Invoke-Element (Find-Descendant $main "ExportJsonlButton")
    Wait-And-ClickOk $process.Id
    $exportPath = Join-Path (Join-Path $exportDir $projectId) "export.jsonl"
    if (-not (Test-Path $exportPath)) {
        throw "export.jsonl was not written at $exportPath"
    }
    $exportCount = (Get-Content -Path $exportPath).Count
    if ($exportCount -ne 22) {
        throw "Expected 22 exported examples, found $exportCount in $exportPath"
    }

    $syntheticRows = for ($index = 0; $index -lt 3; $index++) {
        [ordered]@{
            instruction = "As an AI language model, write a synthetic quality smoke row $index."
            input = ""
            output = "As an AI language model, certainly here is a repeated boilerplate answer for synthetic triage."
        }
    }
    $syntheticLines = $syntheticRows | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 5 }
    [System.IO.File]::AppendAllText(
        $examplesPath,
        ($syntheticLines -join [Environment]::NewLine) + [Environment]::NewLine,
        $utf8NoBom
    )

    Stop-CorpusStudio $process
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id
    $syntheticQualityText = Wait-Until -TimeoutSeconds 10 -Message "Synthetic quality triage did not report warning issues" -Block {
        $summaryValue = Get-ElementValue (Find-Descendant $main "QualitySummaryTextBox")
        $triageValue = Get-ElementText (Find-Descendant $main "QualityTriageSummaryTextBlock")
        if ($summaryValue -like "*Examples: 25*" `
            -and $summaryValue -like "*Synthetic pattern warnings:*" `
            -and $summaryValue -notlike "*Synthetic pattern warnings: 0*" `
            -and $triageValue -like "*Repair:*") {
            return "$summaryValue`n$triageValue"
        }

        return $null
    }
    Select-ListItemContaining $main "SyntheticPatternIssuesListBox" "generic_phrase" | Out-Null
    Invoke-Element (Find-Descendant $main "PrepareSyntheticRewriteButton")
    $syntheticRewritePreparedText = Wait-Until -TimeoutSeconds 10 -Message "Synthetic triage did not prepare AI Assist rewrite controls" -Block {
        $actionValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistActionComboBox")
        $instructionValue = Get-ElementValue (Find-Descendant $main "AiAssistInstructionTextBox")
        if ($actionValue -eq "rewrite-output" `
            -and $instructionValue -like "*Rewrite row 23*" `
            -and $instructionValue -like "*Repair guidance:*" `
            -and $instructionValue -like "*one corrected JSON object only*") {
            return "$actionValue`n$instructionValue"
        }

        return $null
    }
    Select-Tab $main "Writing Studio"
    $syntheticDraftPreparedText = Wait-Until -TimeoutSeconds 10 -Message "Synthetic triage did not load affected row into the draft editor" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*As an AI language model*" `
            -and $value -like "*synthetic quality smoke row 0*") {
            return $value
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "PrepareSyntheticBatchRewriteButton")
    $syntheticBatchRewritePreparedText = Wait-Until -TimeoutSeconds 10 -Message "Synthetic triage did not prepare batch rewrite controls" -Block {
        $actionValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistActionComboBox")
        $instructionValue = Get-ElementValue (Find-Descendant $main "AiAssistInstructionTextBox")
        $batchSummaryValue = Get-ElementText (Find-Descendant $main "AiAssistRewriteBatchSummaryTextBlock")
        if ($actionValue -eq "rewrite-output" `
            -and $instructionValue -like "*Rewrite affected rows 23, 24, 25 as a batch*" `
            -and $instructionValue -like "*corrected JSONL rows only*" `
            -and $instructionValue -like "*generic_phrase*" `
            -and $batchSummaryValue -like "*Saved rewrite batch for rows 23, 24, 25*") {
            return "$actionValue`n$instructionValue`n$batchSummaryValue"
        }

        return $null
    }
    if (-not (Test-Path $aiAssistRewriteBatchesPath)) {
        throw "AI Assist rewrite batches were not written at $aiAssistRewriteBatchesPath"
    }
    $rewriteBatchJson = Get-Content -Raw -Path $aiAssistRewriteBatchesPath
    if ($rewriteBatchJson -notlike "*synthetic quality smoke row 0*" `
        -or $rewriteBatchJson -notlike "*Rewrite affected rows 23, 24, 25 as a batch*" `
        -or $rewriteBatchJson -notlike "*generic_phrase*") {
        throw "AI Assist rewrite batch file did not include expected source draft and instruction at $aiAssistRewriteBatchesPath"
    }
    Select-Tab $main "Writing Studio"
    $syntheticBatchDraftPreparedText = Wait-Until -TimeoutSeconds 10 -Message "Synthetic batch rewrite did not load affected rows into the draft editor" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*synthetic quality smoke row 0*" `
            -and $value -like "*synthetic quality smoke row 1*" `
            -and $value -like "*synthetic quality smoke row 2*" `
            -and $value.Trim().StartsWith("[")) {
            return $value
        }

        return $null
    }

    Stop-CorpusStudio $process
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id
    Select-Tab $main "AI Assist"
    $syntheticRewriteBatchReloadedText = Wait-Until -TimeoutSeconds 10 -Message "Saved AI Assist rewrite batch was not reloaded" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistRewriteBatchSummaryTextBlock")
        if ($summaryValue -like "*Saved rewrite batches: 1*") {
            return $summaryValue
        }

        return $null
    }
    Select-ListItemContaining $main "AiAssistRewriteBatchesListBox" "3 row(s)" | Out-Null
    Invoke-Element (Find-Descendant $main "ResumeAiAssistRewriteBatchButton")
    $syntheticRewriteBatchResumedText = Wait-Until -TimeoutSeconds 10 -Message "Saved AI Assist rewrite batch did not resume into the draft" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*synthetic quality smoke row 0*" `
            -and $value -like "*synthetic quality smoke row 1*" `
            -and $value -like "*synthetic quality smoke row 2*" `
            -and $value.Trim().StartsWith("[")) {
            return $value
        }

        return $null
    }
    Select-Tab $main "AI Assist"
    $syntheticRewriteBatchInstructionReloadedText = Wait-Until -TimeoutSeconds 10 -Message "Saved AI Assist rewrite batch did not restore the instruction" -Block {
        $instructionValue = Get-ElementValue (Find-Descendant $main "AiAssistInstructionTextBox")
        $summaryValue = Get-ElementText (Find-Descendant $main "AiAssistRewriteBatchSummaryTextBlock")
        if ($instructionValue -like "*Rewrite affected rows 23, 24, 25 as a batch*" `
            -and $summaryValue -like "*Resumed rewrite batch for rows 23, 24, 25*") {
            return "$instructionValue`n$summaryValue"
        }

        return $null
    }

    $preferenceProjectName = "Smoke Preference $stamp"
    $preferenceProjectId = "smoke_preference_$stamp"
    $preferenceProjectPath = Join-Path $dataDir $preferenceProjectId
    New-Item -ItemType Directory -Force -Path $preferenceProjectPath | Out-Null
    $preferenceProject = [ordered]@{
        id = $preferenceProjectId
        name = $preferenceProjectName
        schema_id = "preference"
        created_at = "2026-06-30T12:00:00Z"
        updated_at = "2026-06-30T12:00:00Z"
        split_settings = [ordered]@{
            train_ratio = 0.9
            validation_ratio = 0.05
            seed = 42
        }
    } | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText((Join-Path $preferenceProjectPath "project.json"), $preferenceProject + [Environment]::NewLine, $utf8NoBom)

    $preferenceRows = @(
        [ordered]@{
            prompt = "Explain recursion simply."
            chosen = "Recursion is when a function calls itself to solve a smaller version of the same problem."
            rejected = "Recursion is when a function calls itself to solve a smaller version of the same problem."
            reason = "This pair is intentionally weak so the preference review UI can flag low contrast."
        },
        [ordered]@{
            prompt = "Describe a Python list."
            chosen = "A Python list stores ordered mutable values and supports indexing."
            rejected = "A Python list stores values and supports grouping."
            reason = "This pair has some contrast but still shares much of the same claim."
        },
        [ordered]@{
            prompt = "How should code close a file after reading?"
            chosen = "Use a context manager so the file is closed safely even when an error occurs."
            rejected = "Open the file and hope the program exits soon."
            reason = "This pair has clear quality contrast."
        }
    )
    $preferenceLines = $preferenceRows | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 5 }
    [System.IO.File]::WriteAllLines((Join-Path $preferenceProjectPath "examples.jsonl"), $preferenceLines, $utf8NoBom)

    Stop-CorpusStudio $process
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id
    Select-ListItemContaining $main "ProjectsListBox" $preferenceProjectId | Out-Null

    Select-Tab $main "Preference Review"
    $preferenceRankingText = Wait-Until -TimeoutSeconds 10 -Message "Preference ranking summary did not report weak/moderate/strong pairs" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "PreferenceRankingSummaryTextBlock")
        if ($summaryValue -like "*Ranking: 1 weak, 1 moderate, 1 strong.*" `
            -and $summaryValue -like "*Filter: All, showing 3 of 3*") {
            return $summaryValue
        }

        return $null
    }
    Select-ComboBoxItem $main "PreferenceContrastFilterComboBox" "Weak" | Out-Null
    $preferenceWeakFilterText = Wait-Until -TimeoutSeconds 10 -Message "Preference weak filter did not narrow ranking queue" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "PreferenceRankingSummaryTextBlock")
        if ($summaryValue -like "*Filter: Weak, showing 1 of 3*") {
            return $summaryValue
        }

        return $null
    }
    Select-ComboBoxItem $main "PreferenceContrastFilterComboBox" "All" | Out-Null
    Invoke-Element (Find-Descendant $main "ExportPreferenceRankingButton")
    $preferenceRankingExportText = Wait-Until -TimeoutSeconds 10 -Message "Preference ranking export did not update the review summary" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "PreferenceReviewSummaryTextBlock")
        if ($summaryValue -like "*Exported 3 visible preference ranking item(s) for DPO review:*" `
            -and $summaryValue -like "*preference_ranking.json*") {
            return $summaryValue
        }

        return $null
    }
    $preferenceReviewDir = Join-Path (Join-Path $exportDir $preferenceProjectId) "preference_review"
    $preferenceRankingFile = Wait-Until -TimeoutSeconds 10 -Message "Preference ranking export file was not written" -Block {
        if (-not (Test-Path $preferenceReviewDir)) {
            return $null
        }

        $files = @(Get-ChildItem -Path $preferenceReviewDir -Filter "*_preference_ranking.json")
        if ($files.Count -eq 1) {
            return $files[0].FullName
        }

        return $null
    }
    $preferenceRankingJson = Get-Content -Raw -Path $preferenceRankingFile
    if ($preferenceRankingJson -notlike '*"item_count": 3*' `
        -or $preferenceRankingJson -notlike '*DPO and reward-model preference-pair review*' `
        -or $preferenceRankingJson -notlike '*"contrast": "weak"*' `
        -or $preferenceRankingJson -notlike '*"contrast": "strong"*') {
        throw "Preference ranking export did not include expected review fields at $preferenceRankingFile"
    }
    Invoke-Element (Find-Descendant $main "PreparePreferenceBatchJudgeButton")
    $preferenceBatchJudgePreparedText = Wait-Until -TimeoutSeconds 10 -Message "Preference batch judge did not prepare AI Assist controls" -Block {
        $actionValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistActionComboBox")
        $instructionValue = Get-ElementValue (Find-Descendant $main "AiAssistInstructionTextBox")
        if ($actionValue -eq "judge-preference-strength" `
            -and $instructionValue -like "*Judge 3 visible preference pair(s)*" `
            -and $instructionValue -like "*DPO or reward-model readiness*" `
            -and $instructionValue -like "*Visible ranking preview:*") {
            return "$actionValue`n$instructionValue"
        }

        return $null
    }
    Select-Tab $main "Writing Studio"
    $preferenceBatchDraftPreparedText = Wait-Until -TimeoutSeconds 10 -Message "Preference batch judge did not load visible pairs into the draft editor" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*Explain recursion simply*" `
            -and $value -like "*Describe a Python list*" `
            -and $value -like "*How should code close a file after reading?*" `
            -and $value.Trim().StartsWith("[")) {
            return $value
        }

        return $null
    }
    Select-Tab $main "Preference Review"
    Select-ComboBoxItem $main "PreferenceContrastFilterComboBox" "Weak" | Out-Null
    Select-ListItemContaining $main "PreferenceExamplesListBox" "Explain recursion" | Out-Null
    $preferenceReviewText = Wait-Until -TimeoutSeconds 10 -Message "Preference review tab did not show chosen/rejected contrast" -Block {
        $summaryValue = Get-ElementText (Find-Descendant $main "PreferenceReviewSummaryTextBlock")
        $promptValue = Get-ElementValue (Find-Descendant $main "PreferencePromptTextBox")
        $chosenValue = Get-ElementValue (Find-Descendant $main "PreferenceChosenTextBox")
        $rejectedValue = Get-ElementValue (Find-Descendant $main "PreferenceRejectedTextBox")
        if ($summaryValue -like "*Contrast: weak*" `
            -and $summaryValue -like "*Token overlap: 100%*" `
            -and $promptValue -like "*Explain recursion simply*" `
            -and $chosenValue -like "*smaller version of the same problem*" `
            -and $rejectedValue -like "*smaller version of the same problem*") {
            return "$summaryValue`n$promptValue`n$chosenValue`n$rejectedValue"
        }

        return $null
    }
    Invoke-Element (Find-Descendant $main "PreparePreferenceJudgeButton")
    $preferenceJudgePreparedText = Wait-Until -TimeoutSeconds 10 -Message "Preference review did not prepare AI Assist judge controls" -Block {
        $actionValue = Get-ElementSelectionName (Find-Descendant $main "AiAssistActionComboBox")
        $instructionValue = Get-ElementValue (Find-Descendant $main "AiAssistInstructionTextBox")
        if ($actionValue -eq "judge-preference-strength" `
            -and $instructionValue -like "*Judge preference strength for row 1*" `
            -and $instructionValue -like "*DPO or reward-model training*" `
            -and $instructionValue -like "*do not automatically accept*") {
            return "$actionValue`n$instructionValue"
        }

        return $null
    }
    Select-Tab $main "Writing Studio"
    $preferenceDraftPreparedText = Wait-Until -TimeoutSeconds 10 -Message "Preference judge preparation did not load pair into the draft editor" -Block {
        $value = Get-ElementValue (Find-Descendant $main "DraftTextBox")
        if ($value -like "*Explain recursion simply*" `
            -and $value -like "*`"chosen`"*" `
            -and $value -like "*`"rejected`"*") {
            return $value
        }

        return $null
    }

    [pscustomobject]@{
        ProjectId = $projectId
        PreferenceProjectId = $preferenceProjectId
        ExamplesPath = $examplesPath
        ExportPath = $exportPath
        FirstJsonLength = $firstJson.Length
        ReloadedJsonLength = $reloadedJson.Length
        ImportPreviewLength = $importPreview.Length
        QuarantineReviewLength = $quarantineReviewText.Length
        FirstQualityLength = $firstQuality.Length
        FirstQualityHistoryLength = $firstQualityHistory.Length
        ReloadedQualityLength = $reloadedQuality.Length
        SyntheticQualityLength = $syntheticQualityText.Length
        SyntheticRewritePreparedLength = $syntheticRewritePreparedText.Length
        SyntheticDraftPreparedLength = $syntheticDraftPreparedText.Length
        SyntheticBatchRewritePreparedLength = $syntheticBatchRewritePreparedText.Length
        SyntheticBatchDraftPreparedLength = $syntheticBatchDraftPreparedText.Length
        SyntheticRewriteBatchReloadedLength = $syntheticRewriteBatchReloadedText.Length
        SyntheticRewriteBatchResumedLength = $syntheticRewriteBatchResumedText.Length
        SyntheticRewriteBatchInstructionReloadedLength = $syntheticRewriteBatchInstructionReloadedText.Length
        PreferenceRankingLength = $preferenceRankingText.Length
        PreferenceWeakFilterLength = $preferenceWeakFilterText.Length
        PreferenceRankingExportLength = $preferenceRankingExportText.Length
        PreferenceBatchJudgePreparedLength = $preferenceBatchJudgePreparedText.Length
        PreferenceBatchDraftPreparedLength = $preferenceBatchDraftPreparedText.Length
        PreferenceReviewLength = $preferenceReviewText.Length
        PreferenceJudgePreparedLength = $preferenceJudgePreparedText.Length
        PreferenceDraftPreparedLength = $preferenceDraftPreparedText.Length
        SplitSummaryLength = $splitText.Length
        SplitWarningLength = $splitWarningText.Length
        EvaluationComparisonHistoryLength = $evaluationComparisonHistoryText.Length
        EvaluationReportHistoryLength = $evaluationReportHistoryText.Length
        EvaluationComparisonLength = $evaluationComparisonText.Length
        EvaluationFailureQueueLength = $evaluationFailureQueueText.Length
        EvaluationFailureEditDraftLength = $evaluationFailureEditDraftText.Length
        EvaluationFailurePreparedLength = $evaluationFailurePreparedText.Length
        EvaluationFailureDraftLength = $evaluationFailureDraftText.Length
        EvaluationManualReviewLength = $evaluationManualReviewText.Length
        EvaluationRegressionRerunLength = $evaluationRegressionRerunText.Length
        LabSettingsSavedLength = $labSettingsSavedText.Length
        LabSettingsReloadedLength = $labSettingsReloadedText.Length
        EvaluationModelRefreshErrorLength = $evaluationModelRefreshErrorText.Length
        EvaluationHealthLength = $evaluationHealthText.Length
        EvaluationPreflightLength = $evaluationPreflightText.Length
        EvaluationValidationLength = $evaluationValidationText.Length
        AiAssistQueueLoadedLength = $aiAssistQueueLoadedText.Length
        AiAssistRejectedLength = $aiAssistRejectedText.Length
        AiAssistQueueSearchSortLength = $aiAssistQueueSearchSortText.Length
        AiAssistQueueViewSavedLength = $aiAssistQueueViewSavedText.Length
        AiAssistQueueViewLoadedLength = $aiAssistQueueViewLoadedText.Length
        AiAssistBulkRejectedLength = $aiAssistBulkRejectedText.Length
        AiAssistBulkAcceptedLength = $aiAssistBulkAcceptedText.Length
        AiAssistBulkUndoOnceLength = $aiAssistBulkUndoOnceText.Length
        AiAssistBulkUndoLength = $aiAssistBulkUndoText.Length
        AiAssistModelRefreshErrorLength = $aiAssistModelRefreshErrorText.Length
        AiAssistHealthLength = $aiAssistHealthText.Length
        AiAssistValidationLength = $aiAssistValidationText.Length
        TrainingValidationLength = $trainingValidationText.Length
        TrainingGeneratedLength = $trainingGeneratedText.Length
        TrainingConfigPreviewLength = $trainingConfigPreviewText.Length
        SplitDirectory = $splitDir
        QuarantinePath = $quarantineFile
        EvaluationReportPath = $evaluationReportPath
        EvaluationComparisonReportPath = $evaluationComparisonReportPath
        AiAssistQueuePath = $aiAssistQueuePath
        AiAssistQueueViewsPath = $aiAssistQueueViewsPath
        AiAssistRewriteBatchesPath = $aiAssistRewriteBatchesPath
        TrainingConfigPath = $trainingConfigFile
        PreferenceRankingPath = $preferenceRankingFile
    }
}
finally {
    Stop-CorpusStudio $process

    if ($null -eq $oldData) {
        Remove-Item Env:CORPUS_STUDIO_DATA_DIR -ErrorAction SilentlyContinue
    } else {
        $env:CORPUS_STUDIO_DATA_DIR = $oldData
    }

    if ($null -eq $oldExport) {
        Remove-Item Env:CORPUS_STUDIO_EXPORT_DIR -ErrorAction SilentlyContinue
    } else {
        $env:CORPUS_STUDIO_EXPORT_DIR = $oldExport
    }

    if (Test-Path $importPath) {
        Remove-Item -LiteralPath $importPath -Force
    }
}
