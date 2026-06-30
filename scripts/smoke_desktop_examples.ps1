param(
    [string]$Configuration = "Debug"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

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

function Wait-Window([int]$ProcessId) {
    Wait-Until -TimeoutSeconds 20 -Message "Window for process $ProcessId was not found" -Block {
        [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
            [System.Windows.Automation.TreeScope]::Children,
            (Condition-ByProcessId $ProcessId)
        )
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

function Select-Tab($Root, [string]$Name) {
    $tab = Find-ByName $Root $Name 10
    $tab.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
}

function Wait-And-ClickOk([int]$ProcessId) {
    $ok = Wait-Until -TimeoutSeconds 10 -Message "OK button was not found" -Block {
        $windows = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Children,
            (Condition-ByProcessId $ProcessId)
        )

        foreach ($window in $windows) {
            $button = $window.FindFirst(
                [System.Windows.Automation.TreeScope]::Descendants,
                (Condition-ByName "OK")
            )
            if ($null -ne $button) {
                return $button
            }
        }

        return $null
    }

    Invoke-Element $ok
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

    Invoke-Element (Find-Descendant $main "SaveExampleButton")
    Wait-And-ClickOk $process.Id
    Invoke-Element (Find-Descendant $main "RunQualityButton")
    $qualitySummary = Find-Descendant $main "QualitySummaryTextBox"
    $firstQuality = Wait-Until -TimeoutSeconds 10 -Message "Quality summary did not report one example" -Block {
        $value = Get-ElementValue $qualitySummary
        if ($value -like "*Examples: 1*" -and $value -like "*Exact duplicates: 0*") {
            return $value
        }

        return $null
    }

    Select-Tab $main "Examples"
    $detail = Find-Descendant $main "SelectedExampleTextBox"
    $firstJson = Wait-Until -TimeoutSeconds 10 -Message "Saved example JSON did not populate" -Block {
        $value = Get-ElementValue $detail
        if ($value -like "*`"instruction`"*" -and $value -like "*Explain what a variable is*") {
            return $value
        }

        return $null
    }

    $projectPath = Join-Path $dataDir $projectId
    $examplesPath = Join-Path $projectPath "examples.jsonl"
    if (-not (Test-Path $examplesPath)) {
        throw "examples.jsonl was not written at $examplesPath"
    }

    Stop-CorpusStudio $process
    $process = Start-CorpusStudio $dataDir $exportDir
    $main = Wait-Window $process.Id
    $qualitySummary = Find-Descendant $main "QualitySummaryTextBox"
    $reloadedQuality = Wait-Until -TimeoutSeconds 10 -Message "Reloaded quality summary did not report one example" -Block {
        $value = Get-ElementValue $qualitySummary
        if ($value -like "*Examples: 1*" -and $value -like "*Exact duplicates: 0*") {
            return $value
        }

        return $null
    }

    Select-Tab $main "Examples"
    $detail = Find-Descendant $main "SelectedExampleTextBox"
    $reloadedJson = Wait-Until -TimeoutSeconds 10 -Message "Reloaded example JSON did not populate" -Block {
        $value = Get-ElementValue $detail
        if ($value -like "*`"instruction`"*" -and $value -like "*Explain what a variable is*") {
            return $value
        }

        return $null
    }

    Invoke-Element (Find-Descendant $main "ExportJsonlButton")
    Wait-And-ClickOk $process.Id
    $exportPath = Join-Path (Join-Path $exportDir $projectId) "export.jsonl"
    if (-not (Test-Path $exportPath)) {
        throw "export.jsonl was not written at $exportPath"
    }

    [pscustomobject]@{
        ProjectId = $projectId
        ExamplesPath = $examplesPath
        ExportPath = $exportPath
        FirstJsonLength = $firstJson.Length
        ReloadedJsonLength = $reloadedJson.Length
        FirstQualityLength = $firstQuality.Length
        ReloadedQualityLength = $reloadedQuality.Length
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
}
