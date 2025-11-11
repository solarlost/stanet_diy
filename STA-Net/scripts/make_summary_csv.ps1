Param()

$json = Get-Content "$PSScriptRoot/../results/summary_all.json" -Raw | ConvertFrom-Json
$rows = @()
$subjects = $json | Select-Object -Expand subject | Sort-Object -Unique

function Get-ClassAcc {
    param([object]$rec)
    if ($null -eq $rec) { return $null }
    $mr = $rec.mean_results
    return $mr[$mr.Count - 2]
}

function Get-EEGAcc {
    param([object]$rec)
    if ($null -eq $rec) { return $null }
    $mr = $rec.mean_results
    return $mr[$mr.Count - 1]
}

foreach ($s in $subjects) {
    $ib = $json | Where-Object { $_.subject -eq $s -and $_.variant -eq 'ib' }
    $noib = $json | Where-Object { $_.subject -eq $s -and $_.variant -eq 'noib' }
    $ibClass = Get-ClassAcc $ib
    $noibClass = Get-ClassAcc $noib
    $obj = [pscustomobject]@{
        subject = $s
        ib_class_acc = $ibClass
        noib_class_acc = $noibClass
        diff_ib_minus_noib = if ($ibClass -ne $null -and $noibClass -ne $null) { [double]$ibClass - [double]$noibClass } else { $null }
        ib_eeg_acc = Get-EEGAcc $ib
        noib_eeg_acc = Get-EEGAcc $noib
    }
    $rows += $obj
}

$outPath = Join-Path "$PSScriptRoot/../results" "summary_all.csv"
$rows | Sort-Object subject | Export-Csv -NoTypeInformation -UseCulture -Encoding UTF8 -Path $outPath
Write-Output "Wrote: $outPath"



