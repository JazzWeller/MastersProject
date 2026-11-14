# Play the strongest trained bot in the GUI.
#
#   powershell -ExecutionPolicy Bypass -File play_best.ps1                          # you (Fignor) vs the bot (Igor)
#   powershell -ExecutionPolicy Bypass -File play_best.ps1 -Deck igor -BotDeck fignor -First p2
#   powershell -ExecutionPolicy Bypass -File play_best.ps1 -Sims 400                # a stronger, slower bot
#   powershell -ExecutionPolicy Bypass -File play_best.ps1 -Run tier2-v1-s2         # an earlier network
#   powershell -ExecutionPolicy Bypass -File play_best.ps1 -Run real-v1 -SasPair 3   # two random real decks within 3 SAS
#
# The network runs in WSL (ml.infer_server, torch + CUDA); the GUI runs here and its bot
# (`search-net`, agent/agents/registry_entries.py) asks the server over 127.0.0.1:<Port>. The bot
# searches as the Tier 2 rating ladder measured it (within-turn, 100 simulations by default:
# ~0.7 s a decision). The server stops when the GUI closes. -Check starts and stops the server
# without opening the GUI.

param(
    [string]$Run = "tier2-v1-s3",  # a v1 run exported under Code/Non-GUI/results/<any folder> (its best network)
    [int]$Sims = 100,
    [string]$Deck = "fignor",      # yours: a bundled preset or a saved deck JSON path
    [string]$BotDeck = "igor",
    [ValidateSet("", "p1", "p2")][string]$First = "",
    [string]$Seed = "",
    [int]$Port = 6150,
    [double]$SasPair = -1,         # 0 or more: you and the bot get two random held-out real decks within this many SAS
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$gui = $PSScriptRoot
$nonGui = Join-Path (Split-Path $gui -Parent) "Non-GUI"
$runDir = Get-ChildItem (Join-Path $nonGui "results") -Directory | ForEach-Object { Join-Path $_.FullName $Run } |
    Where-Object { Test-Path (Join-Path $_ "best.json") } | Select-Object -First 1
if (-not $runDir) { throw "no exported run '$Run' with a best.json under $nonGui\results\*" }
if ($SasPair -ge 0) {
    $out = Join-Path $env:TEMP "kf-sas-pair"
    Push-Location $nonGui
    try { $picked = & python -X utf8 -m tools.pick_decks --out-dir $out --max-diff $SasPair } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "tools.pick_decks failed" }
    $Deck, $BotDeck = $picked[0], $picked[1]
    Write-Host $picked[2]
}
$bestJson = Get-Content (Join-Path $runDir "best.json") | ConvertFrom-Json
$ckpt = Get-ChildItem (Join-Path $runDir "checkpoints") | Where-Object { $_.Name.StartsWith($bestJson.checkpoint) } | Select-Object -First 1
if (-not $ckpt) { throw "no checkpoint $($bestJson.checkpoint) under $runDir\checkpoints" }

$wslNonGui = (wsl wslpath -a ($nonGui -replace '\\', '/')).Trim()
$wslCkpt = (wsl wslpath -a ($ckpt.FullName -replace '\\', '/')).Trim()
Write-Host "Starting the $Run network (step $($bestJson.step)) in WSL on port $Port..."
# Start-Process joins its arguments with spaces, unquoted: the bash command must arrive as one quoted argument.
$cmd = "cd '$wslNonGui' && exec ~/torchenv/bin/python -m ml.infer_server '$wslCkpt' --port $Port --device cuda"
$server = Start-Process wsl -ArgumentList @("bash", "-c", "`"$cmd`"") -PassThru -WindowStyle Hidden

try {
    $deadline = (Get-Date).AddSeconds(90)
    while ($true) {
        try { $tcp = New-Object Net.Sockets.TcpClient("127.0.0.1", $Port); $tcp.Close(); break } catch { }
        if ($server.HasExited) { throw "the inference server exited (code $($server.ExitCode)); run it by hand in WSL to see why" }
        if ((Get-Date) -gt $deadline) { throw "the inference server didn't open port $Port within 90 s" }
        Start-Sleep -Milliseconds 500
    }
    if ($Check) { Write-Host "Server up on port $Port (check only)."; return }
    Write-Host "Server up. Opening the GUI: you are player 1 ($Deck), the bot is player 2 ($BotDeck, $Sims simulations a decision)."
    $argv = @("main.py", "--p1", "human", "--p2", "bot", "--p1-deck", $Deck, "--p2-deck", $BotDeck, "--bot", "search-net@$Port/$Sims")
    if ($First) { $argv += @("--first", $First) }
    if ($Seed) { $argv += @("--seed", $Seed) }
    Push-Location $gui
    try { & python @argv } finally { Pop-Location }
}
finally {
    wsl bash -c "pkill -f 'ml.infer_serve[r] .*--port $Port'" | Out-Null
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Inference server stopped."
}
