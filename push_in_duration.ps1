$ErrorActionPreference = "Stop"
$repo = "https://github.com/sarumathi2120/Instamart-Customer-copilot.git"

# Run this script from the extracted project folder.
$needed = @("generate_data.py", "mock_llm.py", "part_a.py", "part_b.py", "part_c.py", "part_d.py", "validate.py", "README.md", "requirements.txt", ".gitignore", "sql/queries.sql")
foreach ($file in $needed) {
    if (-not (Test-Path $file)) {
        throw "Required file missing: $file. Open the extracted project folder before running this script."
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or is not on PATH."
}
if (-not (Test-Path ".git")) { git init; if ($LASTEXITCODE -ne 0) { throw "git init failed" } }

git branch -M main
if ($LASTEXITCODE -ne 0) { throw "Could not set branch main" }

$remote = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or -not $remote) {
    git remote add origin $repo
    if ($LASTEXITCODE -ne 0) { throw "Could not add remote" }
} elseif ($remote.Trim() -ne $repo) {
    throw "Existing 'origin' points to $remote. Not changing it automatically."
}

git add -A
if ($LASTEXITCODE -ne 0) { throw "git add failed" }

git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "Complete Instamart capstone project"
    if ($LASTEXITCODE -ne 0) { throw "Commit failed. Check git user.name and user.email." }
}

git push -u origin main
if ($LASTEXITCODE -ne 0) { throw "GitHub push failed. Check GitHub login and permissions." }
Write-Host ""
Write-Host "Upload complete: https://github.com/sarumathi2120/Instamart-Customer-copilot/tree/main"
