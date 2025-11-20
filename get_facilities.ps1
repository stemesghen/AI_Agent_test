# get_facilities.ps1
# -------------------------------
# Calls the company GET /Facilities endpoint
# and prints the JSON response to the console.

param(
    [string]$LocType = "locationType"     # e.g. Ports, Facilities, Terminals
)

# Environment / Auth setup
$baseUrl   = "https://owic-imp.ims.insurity.com/WebAPI"      # replace with your base URL
$token     = "6829747d-6a3a-4fc3-b2be-12987e7ac788"       # Bearer Token
$url       = "$baseUrl/api/$LocType/Facilities"

# Headers
$headers = @{
    "Authorization" = "Bearer $token"
    "Accept"        = "application/json"
    "Content-Type"  = "application/json"
}

Write-Host "Calling $url ..." -ForegroundColor Cyan

# Request
$response = Invoke-RestMethod -Uri $url -Headers $headers -Method Get -TimeoutSec 20

# Optional: pretty print JSON to file
$response | ConvertTo-Json -Depth 8 | Out-File "facilities_response.json" -Encoding utf8

Write-Host "Response saved to facilities_response.json" -ForegroundColor Green
