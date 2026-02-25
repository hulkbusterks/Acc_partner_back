$env:PYTHONPATH = "$PWD"
.\.venv\Scripts\python.exe -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }