@echo off
REM Abre la aplicacion de escenarios de XRP
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -m pip install -q -r analysis\requirements.txt
  start "" pyw analysis\app_xrp.py
) else (
  python -m pip install -q -r analysis\requirements.txt
  start "" pythonw analysis\app_xrp.py
)
