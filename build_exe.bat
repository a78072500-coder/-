@echo off
rem Build an exe folder with PyInstaller. Output: dist\jp_subtitle\jp_subtitle.exe
rem (console window is kept so errors and recognition logs stay visible)
pyinstaller --noconfirm --name jp_subtitle ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  --collect-binaries nvidia ^
  run.py
if errorlevel 1 exit /b 1
if exist config.toml copy /Y config.toml dist\jp_subtitle\ >nul
copy /Y config.example.toml dist\jp_subtitle\ >nul
echo.
echo Done: dist\jp_subtitle\jp_subtitle.exe
