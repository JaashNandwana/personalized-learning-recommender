@echo off
echo ========================================================
echo   Pushing code to GitHub:
echo   https://github.com/JaashNandwana/personalized-learning-recommender.git
echo ========================================================
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo If prompted, please authorize Git Credential Manager in your browser.
)
echo.
pause
