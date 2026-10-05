@echo off
setlocal EnableDelayedExpansion
title YT-DLP CLIP STUDIO FINAL (EXODIA BYPASS)
color 0a

set "DOWNLOAD_DIR=D:\download video\clips"
set "TEMP_DIR=%DOWNLOAD_DIR%\temp"
set "COOKIE_FILE=%DOWNLOAD_DIR%\cookies.txt"
set "UPDATE_MARKER=%DOWNLOAD_DIR%\last_ytdlp_update.marker"

if not exist "%DOWNLOAD_DIR%" mkdir "%DOWNLOAD_DIR%"
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

echo Checking yt-dlp update status...

if not exist "%UPDATE_MARKER%" goto do_update

forfiles /p "%DOWNLOAD_DIR%" /m "last_ytdlp_update.marker" /d -15 /c "cmd /c exit 0" >nul 2>&1
if errorlevel 1 goto skip_update

:do_update
echo.
echo [Auto-Update] yt-dlp has not been updated in 15+ days - updating now...
yt-dlp -U
echo. > "%UPDATE_MARKER%"
echo.
goto after_update

:skip_update
echo yt-dlp was updated recently - skipping auto-update.
echo.

:after_update

:loop
cls
echo ======================================================
echo       YT-DLP CLIP STUDIO (MAX QUALITY EXODIA)
echo ======================================================
echo.
echo Target Folder: %DOWNLOAD_DIR%
echo.

if not exist "%COOKIE_FILE%" (
    echo WARNING: cookies.txt not found in %DOWNLOAD_DIR%!
    echo.
)

set "SOURCE_TYPE="
set "start=00:00:00"
set "end=FULL"

:ask_source
echo Select Video Source:
echo   [1] YouTube
echo   [2] Other - Instagram / TikTok / etc.
set /p "src_choice=Enter choice (1 or 2): "

if "%src_choice%"=="1" set "SOURCE_TYPE=YT"
if "%src_choice%"=="2" set "SOURCE_TYPE=OTHER"
if not defined SOURCE_TYPE (
    echo Invalid choice, try again.
    echo.
    goto ask_source
)

echo.
set /p "link=Paste Video Link: "

if "%SOURCE_TYPE%"=="YT" goto get_timestamps
echo.
echo [Other-source mode] No trim - full video will be downloaded.
goto after_timestamps

:get_timestamps
echo.
echo Enter timestamps (HH:MM:SS)
echo.
set /p "start=Start Time: "
set /p "end=End Time: "

:after_timestamps

echo.
echo [1/4] Fetching Video Title...

set "TITLE_FILE=%TEMP_DIR%\title.txt"
if exist "%TITLE_FILE%" del "%TITLE_FILE%"

if "%SOURCE_TYPE%"=="YT" goto title_yt
goto title_other

:title_yt
yt-dlp --cookies "%COOKIE_FILE%" --js-runtimes node --get-title "!link!" > "%TITLE_FILE%" 2>nul
goto after_title

:title_other
yt-dlp --cookies "%COOKIE_FILE%" --get-title "!link!" > "%TITLE_FILE%" 2>nul
goto after_title

:after_title

set "VIDEO_TITLE="
if exist "%TITLE_FILE%" set /p VIDEO_TITLE=<"%TITLE_FILE%"
if exist "%TITLE_FILE%" del "%TITLE_FILE%"
if not defined VIDEO_TITLE set "VIDEO_TITLE=Untitled Clip"

echo Title: "!VIDEO_TITLE!"

echo.
echo [2/4] Downloading...

if "%SOURCE_TYPE%"=="YT" goto download_yt
goto download_other

:download_yt
echo Solving YouTube JS Puzzle and Downloading clip section...
:: THE EXODIA BYPASS: Forces Node.js to solve the puzzle + Uses Cookies for VIP Access
:: Retry/reconnect flags added to survive TLS resets (-10054) on --download-sections pulls
yt-dlp ^
--cookies "%COOKIE_FILE%" ^
--js-runtimes node ^
-f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" ^
--download-sections "*!start!-!end!" ^
--force-keyframes-at-cuts ^
--hls-prefer-native ^
--retries 20 ^
--fragment-retries 20 ^
--retry-sleep linear=1:10:2 ^
--merge-output-format mp4 ^
--newline ^
--progress ^
-o "%TEMP_DIR%\clip.%%(ext)s" ^
"!link!"
goto after_download

:download_other
echo Downloading full video, no trim - cookies.txt must contain a live Instagram/other login session...
yt-dlp ^
--cookies "%COOKIE_FILE%" ^
-f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" ^
--retries 20 ^
--fragment-retries 20 ^
--retry-sleep linear=1:10:2 ^
--merge-output-format mp4 ^
--newline ^
--progress ^
-o "%TEMP_DIR%\clip.%%(ext)s" ^
"!link!"
goto after_download

:after_download

if not exist "%TEMP_DIR%\clip.mp4" (
    echo.
    echo ERROR: Download failed!
    echo If this was an Instagram/Other link and you saw "empty media response",
    echo that is a known open yt-dlp/Instagram issue, not a script bug.
    echo Try: yt-dlp --update-to nightly    ^(sometimes has newer site fixes^)
    echo.
    pause
    goto loop
)

echo.
echo [3/4] Processing Python Editor...

set "FINAL_OUTPUT=%DOWNLOAD_DIR%\REEL_%random%.mp4"

python face_track_crop.py "%TEMP_DIR%\clip.mp4" "!FINAL_OUTPUT!" "!link!" "!start!" "!end!" "!VIDEO_TITLE!"

echo.
echo [4/4] Cleaning Temporary Files...

del "%TEMP_DIR%\clip.mp4"

echo.
echo =========================================
echo        CLIP EXPORT COMPLETE
echo =========================================
echo Saved to: !FINAL_OUTPUT!
echo.
pause
goto loop