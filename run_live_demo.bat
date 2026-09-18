@echo off
echo ===================================================
echo Starting ATLAS Study Sentinel Live Demo Server...
echo ===================================================

start /b python server.py --port 8000
timeout /t 3 /nobreak >nul

echo Starting Public HTTPS Tunnel...
node -e "import('file:///C:/Users/ANAND R/AppData/Local/npm-cache/_npx/c628db61ff937621/node_modules/untun/dist/index.mjs').then(async m => { const t = await m.startTunnel({ port: 8000 }); console.log('\n>>> LIVE DEMO URL FOR JUDGING: ' + await t.getURL() + ' <<<\n'); })"

pause
