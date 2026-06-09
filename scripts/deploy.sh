#!/usr/bin/env bash
# Deploy SpaceCalibrator overlay to install dir.
# Fails loudly if SteamVR has the file locked.

SRC="$(dirname "$0")/../build/artifacts/Release/SpaceCalibrator.exe"
DST="$(dirname "$0")/../install/SpaceCalibrator.exe"

if cp "$SRC" "$DST" 2>/dev/null; then
    echo "[OK] Deployed: $(ls -la "$DST" | awk '{print $6,$7,$8}')"
else
    # Check if SpaceCalibrator or vrserver is holding the file
    LOCKED=$(powershell.exe -NoProfile -Command "
        Get-Process | Where-Object { \$_.Name -match 'SpaceCalibrator|vrserver|vrmonitor' } |
        Select-Object -ExpandProperty Name
    " 2>/dev/null | tr -d '\r' | tr '\n' ' ')
    echo "[FAIL] Deploy failed — file is locked."
    if [ -n "$LOCKED" ]; then
        echo "       Locked by: $LOCKED"
        echo "       Close SteamVR and try again."
    else
        echo "       Close SteamVR and try again."
    fi
    exit 1
fi
