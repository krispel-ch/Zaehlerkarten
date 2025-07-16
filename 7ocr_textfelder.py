"""
Textfelder OCR Modul für Zählerkarten-Pipeline
==============================================
Erkennt Zählernummer und Zählerart mittels Azure OCR,
vergleicht gegen Regeln aus Config, speichert Einzel-JSONs
und erstellt eine Fehlerliste zur Nachbearbeitung.

Version: 1.1
Autor: Oliver Krispel
Datum: 09.07.2025
"""

import os
import re
import time
import json
import requests
from pathlib import Path

# === Config laden ===
config_path = Path(r"C:/ZaehlerkartenV2/Config/ocr_text_config.json")
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

feld_dir = Path(cfg["textfelder_dir"])
output_dir = Path(cfg["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
fehlerliste = []
regex_znr = re.compile(cfg["zaehlernummer_regex"])
erlaubt_art = cfg["zaehlerart_erlaubt"]

# === Azure Config laden ===
azure_config_path = Path(r"C:/ZaehlerkartenV2/Config/azure_config.json")
with open(azure_config_path, 'r') as f:
    az = json.load(f)

AZURE_ENDPOINT = az['endpoint'].rstrip('/') + "/vision/v3.2/read/analyze"
AZURE_KEY = az['key']
HEADERS = {
    'Ocp-Apim-Subscription-Key': AZURE_KEY,
    'Content-Type': 'application/octet-stream'
}

# === Bilder sammeln ===
bilder = sorted(feld_dir.glob("*_zaehlerart.png")) + sorted(feld_dir.glob("*_zaehlernummer.png"))

for img_path in bilder:
    feldtyp = "zaehlerart" if "zaehlerart" in img_path.name else "zaehlernummer"
    basis = img_path.stem.replace(f"_{feldtyp}", "")
    out_json = output_dir / f"{basis}_{feldtyp}.json"

    with open(img_path, 'rb') as f:
        img_data = f.read()

    response = requests.post(AZURE_ENDPOINT, headers=HEADERS, data=img_data)
    if response.status_code != 202:
        print(f"❌ OCR-Fehler bei {img_path.name}: {response.status_code}")
        fehlerliste.append({"datei": basis, "feld": feldtyp})
        continue

    operation_url = response.headers['Operation-Location']
    result = None
    for _ in range(10):
        time.sleep(0.5)
        r = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': AZURE_KEY})
        result = r.json()
        if result.get("status") == "succeeded":
            break

    lines = result.get("analyzeResult", {}).get("readResults", [{}])[0].get("lines", [])
    rawtext = " ".join([line['text'] for line in lines]).strip()

    eintrag = {
        "feld": feldtyp,
        "text_raw": rawtext,
        "text_normalisiert": rawtext.lower().replace(" ", "")
    }

    if feldtyp == "zaehlernummer":
        raw_cleaned = eintrag["text_normalisiert"]
        re_match = re.fullmatch(r"(\d{5,})([^0-9]*)", raw_cleaned)

        if re_match:
            nummer, rest = re_match.groups()
            if rest == "":
                eintrag["text"] = nummer
                eintrag["status"] = "ok"
            else:
                eintrag["text"] = nummer
                eintrag["status"] = "ok"
                eintrag["korrektur"] = True
        else:
            eintrag["text"] = raw_cleaned
            eintrag["status"] = "unsicher"
            fehlerliste.append({"datei": basis, "feld": feldtyp})

    elif feldtyp == "zaehlerart":
        erkannt = None
        for key, varianten in erlaubt_art.items():
            if eintrag["text_normalisiert"] in varianten:
                erkannt = key
                break

        if erkannt:
            eintrag["status"] = "ok"
            eintrag["text"] = erkannt
        else:
            eintrag["status"] = "unsicher"
            eintrag["text"] = eintrag["text_normalisiert"]
            fehlerliste.append({"datei": basis, "feld": feldtyp})

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eintrag, f, indent=2, ensure_ascii=False)
    print(f"[{eintrag['status'].upper()}] {img_path.name} → {eintrag['text']}")

# === Fehlerliste speichern ===
fehler_path = output_dir / cfg["fehlerliste_name"]
with open(fehler_path, "w", encoding="utf-8") as f:
    json.dump(fehlerliste, f, indent=2, ensure_ascii=False)
print(f"\n📄 Fehlerliste gespeichert: {fehler_path}")
