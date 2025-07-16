"""
QR-Code Reader Modul für Zählerkarten-Pipeline
==============================================
Liest alle *_qr.png Bilder aus dem QR-Ordner,
versucht mehrfach QR-Decoding mit Bildverbesserung,
legt pro Bild eine eigene JSON-Datei an und führt
eine Fehlerliste als FEHLT_MANUELL.json.

Version: 2.3
Autor: Oliver Krispel
Datum: 09.07.2025
"""

import os
import cv2
import json
import datetime
import numpy as np
from pyzbar.pyzbar import decode
from pathlib import Path

# ------------------ Konfiguration laden ------------------
with open(r"C:\ZaehlerkartenV2\Config\qr_code_reader_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

QR_DIR = config["qr_dir"]
OUT_DIR = config["output_dir"]
DEBUG_DIR = os.path.join(QR_DIR, config.get("debug_subdir", "debug_qr"))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ------------------ Hilfsfunktionen ------------------

def try_decode_pyzbar(image):
    decoded_objects = decode(image)
    for obj in decoded_objects:
        if obj.type == "QRCODE":
            return obj.data.decode("utf-8")
    return None

def try_decode_cv2(image):
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    return data if data else None

def enhance_image(image):
    resized = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened

def rotate_image(image, angle):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

def save_json(out_path, data):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ------------------ Hauptverarbeitung ------------------

def main():
    fehlende_qrs = []
    images = sorted([f for f in os.listdir(QR_DIR) if f.lower().endswith("_qr.png")])

    for fname in images:
        fpath = os.path.join(QR_DIR, fname)
        image = cv2.imread(fpath)
        if image is None:
            print(f"[WARN] Bild kann nicht geladen werden: {fname}")
            continue

        decoded = try_decode_pyzbar(image)

        if not decoded:
            enhanced = enhance_image(image)
            decoded = try_decode_pyzbar(enhanced)

        if not decoded:
            decoded = try_decode_cv2(enhanced)

        # 🔁 Fallback-Rotation
        if not decoded:
            for angle in [-10, 10]:
                rotated = rotate_image(enhanced, angle)
                decoded = try_decode_pyzbar(rotated)
                if decoded:
                    break
            if not decoded:
                for angle in [-10, 10]:
                    rotated = rotate_image(enhanced, angle)
                    decoded = try_decode_cv2(rotated)
                    if decoded:
                        break

        base = fname.replace("_qr.png", "")
        json_path = os.path.join(OUT_DIR, base + "_qr.json")

        if decoded:
            print(f"[OK] {fname}: {decoded}")
            save_json(json_path, {"qr": decoded})
        else:
            print(f"[FAIL] Kein QR erkannt: {fname}")
            save_json(json_path, {"qr": "FEHLT_MANUELL"})
            fehlende_qrs.append(base)
            debug_out = os.path.join(DEBUG_DIR, fname)
            cv2.imwrite(debug_out, image)

    # Speichern der Fehlerliste
    fehlt_path = os.path.join(OUT_DIR, "FEHLT_MANUELL.json")
    save_json(fehlt_path, fehlende_qrs)
    print(f"\n📄 Fehlerliste gespeichert: {fehlt_path}")

    # Statistik
    print("\n📊 ZUSAMMENFASSUNG")
    print("===========================")
    print(f"Total Bilder:     {len(images)}")
    print(f"Erfolgreich:      {len(images) - len(fehlende_qrs)}")
    print(f"Fehlgeschlagen:   {len(fehlende_qrs)}")

# ------------------ Startpunkt ------------------
if __name__ == "__main__":
    main()
