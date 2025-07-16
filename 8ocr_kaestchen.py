#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kästchen-OCR Modul für Zählerkarten-Pipeline - ERWEITERT
========================================================
Version: 2.0 - Enhanced für GUI + Training-Data-Sammlung
Autor: Oliver Krispel + KI-System
Datum: 15.07.2025

VERBESSERUNGEN:
- Realistische Schwellwerte (0.7 statt 0.9)
- Pro-Kästchen-Bewertung: "ok", "leicht_unsicher", "unsicher", "fehler"
- Detaillierte JSON-Ausgabe für GUI
- Intelligente Gesamtbewertung
- Vorbereitung für Training-Data-Sammlung

Erkennt Z-Kästchen (Zählerstand) und D-Kästchen (Datum)
via Keras-Modell mit verbesserter Fehleranalyse.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import load_model
from typing import Dict, List, Any
from datetime import datetime
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Konfiguration laden ===
with open(r"C:/ZaehlerkartenV2/Config/ocr_kaestchen_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

Z_DIR = Path(cfg["z_kaestchen_dir"])
D_DIR = Path(cfg["d_kaestchen_dir"])
OUT_DIR = Path(cfg["output_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(cfg["model_path"])
JAHR_FIX = cfg["jahr_fix"]

# ERWEITERTE SCHWELLWERTE - REALISTISCH
CONFIDENCE_THRESHOLDS = {
    "ok": 0.6,           # Sehr sicher
    "leicht_unsicher": 0.5,   # Noch akzeptabel  
    "unsicher": 0.35,     # Problematisch
    "fehler": 0.0        # Unter 0.5 = fehler
}

FEHLERLISTE_NAME = cfg["fehlerliste_name"]

# === Modell laden ===
try:
    model = load_model(MODEL_PATH)
    logger.info(f"✅ Keras-Modell geladen: {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Fehler beim Laden des Modells: {e}")
    raise

# === Basisnamen finden ===
basisnamen = sorted(set("_".join(p.stem.split("_")[0:2]) for p in Z_DIR.glob("*_z_*.png")))
logger.info(f"📊 Gefunden: {len(basisnamen)} Karten zur Verarbeitung")

# Statistiken
stats = {
    "total_cards": len(basisnamen),
    "total_boxes": 0,
    "z_boxes": 0,
    "d_boxes": 0,
    "status_counts": {
        "ok": 0,
        "leicht_unsicher": 0,
        "unsicher": 0,
        "fehler": 0
    },
    "overall_status_counts": {
        "ok": 0,
        "leicht_unsicher": 0, 
        "unsicher": 0,
        "fehler": 0
    }
}

fehlerliste = []
problematic_cards = []

def predict_digit(img_path: Path) -> tuple:
    """
    Vorhersage für ein einzelnes Kästchen-Bild
    
    Returns:
        (predicted_digit, confidence) oder (None, None) bei Fehler
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0
        img = img.reshape((1, 28, 28, 1))
        
        pred = model.predict(img, verbose=0)[0]
        index = int(np.argmax(pred))
        confidence = float(np.max(pred))
        
        return index, confidence
    except Exception as e:
        logger.warning(f"Fehler bei Vorhersage für {img_path}: {e}")
        return None, None

def get_confidence_status(confidence: float) -> str:
    """Bestimmt Status basierend auf Confidence-Score"""
    if confidence >= CONFIDENCE_THRESHOLDS["ok"]:
        return "ok"
    elif confidence >= CONFIDENCE_THRESHOLDS["leicht_unsicher"]:
        return "leicht_unsicher"
    elif confidence >= CONFIDENCE_THRESHOLDS["unsicher"]:
        return "unsicher"
    else:
        return "fehler"

def extrahiere_scan_datum(dateiname: str) -> str:
    """Extrahiert Datum aus Dateiname als Fallback"""
    try:
        datum_raw = dateiname.split("_")[0]  # z. B. 20250616...
        jahr = datum_raw[:4]
        monat = datum_raw[4:6]
        tag = datum_raw[6:8]
        return f"{tag}.{monat}.{jahr}"
    except:
        return f"01.01.{JAHR_FIX}"  # Fallback

def ist_gueltiges_datum(tag: str, monat: str) -> bool:
    """Prüft ob Tag/Monat ein gültiges Datum ergeben"""
    try:
        t = int(tag) if tag.isdigit() else -1
        m = int(monat) if monat.isdigit() else -1
        return 1 <= t <= 31 and 1 <= m <= 12
    except:
        return False

def calculate_overall_status(z_boxes: List[Dict], d_boxes: List[Dict]) -> str:
    """
    Intelligente Gesamtbewertung basierend auf einzelnen Kästchen
    
    Logik:
    - Alle ok → "ok"
    - 1-2 leicht_unsicher → "leicht_unsicher"  
    - 1-2 unsicher ODER 3+ leicht_unsicher → "unsicher"
    - 3+ unsicher ODER 1+ fehler → "fehler"
    """
    all_boxes = z_boxes + d_boxes
    status_counts = {"ok": 0, "leicht_unsicher": 0, "unsicher": 0, "fehler": 0}
    
    for box in all_boxes:
        status_counts[box["status"]] += 1
    
    # Fehler haben höchste Priorität
    if status_counts["fehler"] > 0:
        return "fehler"
    
    # Viele unsichere Kästchen
    if status_counts["unsicher"] >= 3:
        return "fehler"
    elif status_counts["unsicher"] >= 1:
        return "unsicher"
    
    # Leicht unsichere Kästchen
    if status_counts["leicht_unsicher"] >= 3:
        return "unsicher"
    elif status_counts["leicht_unsicher"] >= 1:
        return "leicht_unsicher"
    
    # Alles ok
    return "ok"

def process_single_card(basis: str) -> Dict[str, Any]:
    """
    Verarbeitet eine einzelne Karte und gibt detaillierte Ergebnisse zurück
    """
    z_boxes = []
    d_boxes = []
    z_wert = []
    z_confidences = []
    d_wert = []
    problematic_boxes = []
    
    # === Z-Kästchen (Zählerstand) verarbeiten ===
    for i in range(9):
        fname = f"{basis}_z_{i:02}.png"
        pfad = Z_DIR / fname
        
        box_info = {
            "position": i,
            "predicted": "x",
            "confidence": 0.0,
            "status": "fehler",
            "image_path": str(pfad) if pfad.exists() else None,
            "exists": pfad.exists()
        }
        
        if pfad.exists():
            digit, conf = predict_digit(pfad)
            if digit is not None:
                # Klasse 10 = leeres Kästchen → "x"
                predicted_char = "x" if digit == 10 else str(digit)
                box_info.update({
                    "predicted": predicted_char,
                    "confidence": round(conf, 3),
                    "status": get_confidence_status(conf)
                })
                z_confidences.append(conf)
            else:
                box_info["status"] = "fehler"
                z_confidences.append(0.0)
        else:
            z_confidences.append(0.0)
        
        z_wert.append(box_info["predicted"])
        z_boxes.append(box_info)
        
        # Statistiken
        stats["z_boxes"] += 1
        stats["status_counts"][box_info["status"]] += 1
        
        if box_info["status"] in ["unsicher", "fehler"]:
            problematic_boxes.append(f"z_{i}")

    # === D-Kästchen (Datum) verarbeiten ===
    for i in range(4):
        fname = f"{basis}_d_{i:02}.png"
        pfad = D_DIR / fname
        
        box_info = {
            "position": i,
            "predicted": "x",
            "confidence": 0.0,
            "status": "fehler",
            "image_path": str(pfad) if pfad.exists() else None,
            "exists": pfad.exists()
        }
        
        if pfad.exists():
            digit, conf = predict_digit(pfad)
            if digit is not None:
                predicted_char = "x" if digit == 10 else str(digit)
                box_info.update({
                    "predicted": predicted_char,
                    "confidence": round(conf, 3),
                    "status": get_confidence_status(conf)
                })
            else:
                box_info["status"] = "fehler"
        
        d_wert.append(box_info["predicted"])
        d_boxes.append(box_info)
        
        # Statistiken
        stats["d_boxes"] += 1
        stats["status_counts"][box_info["status"]] += 1
        
        if box_info["status"] in ["unsicher", "fehler"]:
            problematic_boxes.append(f"d_{i}")

    # === Zählerstand zusammensetzen ===
    vor_komma = "".join([d for d in z_wert[:6] if d != "x"])
    nach_komma = "".join([d for d in z_wert[6:] if d != "x"])
    zaehler_text = f"{vor_komma}.{nach_komma}" if nach_komma else vor_komma
    
    # === Datum zusammensetzen ===
    tag_raw = "".join([d for d in d_wert[0:2] if d != "x"])
    monat_raw = "".join([d for d in d_wert[2:4] if d != "x"])
    datum_ok = ist_gueltiges_datum(tag_raw, monat_raw)
    
    if datum_ok:
        ablesedatum = f"{tag_raw.zfill(2)}.{monat_raw.zfill(2)}.{JAHR_FIX}"
        datum_quelle = "ocr"
    else:
        ablesedatum = extrahiere_scan_datum(basis)
        datum_quelle = "scan_fallback"

    # === Intelligente Gesamtbewertung ===
    overall_status = calculate_overall_status(z_boxes, d_boxes)
    needs_manual_review = overall_status in ["unsicher", "fehler"]
    
    # Statistiken
    stats["overall_status_counts"][overall_status] += 1
    
    if needs_manual_review:
        problematic_cards.append(basis)
    
    if overall_status != "ok":
        fehlerliste.append(basis)

    # === Detailliertes Ergebnis ===
    result = {
        "basis": basis,
        "status": overall_status,
        "needs_manual_review": needs_manual_review,
        "zaehlerstand_text": zaehler_text,
        "zaehlerstand_raw": z_wert,
        "ablesedatum": ablesedatum,
        "datum_quelle": datum_quelle,
        "z_kaestchen": z_boxes,
        "d_kaestchen": d_boxes,
        "problematic_boxes": problematic_boxes,
        "problematic_boxes_count": len(problematic_boxes),
        "average_confidence_z": round(np.mean(z_confidences), 3) if z_confidences else 0.0,
        "min_confidence_z": round(min(z_confidences), 3) if z_confidences else 0.0,
        "processing_timestamp": datetime.now().isoformat(),
        "_metadata": {
            "total_z_boxes": 9,
            "total_d_boxes": 4,
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "model_path": str(MODEL_PATH)
        }
    }
    
    return result

def main():
    """Hauptverarbeitung"""
    logger.info("🏭 ENHANCED KÄSTCHEN-OCR STARTER")
    logger.info("="*60)
    logger.info(f"📂 Z-Kästchen: {Z_DIR}")
    logger.info(f"📂 D-Kästchen: {D_DIR}")
    logger.info(f"📂 Output: {OUT_DIR}")
    logger.info(f"🎯 Schwellwerte: {CONFIDENCE_THRESHOLDS}")
    logger.info(f"📊 Zu verarbeiten: {len(basisnamen)} Karten")
    
    if not basisnamen:
        logger.warning("⚠️  Keine Karten zur Verarbeitung gefunden!")
        return
    
    start_time = datetime.now()
    
    # Verarbeite alle Karten
    for i, basis in enumerate(basisnamen, 1):
        logger.info(f"\n[{i}/{len(basisnamen)}] Verarbeite: {basis}")
        
        try:
            result = process_single_card(basis)
            
            # Speichere detaillierte JSON
            output_path = OUT_DIR / f"{basis}_kaestchen.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Konsolen-Output
            status_symbol = {
                "ok": "✅",
                "leicht_unsicher": "⚠️",
                "unsicher": "🔶", 
                "fehler": "❌"
            }
            
            symbol = status_symbol.get(result["status"], "❓")
            problematic_count = result["problematic_boxes_count"]
            avg_conf = result["average_confidence_z"]
            
            logger.info(f"   {symbol} {result['status'].upper():15} | "
                       f"Probleme: {problematic_count:2d}/13 | "
                       f"Avg-Conf: {avg_conf:.3f} | "
                       f"Zählerstand: {result['zaehlerstand_text']}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei {basis}: {e}")
            fehlerliste.append(basis)
            stats["overall_status_counts"]["fehler"] += 1
    
    # === FINALE STATISTIKEN ===
    duration = (datetime.now() - start_time).total_seconds()
    stats["total_boxes"] = stats["z_boxes"] + stats["d_boxes"]
    
    logger.info("\n" + "="*60)
    logger.info("📊 FINALE STATISTIKEN")
    logger.info("="*60)
    logger.info(f"🕐 Verarbeitungszeit: {duration:.1f} Sekunden")
    logger.info(f"📦 Verarbeitete Karten: {stats['total_cards']}")
    logger.info(f"🔢 Gesamte Kästchen: {stats['total_boxes']} ({stats['z_boxes']} Z + {stats['d_boxes']} D)")
    
    logger.info(f"\n🎯 KÄSTCHEN-LEVEL STATISTIKEN:")
    for status, count in stats["status_counts"].items():
        percentage = (count / stats["total_boxes"] * 100) if stats["total_boxes"] > 0 else 0
        logger.info(f"   {status:15}: {count:4d} ({percentage:5.1f}%)")
    
    logger.info(f"\n🃏 KARTEN-LEVEL STATISTIKEN:")
    for status, count in stats["overall_status_counts"].items():
        percentage = (count / stats["total_cards"] * 100) if stats["total_cards"] > 0 else 0
        logger.info(f"   {status:15}: {count:4d} ({percentage:5.1f}%)")
    
    manual_review_needed = len(problematic_cards)
    logger.info(f"\n🔧 MANUELLE ÜBERPRÜFUNG:")
    logger.info(f"   Benötigen Review: {manual_review_needed}/{stats['total_cards']} Karten")
    logger.info(f"   Automatisch OK: {stats['total_cards'] - manual_review_needed} Karten")
    
    # === Speichere Zusammenfassungen ===
    
    # Fehlerliste
    fehler_path = OUT_DIR / FEHLERLISTE_NAME
    with open(fehler_path, "w", encoding="utf-8") as f:
        json.dump(fehlerliste, f, indent=2, ensure_ascii=False)
    
    # Problematische Karten für GUI
    problematic_path = OUT_DIR / "problematic_cards.json"
    with open(problematic_path, "w", encoding="utf-8") as f:
        json.dump(problematic_cards, f, indent=2, ensure_ascii=False)
    
    # Gesamtstatistiken
    summary_path = OUT_DIR / f"ocr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "processing_duration_seconds": duration,
        "statistics": stats,
        "problematic_cards": problematic_cards,
        "problematic_cards_count": len(problematic_cards),
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "model_info": {
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists()
        }
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 DATEIEN GESPEICHERT:")
    logger.info(f"   📄 Fehlerliste: {fehler_path}")
    logger.info(f"   🔧 Problematische Karten: {problematic_path}")
    logger.info(f"   📊 Zusammenfassung: {summary_path}")
    
    logger.info("\n🎯 NÄCHSTE SCHRITTE:")
    logger.info(f"   1. Überprüfe {manual_review_needed} problematische Karten")
    logger.info(f"   2. Verwende GUI für manuelle Korrekturen")
    logger.info(f"   3. Sammle Training-Daten für Keras-Verbesserung")
    
    return summary

if __name__ == "__main__":
    try:
        summary = main()
        logger.info("✅ Enhanced OCR Kästchen-Analyse abgeschlossen!")
    except Exception as e:
        logger.error(f"💥 Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        exit(1)