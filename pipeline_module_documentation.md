# 📚 ZÄHLERKARTEN OCR PIPELINE - MODUL-DOKUMENTATION

## 📋 **ÜBERSICHT:**
Vollständige Dokumentation aller Pipeline-Module, KI-Modelle und Support-Tools.

**Stand:** 15.07.2025  
**Pipeline-Version:** V2.0  
**Python:** 3.10.x  

---

## 🏗️ **PIPELINE-ARCHITEKTUR**

```
📁 Scanner/              ← PDF-Eingabe
    ↓
📦 Modul 1: PDF → PNG
    ↓
📦 Modul 2: YOLO Erkennung
    ↓  
📦 Modul 3: Template Matching
    ↓
📦 Modul 4: Textfeld-Berechnung
    ↓
📦 Modul 5: Feld-Extraktion
    ↓
📦 Modul 6: QR-Code Reader    📦 Modul 7: OCR Textfelder    📦 Modul 8: OCR Kästchen
    ↓                              ↓                            ↓
📦 Modul 9: Pipeline Status Checker
    ↓
📦 Modul 10: Korrektur-GUI ← → 🧠 Keras Training
    ↓
📦 Modul 11: Post-Korrektur Analyzer
    ↓
📊 Excel Export (geplant)
```

---

## 📦 **MODUL 1: PDF CONVERTER**

### **📍 Zweck:**
Konvertiert PDF-Dateien in hochauflösende PNG-Bilder für die weitere Verarbeitung.

### **📂 Dateien:**
- **Script:** `module/1pdf_converter.py`
- **Config:** `Config/pdf_to_png_config.json`

### **⚙️ Konfiguration:**
```json
{
  "input_subdir": "",                    // Scanner-Unterordner
  "output_subdir": "01_converted",       // PNG-Ausgabe
  "dpi": 300,                           // Auflösung
  "format": "PNG",                      // Ausgabeformat
  "thread_count": 4                     // Parallelverarbeitung
}
```

### **🔧 Funktionsweise:**
1. **PDF-Erkennung:** Scannt `Scanner/` nach PDF-Dateien
2. **Konvertierung:** Verwendet `pdf2image` + `poppler`
3. **Qualitätskontrolle:** DPI-Validierung, Größenprüfung
4. **Metadaten:** Speichert Konvertierungs-Info als JSON

### **📥 Input:**
- PDF-Dateien in `Scanner/`
- Unterstützt: Multi-Page PDFs

### **📤 Output:**
- PNG-Bilder: `pipeline_data/01_converted/`
- Metadaten: `pipeline_data/01_converted/metadata/`
- Debug-Bilder: `debug/01_pdf_converter/` (optional)

### **🎯 Erfolg-Kriterien:**
- 300 DPI Auflösung erreicht
- PNG-Datei < 5MB
- Keine Konvertierungsfehler

---

## 📦 **MODUL 2: YOLO DETECTOR**

### **📍 Zweck:**
Erkennt Bereiche auf Zählerkarten mit YOLO-KI: Z-Kästchen, D-Kästchen, QR-Codes.

### **📂 Dateien:**
- **Script:** `module/2yolo_detector.py`
- **Config:** `Config/field_detector_config.json`
- **KI-Modell:** `KI/kaestchen_detector_5bclass.pt`

### **🧠 KI-Modell Details:**
- **Typ:** YOLOv8 (Ultralytics)
- **Klassen:** 5 (nutzt nur 0-2)
  - Klasse 0: Z-Kästchen (9 Stück)
  - Klasse 1: D-Kästchen (4 Stück)  
  - Klasse 2: QR-Code (1 Stück)
- **Modell-Größe:** 6.0 MB
- **Training:** Custom Dataset mit Zählerkarten

### **⚙️ Konfiguration:**
```json
{
  "yolo_settings": {
    "model_filename": "kaestchen_detector_5bclass.pt",
    "active_classes": [0, 1, 2],
    "confidence_threshold": 0.2,
    "iou_threshold": 0.5,
    "device": "cpu"
  },
  "field_expectations": {
    "expected_z_count": 9,
    "expected_d_count": 4,
    "expected_qr_count": 1
  }
}
```

### **🔧 Funktionsweise:**
1. **YOLO-Inferenz:** Lädt PNG-Bild, führt Objekterkennung durch
2. **QR-ROI Optimierung:** Begrenzt QR-Suche auf rechten oberen Quadrant
3. **Validierung:** Prüft erwartete Anzahl erkannter Objekte
4. **Geometrie-Fallback:** Verwendet feste Koordinaten bei YOLO-Fehlern

### **📥 Input:**
- PNG-Bilder: `pipeline_data/01_converted/`
- YOLO-Modell: `KI/kaestchen_detector_5bclass.pt`

### **📤 Output:**
- Erkennungs-Metadaten: `pipeline_data/02_yolo/metadata/`
- Debug-Bilder: `debug/02_field_detector/` (mit Bounding Boxes)

### **🎯 Erfolg-Kriterien:**
- 9 Z-Kästchen erkannt
- 4 D-Kästchen erkannt
- 1 QR-Code erkannt (oder ROI definiert)

---

## 📦 **MODUL 3: TEMPLATE MATCHER**

### **📍 Zweck:**
Findet feste Marker (Plus-Zeichen, Großes Z) mittels Template-Matching für präzise Orientierung.

### **📂 Dateien:**
- **Script:** `module/3template_matcher.py`
- **Config:** `Config/template_matcher_config.json`
- **Templates:** `orientierung/*.png`

### **🖼️ Template-Dateien:**
- `links_unten.png` - Plus-Zeichen unten links
- `rechts_unten.png` - Plus-Zeichen unten rechts  
- `rechts_oben.png` - Plus-Zeichen oben rechts
- `z_links_oben.png` - Großes Z oben links

### **⚙️ Konfiguration:**
```json
{
  "template_matching": {
    "confidence_threshold": 0.45,
    "matching_method": "TM_CCOEFF_NORMED",
    "templates": {
      "links_unten": {
        "roi": [150, 1900, 350, 350],
        "class_id": 3
      }
    }
  }
}
```

### **🔧 Funktionsweise:**
1. **Template-Loading:** Lädt PNG-Templates aus `orientierung/`
2. **ROI-Matching:** Sucht Templates nur in definierten Bereichen
3. **Confidence-Bewertung:** Verwendet OpenCV Template-Matching
4. **Kombination:** Vereint YOLO + Template-Ergebnisse

### **📥 Input:**
- PNG-Bilder: `pipeline_data/01_converted/`
- YOLO-Metadaten: `pipeline_data/02_yolo/metadata/`
- Template-Bilder: `orientierung/`

### **📤 Output:**
- Erweiterte Metadaten: `pipeline_data/03_template/metadata/`
- Debug-Bilder: `debug/template_debug/` (mit Template-Matches)

### **🎯 Erfolg-Kriterien:**
- Mindestens 3 Plus-Zeichen gefunden
- Großes Z gefunden
- Alle Confidence > 0.45

---

## 📦 **MODUL 4: TEXTFELD CALCULATOR**

### **📍 Zweck:**
Berechnet Textfeld-Positionen (Zählernummer, Zählerart) basierend auf erkannten Markern.

### **📂 Dateien:**
- **Script:** `module/4textfeld_calculator.py`
- **Config:** `Config/textfeld_calculator_config.json`

### **⚙️ Konfiguration:**
```json
{
  "calculation": {
    "textfelder": {
      "zaehlernummer": {
        "reference_marker": "plus_links_unten",
        "offset_x": 180,
        "offset_y": -320,
        "width": 800,
        "height": 120
      },
      "zaehlerart": {
        "reference_marker": "plus_rechts_oben", 
        "offset_x": -200,
        "offset_y": 100,
        "width": 400,
        "height": 80
      }
    }
  }
}
```

### **🔧 Funktionsweise:**
1. **Referenz-Analyse:** Verwendet Plus-Zeichen als Bezugspunkte
2. **Offset-Berechnung:** Berechnet Textfeld-Koordinaten relativ zu Markern
3. **Geometrie-Validierung:** Prüft ob Textfelder im Bild-Bereich liegen
4. **Metadaten-Erweiterung:** Fügt Textfeld-Koordinaten zu JSON hinzu

### **📥 Input:**
- Template-Metadaten: `pipeline_data/03_template/metadata/`

### **📤 Output:**
- Textfeld-Metadaten: `pipeline_data/04_textfelder/metadata/`
- Debug-Bilder: `debug/04_textfeld_calculator/` (mit Textfeld-Markierungen)

### **🎯 Erfolg-Kriterien:**
- Alle Textfelder berechnet
- Koordinaten innerhalb Bildgrenzen
- Validierte Geometrie

---

## 📦 **MODUL 5: FIELD EXTRACTOR**

### **📍 Zweck:**
Extrahiert einzelne Bereiche (Kästchen, Textfelder, QR) als separate PNG-Dateien für OCR.

### **📂 Dateien:**
- **Script:** `module/5field_extractor.py`
- **Config:** `Config/field_extractor_config.json`

### **⚙️ Konfiguration:**
```json
{
  "extraction": {
    "z_kaestchen": {
      "enabled": true,
      "output_folder": "z_kaestchen",
      "apply_red_removal": true,
      "resize_to": [28, 28]
    },
    "d_kaestchen": {
      "enabled": true,
      "output_folder": "d_kaestchen", 
      "apply_red_removal": true,
      "resize_to": [28, 28]
    },
    "textfelder": {
      "enabled": true,
      "output_folder": "textfelder"
    },
    "qr_code": {
      "enabled": true,
      "output_folder": "QR"
    }
  }
}
```

### **🔧 Funktionsweise:**
1. **Koordinaten-Parsing:** Liest Bounding Boxes aus Metadaten
2. **Bild-Ausschnitt:** Extrahiert Bereiche aus Haupt-PNG
3. **Rot-Entfernung:** Entfernt rote Linien bei Kästchen (für bessere OCR)
4. **Normalisierung:** Skaliert Kästchen auf 28x28 Pixel (Keras-Input)
5. **Datei-Export:** Speichert als einzelne PNG-Dateien

### **📥 Input:**
- PNG-Bilder: `pipeline_data/01_converted/`
- Textfeld-Metadaten: `pipeline_data/04_textfelder/metadata/`

### **📤 Output:**
- Z-Kästchen: `pipeline_data/05_extracted_fields/z_kaestchen/`
- D-Kästchen: `pipeline_data/05_extracted_fields/d_kaestchen/`  
- Textfelder: `pipeline_data/05_extracted_fields/textfelder/`
- QR-Codes: `pipeline_data/05_extracted_fields/QR/`
- Zählerstände: `pipeline_data/05_extracted_fields/zaehlerstand/`

### **🎯 Erfolg-Kriterien:**
- Alle Felder erfolgreich extrahiert
- Kästchen auf 28x28 normalisiert
- Rot-Entfernung erfolgreich

---

## 📦 **MODUL 6: QR CODE READER**

### **📍 Zweck:**
Liest QR-Codes aus und extrahiert Zählernummer + Zählerart.

### **📂 Dateien:**
- **Script:** `module/6_qr_code_reader.py`
- **Config:** `Config/qr_code_reader_config.json`

### **⚙️ Konfiguration:**
```json
{
  "qr_reader_settings": {
    "library": "pyzbar",
    "timeout_seconds": 5,
    "multiple_attempts": true,
    "preprocessing_enabled": true
  }
}
```

### **🔧 Funktionsweise:**
1. **QR-Erkennung:** Verwendet `pyzbar` Library
2. **Preprocessing:** Kontrast-Verbesserung, Binarisierung
3. **Multi-Attempt:** Verschiedene Bildverarbeitungs-Methoden
4. **Data-Parsing:** Extrahiert strukturierte Daten aus QR-String

### **📥 Input:**
- QR-Bilder: `pipeline_data/05_extracted_fields/QR/`

### **📤 Output:**
- QR-Daten: `pipeline_data/06_data_QR/{basename}_qr.json`

### **🎯 Erfolg-Kriterien:**
- QR-Code erfolgreich gelesen
- Zählernummer extrahiert
- Zählerart extrahiert

---

## 📦 **MODUL 7: OCR TEXTFELDER**

### **📍 Zweck:**
Führt OCR auf Textfeldern durch (Zählernummer, Zählerart) mit Validierung.

### **📂 Dateien:**
- **Script:** `module/7ocr_textfelder.py`
- **Config:** `Config/ocr_text_config.json`

### **⚙️ Konfiguration:**
```json
{
  "zaehlernummer_regex": "^\\d{5,}$",
  "zaehlerart_erlaubt": {
    "gas": ["gas", "6as", "qas"],
    "trinkwasser": ["trinkwasser", "h2o", "wasser"],
    "strom_t1": ["strom t1", "stromt1"],
    "strom_t4": ["strom t4", "stromt4"]
  }
}
```

### **🔧 Funktionsweise:**
1. **OCR-Verarbeitung:** Verwendet Tesseract/PyOCR
2. **Text-Normalisierung:** Bereinigt OCR-Ergebnisse
3. **Validierung:** Prüft gegen Regex/Whitelist
4. **Fehlerliste:** Sammelt fehlgeschlagene Erkennungen

### **📥 Input:**
- Textfeld-Bilder: `pipeline_data/05_extracted_fields/textfelder/`

### **📤 Output:**
- Erfolgreiche OCR: `pipeline_data/07_data_textfelder/{basename}_zaehlerart.json`
- Fehlerliste: `pipeline_data/07_data_textfelder/FEHLT_OCR.json`

### **🎯 Erfolg-Kriterien:**
- Text erkannt und validiert
- Zählerart in Whitelist gefunden
- Zählernummer im korrekten Format

---

## 📦 **MODUL 8: OCR KÄSTCHEN**

### **📍 Zweck:**
Erkennt Ziffern in Z-/D-Kästchen mittels Keras-CNN-Modell mit Confidence-Bewertung.

### **📂 Dateien:**
- **Script:** `module/8ocr_kaestchen.py`
- **Config:** `Config/ocr_kaestchen_config.json`
- **KI-Modell:** `KI/ziffer_model.keras`

### **🧠 KI-Modell Details:**
- **Typ:** Convolutional Neural Network (CNN)
- **Input:** 28x28 Grayscale Images
- **Output:** 11 Klassen (0-9 + "leer")
- **Architektur:** Conv2D → MaxPool → Dense → Softmax
- **Training:** MNIST + Custom Zählerkarten-Daten
- **Modell-Größe:** 2.6 MB

### **⚙️ Konfiguration:**
```json
{
  "model_path": "C:/ZaehlerkartenV2/KI/ziffer_model.keras",
  "confidence_threshold": 0.9,
  "jahr_fix": "2025"
}
```

### **🔧 Funktionsweise:**
1. **Kästchen-Loading:** Lädt 28x28 PNG-Bilder
2. **Keras-Inferenz:** CNN-Vorhersage für jedes Kästchen
3. **Confidence-Bewertung:** 4-stufiges System (ok/leicht_unsicher/unsicher/fehler)
4. **Intelligente Gesamtbewertung:** Kombiniert einzelne Kästchen-Bewertungen
5. **Problematische Karten:** Sammelt Karten für manuelle Review

### **📥 Input:**
- Z-Kästchen: `pipeline_data/05_extracted_fields/z_kaestchen/`
- D-Kästchen: `pipeline_data/05_extracted_fields/d_kaestchen/`

### **📤 Output:**
- Erfolgreiche OCR: `pipeline_data/08_data_kaestchen/{basename}.json`
- Fehlerliste: `pipeline_data/08_data_kaestchen/FEHLT_KAESTCHEN.json`
- Problematische Karten: `pipeline_data/08_data_kaestchen/problematic_cards.json`

### **🎯 Erfolg-Kriterien:**
- Alle 13 Kästchen (9+4) erkannt
- Confidence-Schwellwerte erfüllt
- Plausible Zählerstände/Daten

---

## 📦 **MODUL 9: PIPELINE STATUS CHECKER**

### **📍 Zweck:**
Analysiert Pipeline-Vollständigkeit und identifiziert problematische Karten.

### **📂 Dateien:**
- **Script:** `module/9pipeline_status_checker.py`
- **Config:** Verwendet `pipeline_config.json`

### **🔧 Funktionsweise:**
1. **Vollständigkeits-Check:** Prüft alle Pipeline-Stufen
2. **Datenqualitäts-Analyse:** Bewertet QR/Text/Kästchen-Erfolg
3. **Prioritäts-Klassifizierung:** Kategorisiert Probleme nach Dringlichkeit
4. **Training-Potenzial:** Schätzt Verbesserungs-Möglichkeiten

### **📥 Input:**
- Alle Pipeline-Daten (Stufen 1-8)
- Scanner-PDFs für Referenz

### **📤 Output:**
- Pipeline-Analyse: `dynamic_pipeline_analysis.json`
- Status-Reports und Statistiken

### **🎯 Erfolg-Kriterien:**
- Vollständige Pipeline-Durchlauf-Analyse
- Problematische Karten identifiziert
- Handlungsempfehlungen generiert

---

## 📦 **MODUL 10: KORREKTUR-GUI**

### **📍 Zweck:**
Interaktives GUI zur manuellen Korrektur problematischer Kästchen-OCR-Ergebnisse.

### **📂 Dateien:**
- **Script:** `module/10korrektur_kaestchen.py`
- **Config:** `Config/korrektur_kaestchen_config.json`

### **⚙️ Konfiguration:**
```json
{
  "gui_settings": {
    "window_width": 900,
    "window_height": 700,
    "image_display_size": 220,
    "keyboard_navigation": true
  },
  "training_settings": {
    "enable_training_collection": true,
    "training_batch_size": 16,
    "training_epochs": 3
  }
}
```

### **🔧 Funktionsweise:**
1. **Problem-Loading:** Lädt problematische Kästchen aus Pipeline-Status
2. **GUI-Darstellung:** Zeigt Kästchen-Bild + OCR-Vorhersage
3. **Tastatur-Navigation:** Ziffern 0-9, 'x' für leer
4. **Training-Integration:** Sammelt Korrekturen für Keras-Training
5. **Pipeline-Update:** Schreibt korrigierte Daten zurück

### **📥 Input:**
- Problematische Karten: `pipeline_data/09_pipeline_status/correction_gui_data.json`
- Kästchen-Bilder: `pipeline_data/05_extracted_fields/`

### **📤 Output:**
- Korrigierte Daten: `pipeline_data/08_data_kaestchen/` (updated)
- Training-Daten: `KI/Piplinetraining/kaestchen/` + `json/`
- Korrektur-Statistiken

### **🎯 Erfolg-Kriterien:**
- Alle problematischen Kästchen bearbeitet
- Training-Daten für Keras gesammelt
- Pipeline-Daten aktualisiert

---

## 📦 **MODUL 11: POST-KORREKTUR ANALYZER**

### **📍 Zweck:**
Analysiert Pipeline-Status nach Korrekturen und bereitet finale Datensammlung vor.

### **📂 Dateien:**
- **Script:** `module/11post_korrektur_analyzer.py`
- **Config:** `Config/post_korrektur_analyzer_config.json`

### **⚙️ Konfiguration:**
```json
{
  "analysis_criteria": {
    "z_kaestchen": {
      "post_correction_thresholds": {
        "perfect": {"min_confidence": 0.95, "max_problematic_count": 0},
        "very_good": {"min_confidence": 0.85, "max_problematic_count": 1},
        "acceptable": {"min_confidence": 0.70, "max_problematic_count": 2}
      }
    }
  }
}
```

### **🔧 Funktionsweise:**
1. **Vollständigkeits-Analyse:** Prüft alle 213 Karten nach Korrekturen
2. **Qualitäts-Bewertung:** 5-stufiges System (perfekt → kritisch)
3. **Prioritäts-Listen:** Erstellt Listen für weitere manuelle Arbeit
4. **GUI-Vorbereitung:** Bereitet Daten für Ganze-Karte-Korrektur vor
5. **Export-Readiness:** Bewertet Bereitschaft für Excel-Export

### **📥 Input:**
- Alle Pipeline-Daten (nach Korrekturen)
- FEHLT_*.json Listen
- Korrektur-Ergebnisse

### **📤 Output:**
- Status-Reports: `pipeline_data/11_post_analysis/reports/`
- Prioritäts-Listen: `pipeline_data/11_post_analysis/priority_lists/`
- GUI-Daten: `pipeline_data/11_post_analysis/complete_data/`

### **🎯 Erfolg-Kriterien:**
- Realistische Qualitäts-Bewertung aller Karten
- Prioritäts-Listen für finale Korrekturen
- Export-Readiness Assessment

---

## 🧠 **KERAS TRAINING SYSTEM**

### **📍 Zweck:**
Kontinuierliche Verbesserung des Kästchen-OCR-Modells durch inkrementelles Training.

### **📂 Dateien:**
- **Training-Script:** `KI/train_incremental_keras.py`
- **Haupt-Modell:** `KI/ziffer_model.keras`
- **Training-Daten:** `KI/Piplinetraining/`

### **🧠 Modell-Architektur:**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax')  # 0-9 + "leer"
])
```

### **⚙️ Training-Konfiguration:**
```json
{
  "batch_size": 16,
  "epochs": 3,
  "learning_rate": 0.001,
  "validation_split": 0.2,
  "backup_before_training": true
}
```

### **🔧 Training-Prozess:**
1. **Daten-Loading:** Lädt Korrektur-Daten aus GUI
2. **Format-Konvertierung:** JSON → Keras-Format
3. **Model-Loading:** Lädt existierendes Modell
4. **Fine-Tuning:** Inkrementelles Training mit neuen Daten
5. **Model-Backup:** Automatische Sicherung vor Überschreibung
6. **Validation:** Prüft Modell-Performance

### **📥 Input:**
- **Korrektur-Bilder:** `KI/Piplinetraining/kaestchen/`
- **Label-JSONs:** `KI/Piplinetraining/json/`
- **Basis-Modell:** `KI/ziffer_model.keras`

### **📤 Output:**
- **Verbessertes Modell:** `KI/ziffer_model.keras` (updated)
- **Backup:** `KI/ziffer_model.backup_TIMESTAMP.keras`
- **Training-Log:** Console + GUI

### **🎯 Erfolg-Kriterien:**
- Training ohne Fehler abgeschlossen
- Modell-Performance verbessert
- Backup erfolgreich erstellt

---

## 🏗️ **SUPPORT-TOOLS**

### **START.py - Pipeline Control Center**
- **Zweck:** Zentrales GUI für Pipeline-Steuerung
- **Funktionen:** Aufräumen, Pipeline starten, Korrektur starten
- **Location:** `C:\ZaehlerkartenV2\start.py`

### **YOLO Model Management**
- **Model:** `KI/kaestchen_detector_5bclass.pt`
- **Training:** Custom Dataset mit Zählerkarten-Annotationen
- **Update:** Über `KI/V2training/` Tools

### **Template Management**
- **Templates:** `orientierung/*.png`
- **Purpose:** Feste Orientierungs-Marker
- **Update:** Manueller Austausch bei Layout-Änderungen

---

## 🎯 **PIPELINE-ERFOLG METRIKEN**

### **Quantitative Erfolgs-Kriterien:**
- **PDF→PNG:** 100% Konvertierung erfolgreich
- **YOLO:** >95% korrekte Objekterkennung
- **Template:** >90% Marker gefunden
- **QR-Codes:** >80% erfolgreich gelesen
- **Kästchen-OCR:** >85% korrekte Ziffern (vor Korrektur)
- **Nach Korrektur:** >98% korrekte Daten

### **Qualitative Erfolgs-Kriterien:**
- **Datenintegrität:** Keine verlorenen Karten
- **Reproduzierbarkeit:** Gleiche Ergebnisse bei Wiederholung
- **Skalierbarkeit:** Verarbeitung großer PDF-Mengen
- **Benutzerfreundlichkeit:** Intuitive Korrektur-GUIs

---

## 🔧 **WARTUNG & UPDATES**

### **Regelmäßige Wartung:**
1. **Keras-Training:** Nach 50-100 Korrekturen
2. **Template-Updates:** Bei Layout-Änderungen der Zählerkarten
3. **Config-Anpassung:** Schwellwerte optimieren
4. **YOLO-Retraining:** Bei neuen Karten-Typen

### **Performance-Monitoring:**
- **Pipeline-Durchlaufzeit:** Ziel <5min/PDF
- **OCR-Genauigkeit:** Kontinuierliches Tracking
- **Korrektur-Aufwand:** Minimierung durch Training
- **Export-Qualität:** 100% korrekte Excel-Daten

---

## 📚 **ZUSÄTZLICHE DOKUMENTATION**

- **NEUER_CHAT_DOKUMENTATION.md** - Vollständiger Projektstand
- **MANUELLE_INSTALLATION.md** - Setup-Anleitung
- **requirements.txt** - Python-Dependencies
- **PIPELINE_SETUP.bat** - Automatische Installation

**Die Pipeline ist ein robustes, modulares System für hochqualitative OCR-Verarbeitung von Zählerkarten!** 🚀