#!/usr/bin/env python3
"""
Field Extractor Module für Zählerkarten-Pipeline
=====================================================
Extrahiert einzelne Felder aus den erkannten Bereichen für OCR-Verarbeitung.

Pipeline-Schritt 5: Nach Textfeld-Berechnung
Input: Textfelder JSONs mit Feldkoordinaten + Original-Bilder
Output: Einzelne PNG-Dateien für jedes Feld + erweiterte Metadaten

Version: 1.4 - Korrigiert für bbox-Format Kompatibilität
Datum: 09.07.2025
"""

import os
import cv2
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Logging Setup
logger = logging.getLogger(__name__)


class FieldExtractor:
    """
    Extrahiert einzelne Felder basierend auf erkannten Koordinaten
    
    Unterstützt:
    - Z-Kästchen (mit Rot-Entfernung)
    - D-Kästchen (mit Rot-Entfernung)  
    - Textfelder (berechnet)
    - QR-Code (neu hinzugefügt)
    - Zählerstand-Felder
    
    KOORDINATEN-FORMATE:
    - bbox-Format: {'bbox': {'x1': x, 'y1': y, 'x2': x, 'y2': y, 'width': w, 'height': h}}
    - center+size-Format: {'center': [x, y], 'size': [w, h]}
    - legacy x,y-Format: {'x': x, 'y': y, 'width': w, 'height': h}
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den Field Extractor
        
        Args:
            config_path: Pfad zum Config-Verzeichnis (optional)
        """
        # Lade Konfigurationen
        self.global_config, self.module_config = self._load_configs(config_path)
        
        # Setup Pfade
        self._setup_paths()
        
        # Setup Logging
        self._setup_logging()
        
        # Statistiken
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'extracted_fields': {
                'z_kaestchen': 0,
                'd_kaestchen': 0,
                'textfelder': 0,
                'qr_code': 0,
                'zaehlerstand': 0
            },
            'errors': []
        }
        
        logger.info(f"✅ Field Extractor V{self.module_config['module_version']} initialisiert")
        logger.info(f"   Aktivierte Extraktionen:")
        for field_type, config in self.module_config['extraction'].items():
            if config.get('enabled', False):
                logger.info(f"   ✓ {field_type} → {config['output_folder']}")
    
    def _load_configs(self, config_path: Optional[str]) -> Tuple[Dict, Dict]:
        """Lädt globale und modul-spezifische Konfigurationen"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "Config"
        else:
            config_path = Path(config_path)
        
        # Lade globale Config
        global_config_path = config_path / "pipeline_config.json"
        with open(global_config_path, 'r', encoding='utf-8') as f:
            global_config = json.load(f)
        
        # Lade Modul Config
        module_config_path = config_path / "field_extractor_config.json"
        with open(module_config_path, 'r', encoding='utf-8') as f:
            module_config = json.load(f)
        
        return global_config, module_config
    
    def _setup_paths(self):
        """Richtet alle notwendigen Pfade ein"""
        base_path = Path(self.global_config['base_path'])
        pipeline_data = base_path / self.global_config['paths']['pipeline_data']
        
        # Input Pfade
        self.input_metadata_path = pipeline_data / self.module_config['paths']['input_subdir']
        self.input_images_path = pipeline_data / self.module_config['paths']['input_images_subdir']
        
        # Output Pfade
        self.output_path = pipeline_data / self.module_config['paths']['output_subdir']
        self.output_metadata_path = pipeline_data / self.module_config['paths']['metadata_subdir']
        
        # Erstelle Ausgabeverzeichnisse für alle aktivierten Extraktionen
        for field_type, config in self.module_config['extraction'].items():
            if config.get('enabled', False):
                field_output_path = self.output_path / config['output_folder']
                field_output_path.mkdir(parents=True, exist_ok=True)
        
        # Debug-Pfad
        if self.module_config['debug']['enabled']:
            self.debug_path = base_path / self.global_config['paths']['debug_root'] / self.module_config['paths']['debug_subdir']
            self.debug_path.mkdir(parents=True, exist_ok=True)
            
            # Red removal debug
            self.red_debug_path = self.debug_path / "red_removal"
            self.red_debug_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata Output
        self.output_metadata_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Konfiguriert Logging für dieses Modul"""
        log_level = self.module_config['debug'].get('log_level', 'INFO')
        logging.getLogger().setLevel(getattr(logging, log_level))
    
    def _extract_field(self, image: np.ndarray, field_data: Dict, field_type: str, field_name: str) -> Optional[np.ndarray]:
        """
        Extrahiert ein einzelnes Feld aus dem Bild
        
        Args:
            image: Original-Bild
            field_data: Feld-Koordinaten (verschiedene Formate unterstützt)
            field_type: Typ des Feldes für Konfiguration
            field_name: Name für Ausgabedatei
            
        Returns:
            Extrahiertes Feld-Bild oder None bei Fehler
        """
        config = self.module_config['extraction'][field_type]
        
        # 🔧 KOORDINATEN EXTRAHIEREN - ALLE FORMATE UNTERSTÜTZT
        x, y, w, h = None, None, None, None
        
        if 'bbox' in field_data:
            # 🆕 NEUES BBOX-FORMAT (seit QR-ROI Update)
            bbox = field_data['bbox']
            x = int(bbox['x1'])
            y = int(bbox['y1'])
            w = int(bbox['width'])
            h = int(bbox['height'])
            logger.debug(f"📦 bbox-Format: {field_name} bei ({x},{y}) {w}x{h}")
            
        elif 'center' in field_data and 'size' in field_data:
            # 📐 TEXTFELD-FORMAT (center, size)
            center_x, center_y = field_data['center']
            width, height = field_data['size']
            x = int(center_x - width // 2)
            y = int(center_y - height // 2)
            w = int(width)
            h = int(height)
            logger.debug(f"🎯 center+size-Format: {field_name} bei ({x},{y}) {w}x{h}")
            
        elif 'x' in field_data and 'y' in field_data:
            # 🗂️ LEGACY X,Y-FORMAT (x, y, width, height)
            x = int(field_data['x'])
            y = int(field_data['y'])
            w = int(field_data['width'])
            h = int(field_data['height'])
            logger.debug(f"📋 legacy x,y-Format: {field_name} bei ({x},{y}) {w}x{h}")
            
        else:
            logger.warning(f"❌ Unbekanntes Koordinatenformat für {field_name}: {list(field_data.keys())}")
            return None
        
        # Margin hinzufügen
        margin = config.get('margin_pixels', 0)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + 2 * margin
        h = h + 2 * margin
        
        # Bild-Grenzen prüfen
        img_h, img_w = image.shape[:2]
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        if x >= x2 or y >= y2:
            logger.warning(f"❌ Ungültige Koordinaten für {field_name}: ({x},{y}) bis ({x2},{y2})")
            return None
        
        # Feld extrahieren
        field_image = image[y:y2, x:x2].copy()
        
        # Validierung
        processing_config = self.module_config['processing']
        if (field_image.shape[0] < processing_config['min_field_size'] or 
            field_image.shape[1] < processing_config['min_field_size']):
            logger.warning(f"⚠️ Feld {field_name} zu klein: {field_image.shape}")
            return None
        
        if (field_image.shape[0] > processing_config['max_field_size'] or 
            field_image.shape[1] > processing_config['max_field_size']):
            logger.warning(f"⚠️ Feld {field_name} zu groß: {field_image.shape}")
            return None
        
        # Rot-Entfernung falls konfiguriert
        if config.get('remove_red_border', False):
            field_image = self._remove_red_borders(field_image, field_name)
        
        return field_image
    
    def _extract_qr_code(self, image: np.ndarray, qr_data: Dict, image_name: str) -> Optional[str]:
        """
        Extrahiert QR-Code aus dem Bild (NEU!)
        
        Args:
            image: Original-Bild
            qr_data: QR-Code Koordinaten
            image_name: Name für Ausgabedatei
            
        Returns:
            Pfad zur gespeicherten QR-Code-Datei oder None
        """
        if not self.module_config['extraction']['qr_code']['enabled']:
            return None
        
        # QR-Code Feld extrahieren
        qr_image = self._extract_field(image, qr_data, 'qr_code', f"{image_name}_qr")
        if qr_image is None:
            return None
        
        # Speichern
        config = self.module_config['extraction']['qr_code']
        output_folder = self.output_path / config['output_folder']
        output_folder.mkdir(parents=True, exist_ok=True)
        
        file_name = f"{image_name}_qr.png"
        file_path = output_folder / file_name
        cv2.imwrite(str(file_path), qr_image)
        
        logger.info(f"   📱 QR-Code extrahiert: {file_name}")
        return str(file_path.relative_to(self.output_path))
    
    def _extract_textfelder(self, image: np.ndarray, textfelder_data: Dict, image_name: str) -> List[Dict]:
        """
        Extrahiert berechnete Textfelder
        
        Args:
            image: Original-Bild
            textfelder_data: Dictionary mit Textfeld-Daten
            image_name: Name für Ausgabedateien
            
        Returns:
            Liste der extrahierten Textfeld-Informationen
        """
        if not self.module_config['extraction']['textfelder']['enabled']:
            return []
        
        extracted_textfelder = []
        config = self.module_config['extraction']['textfelder']
        output_folder = self.output_path / config['output_folder']
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for field_name, field_data in textfelder_data.items():
            # Textfeld extrahieren
            textfield_image = self._extract_field(image, field_data, 'textfelder', f"{image_name}_{field_name}")
            if textfield_image is None:
                continue
            
            # Speichern
            file_name = f"{image_name}_{field_name}.png"
            file_path = output_folder / file_name
            cv2.imwrite(str(file_path), textfield_image)
            
            extracted_textfelder.append({
                'field_name': field_name,
                'file_path': str(file_path.relative_to(self.output_path)),
                'dimensions': [textfield_image.shape[1], textfield_image.shape[0]],
                'rotation': field_data.get('rotation', 0.0)
            })
            
            logger.info(f"   📝 Textfeld extrahiert: {field_name} → {file_name}")
        
        return extracted_textfelder
    
    def _extract_kaestchen(self, image: np.ndarray, kaestchen_data: List[Dict], field_type: str, image_name: str) -> List[Dict]:
        """
        Extrahiert Z- oder D-Kästchen
        
        Args:
            image: Original-Bild
            kaestchen_data: Liste der Kästchen-Daten
            field_type: 'z_kaestchen' oder 'd_kaestchen'
            image_name: Name für Ausgabedateien
            
        Returns:
            Liste der extrahierten Kästchen-Informationen
        """
        if not self.module_config['extraction'][field_type]['enabled']:
            return []
        
        extracted_kaestchen = []
        config = self.module_config['extraction'][field_type]
        output_folder = self.output_path / config['output_folder']
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for kaestchen in kaestchen_data:
            field_name = kaestchen.get('field_name', f"{field_type}_{len(extracted_kaestchen):02d}")
            
            # Kästchen extrahieren
            kaestchen_image = self._extract_field(image, kaestchen, field_type, f"{image_name}_{field_name}")
            if kaestchen_image is None:
                continue
            
            # Speichern
            file_name = f"{image_name}_{field_name}.png"
            file_path = output_folder / file_name
            cv2.imwrite(str(file_path), kaestchen_image)
            
            extracted_kaestchen.append({
                'field_name': field_name,
                'file_path': str(file_path.relative_to(self.output_path)),
                'dimensions': [kaestchen_image.shape[1], kaestchen_image.shape[0]],
                'confidence': kaestchen.get('confidence', 0.0),
                'index': kaestchen.get('index', len(extracted_kaestchen))
            })
            
            logger.info(f"   🔲 {field_type} extrahiert: {field_name} → {file_name}")
        
        return extracted_kaestchen
    
    def _extract_zaehlerstand(self, image: np.ndarray, data: Dict, image_name: str) -> Optional[str]:
        """
        Extrahiert Zählerstand-Feld basierend auf Z-Kästchen Position
        
        Args:
            image: Original-Bild
            data: Vollständige Detektionsdaten
            image_name: Name für Ausgabedatei
            
        Returns:
            Pfad zur gespeicherten Zählerstand-Datei oder None
        """
        if not self.module_config['extraction']['zaehlerstand']['enabled']:
            return None
        
        # Prüfe ob Z-Kästchen vorhanden
        detections = data.get('detections', {})
        z_fields = detections.get('z_kaestchen', [])
        
        if not z_fields:
            logger.warning(f"⚠️ Keine Z-Kästchen für Zählerstand-Extraktion gefunden: {image_name}")
            return None
        
        # Berechne Zählerstand-Bereich basierend auf Z-Kästchen
        # Alle Z-Kästchen koordinaten sammeln (ALLE FORMATE UNTERSTÜTZEN!)
        all_coords = []
        for field in z_fields:
            if 'bbox' in field:
                # BBOX-FORMAT (aktuell verwendet)
                bbox = field['bbox']
                all_coords.extend([
                    (bbox['x1'], bbox['y1']),
                    (bbox['x2'], bbox['y2'])
                ])
                logger.debug(f"🔍 Zählerstand Debug: bbox {field.get('field_name', 'unknown')} -> ({bbox['x1']},{bbox['y1']}) bis ({bbox['x2']},{bbox['y2']})")
            elif 'center' in field:
                # CENTER-FORMAT
                center = field['center']
                if isinstance(center, dict):
                    # {'center': {'x': x, 'y': y}}
                    center_x, center_y = center['x'], center['y']
                else:
                    # {'center': [x, y]}
                    center_x, center_y = center[0], center[1]
                    
                size = field.get('size', [100, 100])  # Default size
                all_coords.extend([
                    (center_x - size[0]//2, center_y - size[1]//2),
                    (center_x + size[0]//2, center_y + size[1]//2)
                ])
                logger.debug(f"🔍 Zählerstand Debug: center {field.get('field_name', 'unknown')} -> center({center_x},{center_y}) size{size}")
            elif 'x' in field and 'y' in field:
                # LEGACY X,Y-FORMAT
                all_coords.extend([
                    (field['x'], field['y']),
                    (field['x'] + field['width'], field['y'] + field['height'])
                ])
                logger.debug(f"🔍 Zählerstand Debug: x,y {field.get('field_name', 'unknown')} -> ({field['x']},{field['y']}) {field['width']}x{field['height']}")
            else:
                logger.warning(f"⚠️ Unbekanntes Z-Kästchen Format: {list(field.keys())}")
        
        if not all_coords:
            logger.warning(f"⚠️ Keine gültigen Koordinaten für Zählerstand gefunden: {image_name}")
            logger.warning(f"🔍 Z-Felder: {len(z_fields)} gefunden")
            if z_fields:
                logger.warning(f"🔍 Erstes Z-Feld keys: {list(z_fields[0].keys())}")
            return None
        
        logger.info(f"🔍 Zählerstand Debug: {len(all_coords)} Koordinaten gesammelt von {len(z_fields)} Z-Feldern")
        
        # Bounding Box aller Z-Kästchen berechnen
        min_x = min(coord[0] for coord in all_coords)
        max_x = max(coord[0] for coord in all_coords)
        min_y = min(coord[1] for coord in all_coords)
        max_y = max(coord[1] for coord in all_coords)
        
        logger.info(f"🔍 Zählerstand Bounding Box: ({min_x},{min_y}) bis ({max_x},{max_y}) = {max_x-min_x}x{max_y-min_y}")
        
        # Erweitere mit Margin
        config = self.module_config['extraction']['zaehlerstand']
        margin = config.get('margin_pixels', 50)
        
        # Zählerstand = Gesamtbereich aller Z-Kästchen für Kontrolle
        zaehlerstand_data = {
            'x': min_x - margin,
            'y': min_y - margin,
            'width': (max_x - min_x) + 2 * margin,
            'height': (max_y - min_y) + 2 * margin  # Umschließt alle Z-Kästchen
        }
        
        final_width = zaehlerstand_data['width']
        final_height = zaehlerstand_data['height']
        logger.info(f"🔍 Zählerstand Final: ({zaehlerstand_data['x']},{zaehlerstand_data['y']}) {final_width}x{final_height}")
        
        # Validiere Größe VORHER
        processing_config = self.module_config['processing']
        if (final_height > processing_config['max_field_size'] or 
            final_width > processing_config['max_field_size']):
            logger.warning(f"⚠️ Zählerstand zu groß: {final_width}x{final_height} > {processing_config['max_field_size']}")
            logger.warning(f"🔧 Reduziere Margin von {margin} auf {margin//2}")
            # Reduziere Margin automatisch
            margin = margin // 2
            zaehlerstand_data = {
                'x': min_x - margin,
                'y': min_y - margin,
                'width': (max_x - min_x) + 2 * margin,
                'height': (max_y - min_y) + 2 * margin
            }
            logger.info(f"🔧 Zählerstand Reduziert: {zaehlerstand_data['width']}x{zaehlerstand_data['height']}")
        
        # Extrahiere Zählerstand-Feld
        zaehlerstand_image = self._extract_field(image, zaehlerstand_data, 'zaehlerstand', f"{image_name}_zaehlerstand")
        if zaehlerstand_image is None:
            return None
        
        # Speichern
        output_folder = self.output_path / config['output_folder']
        output_folder.mkdir(parents=True, exist_ok=True)
        
        file_name = f"{image_name}_zaehlerstand.png"
        file_path = output_folder / file_name
        cv2.imwrite(str(file_path), zaehlerstand_image)
        
        logger.info(f"   🔢 Zählerstand extrahiert: {file_name}")
        return str(file_path.relative_to(self.output_path))
    
    def _remove_red_borders(self, image: np.ndarray, field_name: str) -> np.ndarray:
        """
        Entfernt rote Ränder aus dem Feld-Bild
        
        Args:
            image: Feld-Bild
            field_name: Name für Debug-Ausgabe
            
        Returns:
            Bild mit entfernten roten Rändern
        """
        config = self.module_config['red_border_removal']
        border_width = config['border_width']
        
        # Erstelle rote Pixel-Maske
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # HSV-Bereiche für Rot (berücksichtigt Wrap-Around bei Hue=0/180)
        mask1 = cv2.inRange(hsv, 
                           np.array(config['hsv_lower']), 
                           np.array(config['hsv_upper']))
        mask2 = cv2.inRange(hsv, 
                           np.array(config['hsv_lower_wrap']), 
                           np.array(config['hsv_upper_wrap']))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # RGB-basierte Rot-Erkennung als Backup
        b, g, r = cv2.split(image)
        
        # Erstelle RGB-Maske mit korrekten Datentypen
        red_condition1 = (r > config['rgb_red_threshold']).astype(np.uint8)
        red_condition2 = (r > g + config['rgb_red_dominance']).astype(np.uint8)
        red_condition3 = (r > b + config['rgb_red_dominance']).astype(np.uint8)
        
        rgb_red_mask = cv2.bitwise_and(red_condition1, 
                                      cv2.bitwise_and(red_condition2, red_condition3))
        rgb_red_mask = rgb_red_mask * 255  # Skaliere auf 0-255
        
        # Kombiniere Masken
        combined_mask = cv2.bitwise_or(red_mask, rgb_red_mask)
        
        # Entferne nur Ränder (nicht Innenbereiche)
        h, w = image.shape[:2]
        border_mask = np.zeros_like(combined_mask)
        
        # Ränder markieren
        border_mask[:border_width, :] = 255  # Oben
        border_mask[-border_width:, :] = 255  # Unten
        border_mask[:, :border_width] = 255  # Links
        border_mask[:, -border_width:] = 255  # Rechts
        
        # Nur rote Pixel in Randbereichen
        red_border_mask = cv2.bitwise_and(combined_mask, border_mask)
        
        # Debug-Ausgabe falls aktiviert
        if self.module_config['debug']['enabled'] and self.module_config['debug'].get('save_before_after', False):
            debug_before_after = np.hstack([image, cv2.cvtColor(red_border_mask, cv2.COLOR_GRAY2BGR)])
            debug_path = self.red_debug_path / f"{field_name}_comparison.png"
            cv2.imwrite(str(debug_path), debug_before_after)
            
            mask_path = self.red_debug_path / f"{field_name}_mask.png"
            cv2.imwrite(str(mask_path), red_border_mask)
        
        # Ersetze rote Ränder mit weißen Pixeln
        result = image.copy()
        result[red_border_mask > 0] = [255, 255, 255]
        
        return result
    
    def process_image(self, json_path: Path) -> Dict[str, Any]:
        """
        Verarbeitet ein einzelnes Bild und extrahiert alle Felder
        
        Args:
            json_path: Pfad zur Textfelder-JSON Datei
            
        Returns:
            Erweiterte Metadaten mit Extraktions-Informationen
        """
        start_time = datetime.now()
        image_name = json_path.stem.replace('_textfelder', '')
        
        logger.info(f"Extrahiere Felder aus: {image_name}")
        
        # Lade JSON-Daten
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {json_path.name}: {e}")
            self.stats['failed'] += 1
            return {}
        
        # Lade Original-Bild
        image_path = self.input_images_path / f"{image_name}.png"
        if not image_path.exists():
            logger.error(f"❌ Bild nicht gefunden: {image_path}")
            self.stats['failed'] += 1
            return data
        
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"❌ Bild konnte nicht geladen werden: {image_path}")
            self.stats['failed'] += 1
            return data
        
        # Extrahiere alle Felder
        extracted_files = {
            'z_kaestchen': [],
            'd_kaestchen': [],
            'textfelder': [],
            'qr_code': None,
            'zaehlerstand': None
        }
        
        detections = data.get('detections', {})
        
        # Extrahiere Z-Kästchen
        if 'z_kaestchen' in detections and detections['z_kaestchen']:
            extracted_files['z_kaestchen'] = self._extract_kaestchen(
                image, detections['z_kaestchen'], 'z_kaestchen', image_name)
            self.stats['extracted_fields']['z_kaestchen'] += len(extracted_files['z_kaestchen'])
        
        # Extrahiere D-Kästchen
        if 'd_kaestchen' in detections and detections['d_kaestchen']:
            extracted_files['d_kaestchen'] = self._extract_kaestchen(
                image, detections['d_kaestchen'], 'd_kaestchen', image_name)
            self.stats['extracted_fields']['d_kaestchen'] += len(extracted_files['d_kaestchen'])
        
        # Extrahiere QR-Code
        if 'qr_code' in detections and detections['qr_code']:
            extracted_files['qr_code'] = self._extract_qr_code(
                image, detections['qr_code'], image_name)
            if extracted_files['qr_code']:
                self.stats['extracted_fields']['qr_code'] += 1
        
        # Extrahiere Textfelder
        if 'textfelder' in detections and detections['textfelder']:
            extracted_files['textfelder'] = self._extract_textfelder(
                image, detections['textfelder'], image_name)
            self.stats['extracted_fields']['textfelder'] += len(extracted_files['textfelder'])
        
        # Extrahiere Zählerstand
        extracted_files['zaehlerstand'] = self._extract_zaehlerstand(
            image, data, image_name)
        if extracted_files['zaehlerstand']:
            self.stats['extracted_fields']['zaehlerstand'] += 1
        
        # Erstelle Debug-Visualisierung
        if self.module_config['debug']['enabled']:
            self._create_extraction_visualization(image, data, image_name)
        
        # Update Metadaten
        processing_time = (datetime.now() - start_time).total_seconds()
        data = self._create_extraction_metadata(extracted_files, data, processing_time)
        
        # Log Ergebnisse
        total_extracted = data['extraction']['statistics']['total_fields_extracted']
        logger.info(f"   ✅ {total_extracted} Felder extrahiert")
        
        self.stats['successful'] += 1
        return data
    
    def _create_extraction_visualization(self, image: np.ndarray, data: Dict, image_name: str):
        """Erstellt Debug-Visualisierung mit allen extrahierten Bereichen"""
        if not self.module_config['visualization']['enabled']:
            return
        
        vis_image = image.copy()
        viz_config = self.module_config['visualization']
        detections = data.get('detections', {})
        
        # Zeichne Z-Kästchen
        for z_field in detections.get('z_kaestchen', []):
            self._draw_field_box(vis_image, z_field, viz_config['box_color'], 
                               z_field.get('field_name', 'z'), viz_config)
        
        # Zeichne D-Kästchen
        for d_field in detections.get('d_kaestchen', []):
            self._draw_field_box(vis_image, d_field, (0, 255, 255), 
                               d_field.get('field_name', 'd'), viz_config)
        
        # Zeichne QR-Code
        if 'qr_code' in detections and detections['qr_code']:
            self._draw_field_box(vis_image, detections['qr_code'], (255, 0, 0), 
                               'QR', viz_config)
        
        # Zeichne Textfelder
        for field_name, field_data in detections.get('textfelder', {}).items():
            self._draw_field_box(vis_image, field_data, viz_config['text_color'], 
                               field_name, viz_config)
        
        # Zeichne Zählerstand-Bereich (approximiert)
        z_fields = detections.get('z_kaestchen', [])
        if z_fields:
            # Berechne Zählerstand-Box
            all_coords = []
            for field in z_fields:
                if 'bbox' in field:
                    bbox = field['bbox']
                    all_coords.extend([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])])
                elif 'x' in field:
                    all_coords.extend([(field['x'], field['y']), 
                                     (field['x'] + field['width'], field['y'] + field['height'])])
            
            if all_coords:
                min_x = min(coord[0] for coord in all_coords)
                max_x = max(coord[0] for coord in all_coords)
                min_y = min(coord[1] for coord in all_coords)
                max_y = max(coord[1] for coord in all_coords)
                
                margin = self.module_config['extraction']['zaehlerstand'].get('margin_pixels', 50)
                zaehler_x = min_x - margin
                zaehler_y = min_y - margin
                zaehler_w = (max_x - min_x) + 2 * margin  
                zaehler_h = (max_y - min_y) + 2 * margin  # Umschließt alle Z-Kästchen
                
                cv2.rectangle(vis_image, (zaehler_x, zaehler_y), 
                             (zaehler_x + zaehler_w, zaehler_y + zaehler_h), 
                             (255, 165, 0), viz_config['line_thickness'])
                if viz_config['show_field_names']:
                    cv2.putText(vis_image, "Zaehlerstand", (zaehler_x, zaehler_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, viz_config['font_scale'], (255, 165, 0), 1)
        
        # Speichern
        output_path = self.debug_path / f"{image_name}_extraction_debug.png"
        cv2.imwrite(str(output_path), vis_image)
        logger.debug(f"Extraction Debug gespeichert: {output_path.name}")
    
    def _draw_field_box(self, image: np.ndarray, field_data: Dict, color: Tuple[int, int, int], label: str, viz_config: Dict):
        """Zeichnet eine Bounding Box für ein Feld"""
        try:
            # Koordinaten extrahieren
            if 'bbox' in field_data:
                bbox = field_data['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            elif 'center' in field_data and 'size' in field_data:
                center_x, center_y = field_data['center']
                width, height = field_data['size']
                x1 = int(center_x - width // 2)
                y1 = int(center_y - height // 2)
                x2 = x1 + int(width)
                y2 = y1 + int(height)
            elif 'x' in field_data:
                x1, y1 = field_data['x'], field_data['y']
                x2 = x1 + field_data['width']
                y2 = y1 + field_data['height']
            else:
                return
            
            # Box zeichnen
            cv2.rectangle(image, (x1, y1), (x2, y2), color, viz_config['line_thickness'])
            
            # Label hinzufügen
            if viz_config['show_field_names']:
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, viz_config['font_scale'], color, 1)
        except Exception as e:
            logger.debug(f"Fehler beim Zeichnen von {label}: {e}")
    
    def _create_extraction_metadata(self, extracted_files: Dict, data: Dict, processing_time: float) -> Dict:
        """Erstellt erweiterte Metadaten für Extraktion"""
        
        # Zähle extrahierte Felder
        total_extracted = sum([
            len(extracted_files['z_kaestchen']),
            len(extracted_files['d_kaestchen']),
            len(extracted_files['textfelder']),
            1 if extracted_files['qr_code'] else 0,
            1 if extracted_files['zaehlerstand'] else 0
        ])
        
        extraction_stats = {
            'total_fields_extracted': total_extracted,
            'z_kaestchen_count': len(extracted_files['z_kaestchen']),
            'd_kaestchen_count': len(extracted_files['d_kaestchen']),
            'textfelder_count': len(extracted_files['textfelder']),
            'qr_code_extracted': bool(extracted_files['qr_code']),
            'zaehlerstand_extracted': bool(extracted_files['zaehlerstand'])
        }
        
        # Erweitere Metadaten
        if 'extraction' not in data:
            data['extraction'] = {}
        
        data['extraction'].update({
            'extracted_files': extracted_files,
            'extraction_timestamp': datetime.now().isoformat(),
            'extractor_version': self.module_config['module_version'],
            'processing_time': processing_time,
            'statistics': extraction_stats
        })
        
        # File Info erweitern
        if '_file_info' not in data:
            data['_file_info'] = {}
        
        data['_file_info'].update({
            'field_extraction_version': self.module_config['module_version'],
            'total_extracted_fields': total_extracted,
            'extraction_complete': total_extracted > 0
        })
        
        return data
    
    def process_all(self) -> Dict[str, Any]:
        """
        Verarbeitet alle Bilder im Input-Verzeichnis
        
        Returns:
            Zusammenfassung der Verarbeitung
        """
        logger.info("="*60)
        logger.info(f"FIELD EXTRACTOR - Stapelverarbeitung")
        logger.info("="*60)
        
        # Finde alle textfelder JSON Dateien
        json_files = sorted(self.input_metadata_path.glob("*_textfelder.json"))
        if not json_files:
            logger.warning(f"Keine textfelder JSON Dateien gefunden in {self.input_metadata_path}")
            return self._create_summary()
        
        logger.info(f"Gefunden: {len(json_files)} Dateien")
        
        # Debug-Zähler
        debug_count = 0
        max_debug = self.module_config['debug'].get('max_debug_images', 5)
        
        # Verarbeite jede Datei
        for i, json_path in enumerate(json_files, 1):
            logger.info(f"\n[{i}/{len(json_files)}] Verarbeite: {json_path.name}")
            
            try:
                # Verarbeite Bild
                result = self.process_image(json_path)
                
                if result:
                    # Speichere erweiterte JSON
                    output_json_path = self.output_metadata_path / f"{json_path.stem}_extracted.json"
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        if self.module_config['metadata']['pretty_print']:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        else:
                            json.dump(result, f, ensure_ascii=False)
                    
                    debug_count += 1
                
                self.stats['total_images'] += 1
                
            except Exception as e:
                logger.error(f"❌ Fehler bei {json_path.name}: {e}")
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'file': json_path.name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Zusammenfassung
        return self._create_summary()
    
    def _create_summary(self) -> Dict[str, Any]:
        """Erstellt Verarbeitungsstatistik"""
        total_processed = self.stats['successful'] + self.stats['failed']
        success_rate = (self.stats['successful'] / total_processed * 100) if total_processed > 0 else 0
        
        summary = {
            'module': self.module_config['module_name'],
            'version': self.module_config['module_version'],
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_images': self.stats['total_images'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': f"{success_rate:.1f}%",
                'extracted_fields': self.stats['extracted_fields'],
                'total_fields_extracted': sum(self.stats['extracted_fields'].values())
            },
            'errors': self.stats['errors']
        }
        
        # Log Zusammenfassung
        logger.info("\n" + "="*60)
        logger.info("FIELD EXTRACTOR - ZUSAMMENFASSUNG")
        logger.info("="*60)
        logger.info(f"✓ Verarbeitet: {self.stats['total_images']} Bilder")
        logger.info(f"✓ Erfolgreich: {self.stats['successful']} ({success_rate:.1f}%)")
        logger.info(f"✗ Fehlgeschlagen: {self.stats['failed']}")
        logger.info(f"📂 Extrahierte Felder:")
        for field_type, count in self.stats['extracted_fields'].items():
            if count > 0:
                logger.info(f"   {field_type}: {count}")
        logger.info(f"📁 Gesamt extrahiert: {sum(self.stats['extracted_fields'].values())} Felder")
        logger.info("="*60)
        
        return summary


def main():
    """Hauptfunktion für direkten Aufruf"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Field Extractor für Zählerkarten')
    parser.add_argument('--config', type=str, help='Pfad zum Config-Verzeichnis')
    parser.add_argument('--single', type=str, help='Einzelne JSON-Datei verarbeiten')
    parser.add_argument('--debug', action='store_true', help='Debug-Modus aktivieren')
    
    args = parser.parse_args()
    
    # Logging konfigurieren
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # Initialisiere Extractor
        extractor = FieldExtractor(config_path=args.config)
        
        if args.single:
            # Einzeldatei verarbeiten
            json_path = Path(args.single)
            if not json_path.exists():
                logger.error(f"Datei nicht gefunden: {json_path}")
                return 1
            
            result = extractor.process_image(json_path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Stapelverarbeitung
            extractor.process_all()
        
        return 0
        
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

    # ===================== NEU: Sortierung Z/D korrekt von links nach rechts =====================
    z_boxes = [b for b in boxes if b['class_name'] == 'Z-Kaestchen']
    z_boxes = sorted(z_boxes, key=lambda b: (round(b['y'] / 50), b['x']))
    for i, b in enumerate(z_boxes):
        b['field_name'] = f"z_{i:02d}"

    d_boxes = [b for b in boxes if b['class_name'] == 'D-Kaestchen']
    d_boxes = sorted(d_boxes, key=lambda b: (round(b['y'] / 50), b['x']))
    for i, b in enumerate(d_boxes):
        b['field_name'] = f"d_{i:02d}"

    other_boxes = [b for b in boxes if b['class_name'] not in ['Z-Kaestchen', 'D-Kaestchen']]
    boxes = z_boxes + d_boxes + other_boxes
    # ==============================================================================

