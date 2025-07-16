#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4.6 Oliver Krispel
YOLO Detector Module - 5 Klassen Modell, 3 Klassen Nutzung
==========================================================
Nutzt das 5b-Klassen Modell (trainiert mit mehr Daten), 
erkennt aber nur:
- Klasse 0: Z-Kästchen (9 Stück)
- Klasse 1: D-Kästchen (4 Stück)
- Klasse 2: QR-Code (1 Stück)

Plus-Zeichen und Großes Z werden durch Template-Matching erkannt,
da diese feste Positionen haben und Template präziser ist.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from ultralytics import YOLO
import traceback

# LOGGING FIX: Setze Debug-Level für alle Ausgaben
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console output
    ]
)

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO-basierter Detektor für 5 Klassen auf Zählerkarten"""
    
    def __init__(self, config_path: str = None):
        """
        Initialisiert YOLO Detector mit Config-basierter Konfiguration
        
        Args:
            config_path: Pfad zum Config-Verzeichnis (optional)
        """
        # Lade Konfigurationen
        try:
            configs = self._load_configs(config_path)
            self.global_config = configs[0]
            self.module_config = configs[1]
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Konfiguration: {e}")
            raise
        
        # Setze Parameter aus Config
        self.confidence_threshold = self.module_config['yolo_settings']['confidence_threshold']
        self.iou_threshold = self.module_config['yolo_settings']['iou_threshold']
        self.device = self.module_config['yolo_settings']['device']
        
        # Klassen-Mapping aus Config
        self.class_mapping = {
            int(k): v.lower().replace('-', '_').replace(' ', '_') 
            for k, v in self.module_config['field_expectations']['class_mapping'].items()
        }
        
        # Erwartete Anzahl aus Config (nur für Z, D und QR)
        self.expected_counts = {
            'z_kaestchen': self.module_config['field_expectations']['expected_z_count'],
            'd_kaestchen': self.module_config['field_expectations']['expected_d_count'],
            'qr_code': self.module_config['field_expectations']['expected_qr_count']
            # Plus und Z werden durch Template-Matching erkannt
        }
        
        # QR-Code ROI Konfiguration - Sehr großzügig: rechter oberer Quadrant
        # Karte wird in 4 Bereiche geteilt, QR ist immer rechts oben
        self.qr_roi_config = {
            'center_x': 3897,    # 75% der Bildbreite (5196 * 0.75)
            'center_y': 814,     # 25% der Bildhöhe (3259 * 0.25)  
            'margin_x': 1299,    # 25% der Bildbreite (5196 * 0.25) - erfasst gesamten rechten oberen Quadrant
            'margin_y': 814,     # 25% der Bildhöhe (3259 * 0.25) - erfasst gesamten rechten oberen Quadrant
            'min_confidence': 0.3,  # Niedrigere Schwelle für ROI-Suche
            'enabled': True,
            '_comment': 'ROI erfasst den kompletten rechten oberen Quadrant der Karte'
        }
        
        # Modell-Pfad aus Config
        base_path = Path(self.global_config['base_path'])
        models_dir = base_path / self.global_config['paths']['models_dir']
        self.model_path = models_dir / self.module_config['yolo_settings']['model_filename']
        
        # YOLO-Modell laden
        self.model = self._load_model()
        
        # Pfade setzen
        self._setup_paths()
        
        logger.info(f"✅ YOLO Detector V{self.module_config['module_version']} initialisiert")
        logger.info(f"   🔥 DEBUG VERSION AKTIV - ALLE AUSGABEN WERDEN ANGEZEIGT 🔥")
        logger.info(f"   Modell: {self.model_path.name}")
        logger.info(f"   Aktive Klassen: Z-Kästchen, D-Kästchen, QR-Code")
        logger.info(f"   QR-Code ROI: ({self.qr_roi_config['center_x']}, {self.qr_roi_config['center_y']}) ±{self.qr_roi_config['margin_x']}px")
        logger.info(f"   (Plus/Z durch Template-Matching)")
        logger.info(f"   Confidence: {self.confidence_threshold}")
        logger.info(f"   IoU: {self.iou_threshold}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Debug: {'ON' if self.module_config['debug']['enabled'] else 'OFF'}")
    
    def _load_configs(self, config_path: Optional[str]) -> Tuple[Dict, Dict]:
        """Lädt globale und modul-spezifische Konfigurationen"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "Config"
        else:
            config_path = Path(config_path)
        
        try:
            # Lade globale Config
            global_config_path = config_path / "pipeline_config.json"
            logger.debug(f"Lade globale Config: {global_config_path}")
            with open(global_config_path, 'r', encoding='utf-8') as f:
                global_config = json.load(f)
            
            # Lade Modul-Config
            module_config_path = config_path / "field_detector_config.json"
            logger.debug(f"Lade Modul-Config: {module_config_path}")
            with open(module_config_path, 'r', encoding='utf-8') as f:
                module_config = json.load(f)
            
            return global_config, module_config
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Config-Dateien: {e}")
            logger.error(f"   Config-Pfad: {config_path}")
            logger.error(f"   Globale Config: {global_config_path if 'global_config_path' in locals() else 'nicht gefunden'}")
            logger.error(f"   Modul Config: {module_config_path if 'module_config_path' in locals() else 'nicht gefunden'}")
            raise
    
    def _setup_paths(self):
        """Richtet alle notwendigen Pfade ein"""
        base_path = Path(self.global_config['base_path'])
        pipeline_data = base_path / self.global_config['paths']['pipeline_data']
        
        # Input/Output Pfade
        self.input_path = pipeline_data / self.module_config['paths']['input_subdir']
        self.metadata_path = pipeline_data / self.module_config['paths']['metadata_subdir']
        
        # Debug-Pfad
        if self.module_config['debug']['enabled']:
            self.debug_path = base_path / self.global_config['paths']['debug_root'] / self.module_config['paths']['debug_subdir']
            self.debug_path.mkdir(parents=True, exist_ok=True)
        
        # Erstelle Output-Verzeichnisse
        self.metadata_path.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> YOLO:
        """Lädt das YOLO-Modell"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO-Modell nicht gefunden: {self.model_path}")
        
        try:
            logger.debug(f"Lade YOLO-Modell: {self.model_path}")
            model = YOLO(str(self.model_path))
            logger.info(f"✅ YOLO-Modell geladen: {self.model_path.name}")
            return model
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des YOLO-Modells: {e}")
            raise
    
    def _is_qr_in_roi(self, detection: Dict) -> bool:
        """
        Prüft ob QR-Code in erwartetem ROI liegt
        
        Args:
            detection: QR-Code Detektion
            
        Returns:
            True wenn QR-Code im ROI liegt
        """
        if not self.qr_roi_config['enabled']:
            return True
        
        center_x = (detection['bbox']['x1'] + detection['bbox']['x2']) / 2
        center_y = (detection['bbox']['y1'] + detection['bbox']['y2']) / 2
        
        expected_x = self.qr_roi_config['center_x']
        expected_y = self.qr_roi_config['center_y']
        margin_x = self.qr_roi_config['margin_x']
        margin_y = self.qr_roi_config['margin_y']
        
        # Prüfe ob im ROI
        in_roi = (abs(center_x - expected_x) <= margin_x and 
                  abs(center_y - expected_y) <= margin_y)
        
        if not in_roi:
            logger.debug(f"QR-Code außerhalb ROI: ({center_x:.0f}, {center_y:.0f}) vs erwartet ({expected_x}, {expected_y}) ±{margin_x}")
        
        return in_roi
    
    def _get_qr_roi_box(self, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Berechnet QR-ROI Bounding Box
        
        Args:
            image_shape: (height, width) des Bildes
            
        Returns:
            (x1, y1, x2, y2) des ROI-Bereichs
        """
        # Handle beide Formate: (height, width) und (height, width, channels)
        if len(image_shape) == 2:
            height, width = image_shape
        elif len(image_shape) == 3:
            height, width, _ = image_shape
        else:
            raise ValueError(f"Unerwartete image_shape: {image_shape}")
        
        center_x = self.qr_roi_config['center_x']
        center_y = self.qr_roi_config['center_y']
        margin_x = self.qr_roi_config['margin_x']
        margin_y = self.qr_roi_config['margin_y']
        
        # ROI-Box berechnen mit Bildgrenzen-Clipping
        x1 = max(0, center_x - margin_x)
        y1 = max(0, center_y - margin_y)
        x2 = min(width, center_x + margin_x)
        y2 = min(height, center_y + margin_y)
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def _search_qr_in_roi(self, image: np.ndarray, image_name: str = "unknown") -> Optional[Dict]:
        """
        Sucht gezielt nach QR-Code im ROI-Bereich
        
        Args:
            image: Original-Bild
            image_name: Name des Bildes für Logging
            
        Returns:
            QR-Code Detektion oder None
        """
        logger.info(f"🔍 [{image_name}] Suche QR-Code gezielt im ROI-Bereich...")
        
        try:
            # ROI-Bereich extrahieren
            x1, y1, x2, y2 = self._get_qr_roi_box(image.shape)
            roi_image = image[y1:y2, x1:x2].copy()
            
            logger.debug(f"[{image_name}] ROI-Bereich: ({x1}, {y1}) bis ({x2}, {y2}), Größe: {x2-x1}x{y2-y1}")
            
            # YOLO auf ROI anwenden mit niedrigerer Confidence
            roi_results = self.model(
                roi_image,
                conf=self.qr_roi_config['min_confidence'],  # Niedrigere Schwelle
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Suche QR-Code in ROI-Ergebnissen
            for r in roi_results:
                if r.boxes is None:
                    continue
                
                for box in r.boxes:
                    try:
                        class_id = int(box.cls[0])
                        
                        # Nur QR-Codes (Klasse 2)
                        if class_id == 2:
                            confidence = float(box.conf[0])
                            
                            # FIXED: Bessere Bounding Box Extraktion
                            bbox_coords = box.xyxy[0].cpu().numpy()
                            logger.debug(f"[{image_name}] YOLO bbox_coords shape: {bbox_coords.shape}, values: {bbox_coords}")
                            
                            if len(bbox_coords) != 4:
                                logger.error(f"❌ [{image_name}] Unerwartete bbox_coords Anzahl: {len(bbox_coords)}, erwartet: 4")
                                continue
                            
                            roi_x1, roi_y1, roi_x2, roi_y2 = bbox_coords
                            
                            # Koordinaten von ROI zurück zu Original-Bild transformieren
                            orig_x1 = x1 + roi_x1
                            orig_y1 = y1 + roi_y1
                            orig_x2 = x1 + roi_x2
                            orig_y2 = y1 + roi_y2
                            
                            qr_detection = {
                                'class_id': class_id,
                                'class_name': self.class_mapping[class_id],
                                'confidence': confidence,
                                'bbox': {
                                    'x1': int(orig_x1),
                                    'y1': int(orig_y1),
                                    'x2': int(orig_x2),
                                    'y2': int(orig_y2),
                                    'width': int(orig_x2 - orig_x1),
                                    'height': int(orig_y2 - orig_y1)
                                },
                                'center': {
                                    'x': int((orig_x1 + orig_x2) / 2),
                                    'y': int((orig_y1 + orig_y2) / 2)
                                },
                                'source': 'roi_search'
                            }
                            
                            logger.info(f"✅ [{image_name}] QR-Code im ROI gefunden: ({qr_detection['center']['x']}, {qr_detection['center']['y']}) Conf: {confidence:.3f}")
                            return qr_detection
                            
                    except Exception as e:
                        logger.error(f"❌ [{image_name}] Fehler beim Extrahieren der Bounding Box: {e}")
                        logger.error(f"   box.xyxy[0] type: {type(box.xyxy[0])}")
                        logger.error(f"   box.xyxy[0] shape: {box.xyxy[0].shape if hasattr(box.xyxy[0], 'shape') else 'keine shape'}")
                        logger.error(f"   box.xyxy[0] content: {box.xyxy[0] if hasattr(box.xyxy[0], 'shape') else 'nicht darstellbar'}")
                        continue
            
            logger.warning(f"❌ [{image_name}] Kein QR-Code im ROI-Bereich gefunden")
            return None
            
        except Exception as e:
            logger.error(f"❌ [{image_name}] Fehler bei ROI-Suche: {e}")
            logger.error(f"   Image shape: {image.shape}")
            logger.error(f"   ROI config: {self.qr_roi_config}")
            logger.error(traceback.format_exc())
            return None
    
    def _enhance_qr_detection(self, image: np.ndarray, initial_detections: List[Dict], image_name: str = "unknown") -> Optional[Dict]:
        """
        Verbesserte QR-Code Erkennung mit ROI-Validierung und Fallback-Suche
        
        Args:
            image: Original-Bild
            initial_detections: Initial gefundene Detektionen
            image_name: Name des Bildes für Logging
            
        Returns:
            Beste QR-Code Detektion oder None
        """
        # DEBUG: Zeige alle initialen Detektionen
        logger.info(f"🔍 [{image_name}] DEBUG: {len(initial_detections)} initiale Detektionen gefunden")
        
        # Sammle alle QR-Code Detektionen
        qr_detections = [det for det in initial_detections if det['class_name'] == 'qr_code']
        
        logger.info(f"🔍 [{image_name}] DEBUG: {len(qr_detections)} QR-Code Detektionen gefunden")
        for i, qr_det in enumerate(qr_detections):
            logger.info(f"   DEBUG QR {i+1}: Position ({qr_det['center']['x']}, {qr_det['center']['y']}) Conf: {qr_det['confidence']:.3f}")
        
        if not qr_detections:
            logger.info(f"❌ [{image_name}] Keine QR-Codes in initialer Suche gefunden")
            logger.info(f"🔍 [{image_name}] DEBUG: Starte ROI-Suche (Grund: Keine QR-Codes gefunden)")
            # Fallback: Gezielt im ROI suchen
            roi_qr = self._search_qr_in_roi(image, image_name)
            return roi_qr
        
        # Filtere QR-Codes im ROI
        valid_qr_detections = []
        invalid_qr_detections = []
        
        logger.info(f"🔍 [{image_name}] DEBUG: ROI-Prüfung für {len(qr_detections)} QR-Codes")
        logger.info(f"   DEBUG ROI-Config: center=({self.qr_roi_config['center_x']}, {self.qr_roi_config['center_y']}) margin=({self.qr_roi_config['margin_x']}, {self.qr_roi_config['margin_y']})")
        
        for i, qr_det in enumerate(qr_detections):
            is_in_roi = self._is_qr_in_roi(qr_det)
            logger.info(f"   DEBUG QR {i+1}: ({qr_det['center']['x']}, {qr_det['center']['y']}) → {'IM ROI' if is_in_roi else 'AUSSERHALB ROI'}")
            
            if is_in_roi:
                valid_qr_detections.append(qr_det)
                logger.info(f"✅ [{image_name}] QR-Code im ROI: ({qr_det['center']['x']}, {qr_det['center']['y']}) Conf: {qr_det['confidence']:.3f}")
            else:
                invalid_qr_detections.append(qr_det)
                logger.warning(f"⚠️ [{image_name}] QR-Code außerhalb ROI: ({qr_det['center']['x']}, {qr_det['center']['y']}) Conf: {qr_det['confidence']:.3f}")
        
        logger.info(f"🔍 [{image_name}] DEBUG: {len(valid_qr_detections)} gültige, {len(invalid_qr_detections)} ungültige QR-Codes")
        
        # Wähle besten gültigen QR-Code
        if valid_qr_detections:
            # Sortiere nach Confidence und wähle besten
            best_qr = max(valid_qr_detections, key=lambda x: x['confidence'])
            best_qr['source'] = 'roi_validated'
            logger.info(f"✅ [{image_name}] DEBUG: Bester gültiger QR-Code gewählt: ({best_qr['center']['x']}, {best_qr['center']['y']}) Conf: {best_qr['confidence']:.3f}")
            return best_qr
        
        # Kein gültiger QR-Code gefunden - versuche ROI-Suche
        logger.warning(f"⚠️ [{image_name}] {len(invalid_qr_detections)} QR-Code(s) außerhalb ROI gefunden - starte ROI-Suche")
        logger.info(f"🔍 [{image_name}] DEBUG: Starte ROI-Suche (Grund: Alle QR-Codes außerhalb ROI)")
        roi_qr = self._search_qr_in_roi(image, image_name)
        
        if roi_qr:
            logger.info(f"✅ [{image_name}] DEBUG: ROI-Suche erfolgreich: ({roi_qr['center']['x']}, {roi_qr['center']['y']}) Conf: {roi_qr['confidence']:.3f}")
            return roi_qr
        
        logger.warning(f"❌ [{image_name}] DEBUG: ROI-Suche fehlgeschlagen")
        
        # Als letzter Ausweg: Nehme besten außerhalb ROI (aber logge Warnung)
        if invalid_qr_detections:
            best_invalid = max(invalid_qr_detections, key=lambda x: x['confidence'])
            best_invalid['source'] = 'outside_roi_fallback'
            logger.warning(f"⚠️ [{image_name}] DEBUG: Verwende QR-Code außerhalb ROI als Fallback: ({best_invalid['center']['x']}, {best_invalid['center']['y']}) Conf: {best_invalid['confidence']:.3f}")
            return best_invalid
        
        logger.error(f"❌ [{image_name}] DEBUG: Kein QR-Code gefunden!")
        return None
    
    def detect(self, image: np.ndarray, image_name: str = "unknown") -> Dict[str, Any]:
        """
        Hauptfunktion für YOLO-Erkennung mit verbesserter QR-Code Suche
        
        Args:
            image: Input-Bild als numpy array
            image_name: Name des Bildes für Logging
            
        Returns:
            Dictionary mit strukturierten Erkennungsergebnissen
        """
        height, width = image.shape[:2]
        logger.info(f"🔍 [{image_name}] Starte YOLO-Erkennung auf Bild {width}x{height}")
        
        try:
            # YOLO Inference
            results = self.model(
                image, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Extrahiere alle Detektionen
            initial_detections = self._extract_detections(results)
            
            # Verbesserte QR-Code Erkennung
            enhanced_qr = self._enhance_qr_detection(image, initial_detections, image_name)
            
            # Strukturiere Ergebnisse
            structured_results = self._structure_by_class(initial_detections, enhanced_qr)
            
            # Füge Statistiken hinzu
            structured_results['statistics'] = self._calculate_statistics(structured_results)
            structured_results['image_size'] = {'width': width, 'height': height}
            
            # Füge QR-ROI Info hinzu
            if enhanced_qr and 'source' in enhanced_qr:
                structured_results['qr_detection_info'] = {
                    'source': enhanced_qr['source'],
                    'roi_config': self.qr_roi_config
                }
            
            self._log_results(structured_results)
            
            return structured_results
            
        except Exception as e:
            logger.error(f"❌ YOLO-Erkennung fehlgeschlagen: {e}")
            logger.error(f"   Bild-Info: {width}x{height}, dtype: {image.dtype}")
            logger.error(f"   Model path: {self.model_path}")
            logger.error(f"   Confidence: {self.confidence_threshold}")
            logger.error(traceback.format_exc())
            raise
    
    def _extract_detections(self, results) -> List[Dict]:
        """Extrahiert alle Detektionen aus YOLO-Ergebnissen"""
        detections = []
        
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                try:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Nur aktive Klassen (0, 1, 2)
                    if class_id not in [0, 1, 2]:
                        continue
                    
                    # Bounding Box extrahieren
                    bbox_coords = box.xyxy[0].cpu().numpy()
                    if len(bbox_coords) != 4:
                        logger.warning(f"Überspringe Box mit {len(bbox_coords)} Koordinaten")
                        continue
                    
                    x1, y1, x2, y2 = bbox_coords
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': self.class_mapping[class_id],
                        'confidence': confidence,
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        },
                        'center': {
                            'x': int((x1 + x2) / 2),
                            'y': int((y1 + y2) / 2)
                        }
                    }
                    
                    detections.append(detection)
                    
                except Exception as e:
                    logger.warning(f"Fehler beim Extrahieren einer Detektion: {e}")
                    continue
        
        logger.info(f"   Extrahiert: {len(detections)} Detektionen")
        return detections
    
    def _structure_by_class(self, initial_detections: List[Dict], enhanced_qr: Optional[Dict]) -> Dict:
        """Strukturiert Detektionen nach Klassen"""
        # Filtere nach Klassen
        z_detections = [det for det in initial_detections if det['class_name'] == 'z_kaestchen']
        d_detections = [det for det in initial_detections if det['class_name'] == 'd_kaestchen']
        
        # Sortiere Z-Kästchen
        z_sorted = self._sort_kaestchen(z_detections, 'z')
        
        # Sortiere D-Kästchen
        d_sorted = self._sort_kaestchen(d_detections, 'd')
        
        # Alle Detektionen für Debug
        all_detections = initial_detections.copy()
        if enhanced_qr:
            all_detections.append(enhanced_qr)
        
        return {
            'z_kaestchen': z_sorted,
            'd_kaestchen': d_sorted,
            'qr_code': enhanced_qr,
            'all_detections': all_detections
        }
    
    def _sort_kaestchen(self, detections: List[Dict], kaestchen_type: str) -> List[Dict]:
        """Sortiert Kästchen nach Position und weist Indizes zu"""
        if not detections:
            return []
        
        # Sortiere nach Y-Position (oben nach unten), dann X-Position (links nach rechts)
        sorted_detections = sorted(detections, key=lambda d: (d['center']['x'], d['center']['x']))
        
        # Weise Indizes zu
        for i, det in enumerate(sorted_detections):
            det['index'] = i
            det['field_name'] = f"{kaestchen_type}_{i:02d}"
            det['type'] = kaestchen_type
        
        return sorted_detections
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Berechnet Statistiken über die Erkennungen"""
        stats = {}
        
        for class_name, expected in self.expected_counts.items():
            if class_name in ['qr_code']:
                found = 1 if results[class_name] is not None else 0
            else:
                found = len(results[class_name])
            
            stats[class_name] = {
                'found': found,
                'expected': expected,
                'complete': found == expected,
                'missing': max(0, expected - found)
            }
        
        # Gesamt-Status
        stats['all_complete'] = all(s['complete'] for s in stats.values())
        stats['total_detections'] = len(results['all_detections'])
        
        return stats
    
    def _log_results(self, results: Dict):
        """Loggt die Erkennungsergebnisse"""
        stats = results['statistics']
        
        logger.info("📊 YOLO-Erkennungsergebnisse:")
        
        for class_name in ['z_kaestchen', 'd_kaestchen', 'qr_code']:
            if class_name in stats:
                stat = stats[class_name]
                status = "✅" if stat['complete'] else "⚠️"
                logger.info(f"   {status} {class_name}: {stat['found']}/{stat['expected']}")
        
        # QR-Code Quelle loggen
        if results.get('qr_code') and 'qr_detection_info' in results:
            source = results['qr_detection_info']['source']
            logger.info(f"   QR-Code Quelle: {source}")
        
        logger.info(f"   Gesamt: {stats['total_detections']} Detektionen")
        logger.info(f"   (Plus/Z durch Template-Matching)")
        
        if not stats['all_complete']:
            logger.warning("   ⚠️ Nicht alle erwarteten Objekte gefunden!")
    
    def process_all_images(self):
        """Verarbeitet alle Bilder aus dem Input-Verzeichnis"""
        logger.info("="*60)
        logger.info(f"🚀 Starte Batch-Verarbeitung")
        logger.info(f"   Input: {self.input_path}")
        logger.info(f"   Output: {self.metadata_path}")
        logger.info("="*60)
        
        # Finde alle PNG-Dateien
        png_files = sorted(self.input_path.glob("*.png"))
        
        if not png_files:
            logger.warning(f"Keine PNG-Dateien in {self.input_path} gefunden!")
            return
        
        logger.info(f"📁 Gefunden: {len(png_files)} Bilder")
        
        # Statistiken
        stats = {
            'total': len(png_files),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'qr_sources': {'roi_validated': 0, 'roi_search': 0, 'outside_roi_fallback': 0}
        }
        
        # Verarbeite jedes Bild
        for i, png_file in enumerate(png_files, 1):
            logger.info(f"\n[{i}/{len(png_files)}] Verarbeite: {png_file.name}")
            
            try:
                # Bild laden
                image = cv2.imread(str(png_file))
                if image is None:
                    raise ValueError(f"Kann Bild nicht laden: {png_file}")
                
                # YOLO-Erkennung
                results = self.detect(image, png_file.stem)
                
                # Speichere Metadata
                self._save_metadata(png_file.stem, results)
                
                # Speichere Debug-Visualisierung
                if self.module_config['debug']['enabled']:
                    self._save_visualization(png_file.stem, image, results)
                
                stats['processed'] += 1
                if results['statistics']['all_complete']:
                    stats['successful'] += 1
                
                # QR-Source Statistik
                if 'qr_detection_info' in results and results['qr_detection_info']['source'] in stats['qr_sources']:
                    stats['qr_sources'][results['qr_detection_info']['source']] += 1
                
            except Exception as e:
                logger.error(f"❌ Fehler bei {png_file.name}: {e}")
                logger.error(f"   Fehler-Details: {traceback.format_exc()}")
                stats['failed'] += 1
        
        # Zusammenfassung
        logger.info("\n" + "="*60)
        logger.info("📊 BATCH-VERARBEITUNG ABGESCHLOSSEN")
        logger.info(f"   Verarbeitet: {stats['processed']}/{stats['total']}")
        logger.info(f"   Erfolgreich: {stats['successful']}")
        logger.info(f"   Fehlgeschlagen: {stats['failed']}")
        
        # QR-Source Statistik
        if any(stats['qr_sources'].values()):
            logger.info("   QR-Code Quellen:")
            for source, count in stats['qr_sources'].items():
                if count > 0:
                    logger.info(f"     {source}: {count}")
        
        return stats
    
    def _save_metadata(self, image_name: str, results: Dict):
        """Speichert die Erkennungsergebnisse als JSON"""
        metadata = {
            'source_image': f"{image_name}.png",
            'processing_info': {
                'module': self.module_config['module_name'],
                'version': self.module_config['module_version'],
                'timestamp': datetime.now().isoformat(),
                'model_file': self.model_path.name,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'note': 'Plus-Zeichen und Z-Marker werden durch Template-Matching erkannt'
            },
            'detections': {
                'z_kaestchen': results['z_kaestchen'],
                'd_kaestchen': results['d_kaestchen'],
                'qr_code': results['qr_code']
            },
            'statistics': results['statistics'],
            'image_size': results['image_size']
        }
        
        # Füge QR-Info hinzu falls vorhanden
        if 'qr_detection_info' in results:
            metadata['qr_detection_info'] = results['qr_detection_info']
        
        # Speichere als JSON
        output_file = self.metadata_path / f"{image_name}_yolo.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   → Metadata gespeichert: {output_file.name}")
    
    def _save_visualization(self, image_name: str, image: np.ndarray, results: Dict):
        """Speichert Debug-Visualisierung"""
        vis_image = image.copy()
        colors = {
            'z_kaestchen': (0, 0, 255),      # Rot
            'd_kaestchen': (0, 255, 0),      # Grün
            'qr_code': (255, 0, 0)           # Blau
        }
        
        # Zeichne QR-ROI
        if self.qr_roi_config['enabled']:
            x1, y1, x2, y2 = self._get_qr_roi_box(vis_image.shape)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Gelb
            cv2.putText(vis_image, "QR-ROI", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Zeichne alle Detektionen
        for det in results['all_detections']:
            bbox = det['bbox']
            color = colors.get(det['class_name'], (128, 128, 128))
            
            cv2.rectangle(vis_image, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         color, 3)
            
            # Label mit Index
            label = f"{det['class_name']}"
            if det['class_name'] in ['z_kaestchen', 'd_kaestchen']:
                # Finde Index in strukturierten Ergebnissen
                structured_fields = results[det['class_name']]
                for field in structured_fields:
                    if (field['center']['x'] == det['center']['x'] and field['center']['y'] == det['center']['y']):
                        if 'index' in field:
                            label += f"_{field['index']:02d}"
                        break
            
            # Confidence hinzufügen
            label += f" ({det['confidence']:.2f})"
            
            cv2.putText(vis_image, label, 
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # QR-Code Quelle anzeigen
        if results.get('qr_code') and 'qr_detection_info' in results:
            source = results['qr_detection_info']['source']
            qr = results['qr_code']
            source_text = f"QR: {source}"
            cv2.putText(vis_image, source_text,
                       (qr['bbox']['x1'], qr['bbox']['y1'] + qr['bbox']['height'] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Speichere Debug-Bild
        output_file = self.debug_path / f"{image_name}_yolo_debug.png"
        cv2.imwrite(str(output_file), vis_image)
        logger.info(f"   → Debug-Bild gespeichert: {output_file.name}")


def demo():
    """Demo-Funktion für einzelnes Bild"""
    print("🔍 YOLO Detector - Demo Modus")
    
    try:
        detector = YOLODetector()
        
        # Suche nach einem Testbild
        test_image_path = Path("C:/ZaehlerkartenV2/pipeline_data/01_converted").glob("*.png")
        test_image_path = next(test_image_path, None)
        
        if not test_image_path or not test_image_path.exists():
            print("❌ Kein Testbild gefunden")
            return
        
        print(f"✅ Verwende Testbild: {test_image_path}")
        
        # Bild laden
        image = cv2.imread(str(test_image_path))
        print(f"✅ Bild geladen: {image.shape}")
        
        # Erkennung durchführen
        results = detector.detect(image, test_image_path.stem)
        
        print("\n📊 Erkennungs-Ergebnisse:")
        for class_name, stats in results['statistics'].items():
            if isinstance(stats, dict) and 'found' in stats:
                print(f"   {class_name}: {stats['found']}/{stats['expected']}")
        
        if 'qr_detection_info' in results:
            print(f"   QR-Code Quelle: {results['qr_detection_info']['source']}")
        
    except Exception as e:
        print(f"❌ Demo-Fehler: {e}")
        traceback.print_exc()


def main():
    """Hauptfunktion für Batch-Verarbeitung"""
    print("🔥 DEBUG: main() Funktion gestartet - Logging-Level gesetzt!")
    print("🔍 YOLO Field Detector - Batch Processing mit QR-ROI")
    print("="*60)
    
    try:
        # Initialisiere Detector
        print("🔥 DEBUG: Initialisiere YOLODetector...")
        detector = YOLODetector()
        print("🔥 DEBUG: YOLODetector initialisiert!")
        
        # Verarbeite alle Bilder
        print("🔥 DEBUG: Starte process_all_images()...")
        stats = detector.process_all_images()
        
        print("\n✅ Verarbeitung abgeschlossen!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        print(f"Fehler-Details:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Demo-Modus für einzelnes Bild
        demo()
    else:
        # Standard: Batch-Verarbeitung
        exit(main())