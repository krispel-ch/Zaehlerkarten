#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template Matcher Module V1.0
============================
Findet die festen Marker (3x Plus-Zeichen und großes Z) mittels Template-Matching.
Exakte Implementierung basierend auf annotation_tool_v2_1.py

Workflow:
1. Lädt YOLO-Ergebnisse aus 02_yolo/metadata/
2. Führt Template-Matching für Plus und Z durch
3. Kombiniert Ergebnisse
4. Speichert nach 03_template/metadata/
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TemplateMatcher:
    """Template-basierte Erkennung für fixe Marker auf Zählerkarten"""
    
    def __init__(self, config_path: str = None):
        """
        Initialisiert Template Matcher mit Config aus template_matcher_config.json
        
        Args:
            config_path: Pfad zum Config-Verzeichnis (optional)
        """
        self.start_time = datetime.now()
        
        # Lade Konfigurationen
        self.global_config, self.module_config = self._load_configs(config_path)
        
        # Template-Einstellungen aus Config
        self.template_config = self.module_config['template_matching']
        self.template_confidence = self.template_config['confidence_threshold']
        self.matching_method = getattr(cv2, self.template_config['matching_method'])
        
        # Templates-Verzeichnis
        self.templates_dir = Path(self.global_config['base_path']) / self.global_config['paths']['orientierung_dir']
        
        # Templates Container
        self.templates = {}
        
        # Pfade setzen
        self._setup_paths()
        
        # Logging setup
        self._setup_logging()
        
        # Templates laden
        self._load_templates()
        
        # Statistiken
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'complete': 0,
            'partial': 0,
            'failed': 0
        }
        
        logger.info(f"✅ Template Matcher V{self.module_config['module_version']} initialisiert")
        logger.info(f"   Templates geladen: {len(self.templates)}")
        logger.info(f"   Konfidenz-Schwelle: {self.template_confidence}")
        logger.info(f"   Matching-Methode: {self.template_config['matching_method']}")
    
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
        
        # Lade Template Matcher Config
        module_config_path = config_path / "template_matcher_config.json"
        with open(module_config_path, 'r', encoding='utf-8') as f:
            module_config = json.load(f)
        
        return global_config, module_config
    
    def _setup_paths(self):
        """Richtet alle notwendigen Pfade ein"""
        base_path = Path(self.global_config['base_path'])
        pipeline_data = base_path / self.global_config['paths']['pipeline_data']
        
        # Input Pfade
        self.input_images_path = pipeline_data / "01_converted"
        self.input_metadata_path = pipeline_data / self.module_config['paths']['input_subdir']
        
        # Output Pfade
        self.output_path = pipeline_data / self.module_config['paths']['output_subdir']
        self.metadata_path = pipeline_data / self.module_config['paths']['metadata_subdir']
        
        # Debug-Pfad
        self.debug_path = base_path / "temp" / self.module_config['paths']['debug_subdir']
        
        # Log-Pfad
        self.log_path = base_path / self.global_config['paths']['log_dir'] / "template_matcher"
        
        # Erstelle Ausgabeverzeichnisse
        for path in [self.output_path, self.metadata_path, self.debug_path, self.log_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Konfiguriert das Logging-System"""
        log_file = self.log_path / f"template_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.module_config['debug']['log_level']),
            format=self.global_config['logging']['format'],
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_templates(self):
        """Lädt die Marker-Templates genau wie in annotation_tool_v2_1.py"""
        for marker_name, template_info in self.template_config['templates'].items():
            filename = template_info['filename']
            template_path = self.templates_dir / filename
            
            if template_path.exists():
                # Lade als Graustufen (wie im annotation_tool)
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                
                if template is not None:
                    # Falls immer noch 3D mit einem Kanal, dann squeeze
                    if len(template.shape) == 3 and template.shape[2] == 1:
                        template = np.squeeze(template)
                    
                    # Sicherstellen dass es wirklich 2D ist
                    if len(template.shape) == 2:
                        self.templates[marker_name] = {
                            'template': template,
                            'class': template_info['class'],
                            'class_id': template_info['class_id'],
                            'roi': template_info['roi'],
                            'description': template_info.get('description', marker_name)
                        }
                        logger.info(f"✅ Template geladen: {filename} ({template.shape})")
                    else:
                        logger.warning(f"⚠️ Template {filename} hat unerwartete Form: {template.shape}")
                else:
                    logger.error(f"⚠️ Template konnte nicht geladen werden: {filename}")
            else:
                logger.error(f"❌ Template nicht gefunden: {template_path}")
    
    def process_all(self) -> Dict[str, Any]:
        """
        Verarbeitet alle Bilder mit YOLO-Ergebnissen
        
        Returns:
            Dictionary mit Verarbeitungsstatistiken
        """
        logger.info("="*60)
        logger.info("Starte Template-Matching für alle Bilder...")
        
        # Finde alle PNG-Dateien
        png_files = sorted(self.input_images_path.glob("*.png"))
        
        if not png_files:
            logger.warning(f"Keine PNG-Dateien in {self.input_images_path} gefunden!")
            return self._create_summary()
        
        logger.info(f"Gefunden: {len(png_files)} PNG-Dateien")
        self.stats['total_images'] = len(png_files)
        
        # Verarbeite jede Datei
        for i, png_path in enumerate(png_files, 1):
            logger.info(f"\n[{i}/{len(png_files)}] Verarbeite: {png_path.name}")
            
            try:
                # Prüfe ob bereits verarbeitet
                if self._is_already_processed(png_path) and self.module_config['processing']['skip_existing']:
                    logger.info(f"  → Übersprungen (bereits verarbeitet)")
                    continue
                
                # Bild laden
                image = cv2.imread(str(png_path))
                if image is None:
                    raise ValueError(f"Kann Bild nicht laden: {png_path}")
                
                # Template-Matching durchführen
                template_results = self._run_template_matching(image, png_path.stem)
                
                # Mit YOLO-Ergebnissen kombinieren
                combined_results = self._combine_with_yolo(png_path.stem, image, template_results)
                
                # Ergebnisse speichern
                self._save_results(png_path.stem, combined_results)
                
                # Debug-Visualisierung
                if self.module_config['debug']['save_detection_image']:
                    self._create_visualization(image, combined_results, png_path.stem)
                
                # Statistik aktualisieren
                self._update_statistics(combined_results)
                self.stats['processed'] += 1
                
            except Exception as e:
                logger.error(f"  ❌ Fehler bei {png_path.name}: {e}")
                self.stats['failed'] += 1
                import traceback
                logger.error(traceback.format_exc())
        
        # Zusammenfassung erstellen
        summary = self._create_summary()
        
        # Zusammenfassung speichern
        if self.module_config['debug']['create_summary_report']:
            self._save_summary(summary)
        
        return summary
    
    def _run_template_matching(self, image: np.ndarray, image_name: str) -> Dict[str, Any]:
        """
        Führt Template-Matching durch (exakt wie in annotation_tool_v2_1.py)
        
        Args:
            image: BGR Bild
            image_name: Name für Debug-Ausgabe
            
        Returns:
            Dictionary mit gefundenen Markern
        """
        # Bild in Graustufen konvertieren
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape
        
        found_markers = []
        
        # Für jeden Marker
        for marker_name, template_data in self.templates.items():
            template = template_data['template']
            roi = template_data['roi']
            roi_x, roi_y, roi_w, roi_h = roi
            
            # ROI-Bereich validieren (wie im annotation_tool)
            if roi_x < 0 or roi_y < 0 or roi_x + roi_w > img_w or roi_y + roi_h > img_h:
                logger.warning(f"⚠️ ROI für {marker_name} außerhalb des Bildes")
                continue
            
            # ROI extrahieren
            roi_image = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Template Matching (genau wie im annotation_tool)
            try:
                result = cv2.matchTemplate(roi_image, template, self.matching_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val >= self.template_confidence:
                    # Position im ROI gefunden
                    template_h, template_w = template.shape
                    
                    # Absolute Position berechnen
                    x1 = roi_x + max_loc[0]
                    y1 = roi_y + max_loc[1]
                    x2 = x1 + template_w
                    y2 = y1 + template_h
                    
                    # Marker-Daten erstellen
                    marker_data = {
                        'marker_name': marker_name,
                        'class': template_data['class'],
                        'class_id': template_data['class_id'],
                        'x': x1,
                        'y': y1,
                        'width': template_w,
                        'height': template_h,
                        'center_x': x1 + template_w // 2,
                        'center_y': y1 + template_h // 2,
                        'confidence': float(max_val),
                        'roi': roi,
                        'description': template_data['description']
                    }
                    
                    found_markers.append(marker_data)
                    logger.info(f"✅ {marker_name} gefunden bei [{x1}, {y1}] (Konfidenz: {max_val:.3f})")
                else:
                    logger.info(f"❌ {marker_name} nicht gefunden (Konfidenz: {max_val:.3f} < {self.template_confidence})")
                    
            except cv2.error as e:
                logger.error(f"❌ Template-Matching Fehler für {marker_name}: {e}")
        
        # Ergebnisse strukturieren
        return self._structure_template_results(found_markers)
    
    def _structure_template_results(self, markers: List[Dict]) -> Dict[str, Any]:
        """Strukturiert Template-Matching Ergebnisse"""
        result = {
            'plus_zeichen': [],
            'grosses_z': None,
            'all_markers': markers,
            'statistics': {}
        }
        
        # Marker nach Typ sortieren
        for marker in markers:
            if marker['class'] == 'plus':
                result['plus_zeichen'].append(marker)
            elif marker['class'] == 'z_marker':
                result['grosses_z'] = marker
        
        # Plus-Zeichen nach Position sortieren (links → rechts)
        result['plus_zeichen'] = sorted(result['plus_zeichen'], key=lambda x: x['x'])
        
        # Statistik
        result['statistics'] = {
            'plus_found': len(result['plus_zeichen']),
            'plus_expected': self.module_config['quality_control']['expected_markers']['plus'],
            'z_found': 1 if result['grosses_z'] else 0,
            'z_expected': self.module_config['quality_control']['expected_markers']['z_marker'],
            'total_found': len(markers),
            'all_complete': (
                len(result['plus_zeichen']) == self.module_config['quality_control']['expected_markers']['plus'] and
                result['grosses_z'] is not None
            )
        }
        
        return result
    
    def _combine_with_yolo(self, image_name: str, image: np.ndarray, template_results: Dict) -> Dict[str, Any]:
        """Kombiniert YOLO- und Template-Ergebnisse"""
        # Lade YOLO-Ergebnisse
        yolo_file = Path(self.input_metadata_path) / f"{image_name}_yolo.json"
        
        if yolo_file.exists():
            with open(yolo_file, 'r', encoding='utf-8') as f:
                yolo_data = json.load(f)
        else:
            logger.warning(f"YOLO-Ergebnisse nicht gefunden: {yolo_file}")
            yolo_data = {
                'detections': {'z_kaestchen': [], 'd_kaestchen': [], 'qr_code': None},
                'statistics': {},
                'image_size': {'width': image.shape[1], 'height': image.shape[0]}
            }
        
        # Kombiniere Ergebnisse
        combined = {
            '_file_info': {
                'source_image': f"{image_name}.png",
                'timestamp': datetime.now().isoformat(),
                'module_version': self.module_config['module_version'],
                'yolo_source': str(yolo_file) if yolo_file.exists() else None,
                'template_matcher_version': self.module_config['module_version']
            },
            'detections': {
                # YOLO-Ergebnisse
                'z_kaestchen': yolo_data.get('detections', {}).get('z_kaestchen', []),
                'd_kaestchen': yolo_data.get('detections', {}).get('d_kaestchen', []),
                'qr_code': yolo_data.get('detections', {}).get('qr_code', None),
                # Template-Ergebnisse
                'plus_zeichen': template_results['plus_zeichen'],
                'grosses_z': template_results['grosses_z']
            },
            'statistics': {
                # YOLO-Statistiken
                **yolo_data.get('statistics', {}),
                # Template-Statistiken
                'plus_zeichen': {
                    'found': template_results['statistics']['plus_found'],
                    'expected': template_results['statistics']['plus_expected'],
                    'complete': template_results['statistics']['plus_found'] == template_results['statistics']['plus_expected']
                },
                'grosses_z': {
                    'found': template_results['statistics']['z_found'],
                    'expected': template_results['statistics']['z_expected'],
                    'complete': template_results['statistics']['z_found'] == template_results['statistics']['z_expected']
                },
                'template_all_complete': template_results['statistics']['all_complete']
            },
            'image_size': yolo_data.get('image_size', {'width': image.shape[1], 'height': image.shape[0]}),
            'template_matching_info': {
                'confidence_threshold': self.template_confidence,
                'matching_method': self.template_config['matching_method'],
                'templates_used': list(self.templates.keys())
            }
        }
        
        return combined
    
    def _save_results(self, image_name: str, results: Dict):
        """Speichert die kombinierten Ergebnisse"""
        output_file = self.metadata_path / f"{image_name}_combined.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if self.module_config['metadata']['pretty_print']:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)
        
        logger.info(f"  → Ergebnisse gespeichert: {output_file.name}")
    
    def _create_visualization(self, image: np.ndarray, results: Dict, image_name: str):
        """Erstellt Debug-Visualisierung mit Markern und ROIs"""
        vis_image = image.copy()
        
        # Farben aus Config
        colors = self.module_config['visualization']['box_colors']
        
        # Template-Marker zeichnen
        for marker in results['detections'].get('plus_zeichen', []):
            self._draw_detection(vis_image, marker, colors['plus'], "Plus")
        
        if results['detections'].get('grosses_z'):
            self._draw_detection(vis_image, results['detections']['grosses_z'], colors['z_marker'], "Z")
        
        # ROI-Bereiche zeichnen (wenn aktiviert)
        if self.module_config['visualization']['draw_roi_areas']:
            self._draw_roi_areas(vis_image)
        
        # Statistik-Overlay
        self._draw_statistics(vis_image, results)
        
        # Speichern
        output_file = self.debug_path / f"{image_name}_template_debug.png"
        cv2.imwrite(str(output_file), vis_image)
        logger.info(f"  → Debug-Bild gespeichert: {output_file.name}")
    
    def _draw_detection(self, image: np.ndarray, detection: Dict, color: List[int], label_prefix: str):
        """Zeichnet eine einzelne Detektion"""
        x = detection['x']
        y = detection['y']
        w = detection['width']
        h = detection['height']
        
        # Box
        thickness = self.module_config['visualization']['box_thickness']
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Label
        if self.module_config['visualization']['show_labels']:
            label = f"{label_prefix}"
            if 'marker_name' in detection:
                label = detection['marker_name']
            
            if self.module_config['visualization']['show_confidence']:
                label += f" ({detection['confidence']:.2f})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.module_config['visualization']['font_scale']
            font_thickness = self.module_config['visualization']['font_thickness']
            
            # Text-Hintergrund
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y), color, -1)
            
            # Text
            cv2.putText(image, label, (x, y - 4), font, font_scale, (255, 255, 255), font_thickness)
    
    def _draw_roi_areas(self, image: np.ndarray):
        """Zeichnet ROI-Bereiche semi-transparent"""
        overlay = image.copy()
        roi_color = self.module_config['visualization']['roi_color']
        roi_thickness = self.module_config['visualization']['roi_thickness']
        
        for marker_name, template_data in self.templates.items():
            x, y, w, h = template_data['roi']
            cv2.rectangle(overlay, (x, y), (x + w, y + h), roi_color, roi_thickness)
            cv2.putText(overlay, f"ROI: {marker_name}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)
        
        # Alpha-Blending
        alpha = self.module_config['visualization']['roi_alpha']
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    def _draw_statistics(self, image: np.ndarray, results: Dict):
        """Zeichnet Statistik-Overlay"""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Template-Statistiken
        stats = results['statistics']
        
        # Plus-Zeichen
        plus_text = f"Plus-Zeichen: {stats['plus_zeichen']['found']}/{stats['plus_zeichen']['expected']}"
        color = (0, 255, 0) if stats['plus_zeichen']['complete'] else (0, 165, 255)
        cv2.putText(image, plus_text, (10, y_offset), font, 0.8, color, 2)
        y_offset += 30
        
        # Z-Marker
        z_text = f"Z-Marker: {stats['grosses_z']['found']}/{stats['grosses_z']['expected']}"
        color = (0, 255, 0) if stats['grosses_z']['complete'] else (0, 165, 255)
        cv2.putText(image, z_text, (10, y_offset), font, 0.8, color, 2)
        y_offset += 30
        
        # Template-Status
        template_status = "Template: ✓ Komplett" if stats.get('template_all_complete', False) else "Template: ✗ Unvollständig"
        color = (0, 255, 0) if stats.get('template_all_complete', False) else (0, 0, 255)
        cv2.putText(image, template_status, (10, y_offset), font, 0.8, color, 2)
    
    def _update_statistics(self, results: Dict):
        """Aktualisiert die Verarbeitungsstatistiken"""
        stats = results['statistics']
        
        if stats.get('template_all_complete', False):
            self.stats['complete'] += 1
        elif stats.get('plus_zeichen', {}).get('found', 0) > 0 or stats.get('grosses_z', {}).get('found', 0) > 0:
            self.stats['partial'] += 1
    
    def _is_already_processed(self, image_path: Path) -> bool:
        """Prüft ob Bild bereits verarbeitet wurde"""
        metadata_file = self.metadata_path / f"{image_path.stem}_combined.json"
        return metadata_file.exists()
    
    def _create_summary(self) -> Dict[str, Any]:
        """Erstellt Zusammenfassung der Verarbeitung"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            'module': self.module_config['module_name'],
            'version': self.module_config['module_version'],
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(duration, 2),
            'configuration': {
                'confidence_threshold': self.template_confidence,
                'matching_method': self.template_config['matching_method'],
                'templates_loaded': len(self.templates)
            },
            'statistics': {
                'total_images': self.stats['total_images'],
                'processed': self.stats['processed'],
                'complete': self.stats['complete'],
                'partial': self.stats['partial'],
                'failed': self.stats['failed'],
                'success_rate': round(self.stats['complete'] / max(self.stats['processed'], 1) * 100, 1)
            }
        }
        
        # Log Zusammenfassung
        logger.info("\n" + "="*60)
        logger.info("ZUSAMMENFASSUNG")
        logger.info("="*60)
        logger.info(f"Verarbeitet: {self.stats['processed']}/{self.stats['total_images']} Bilder")
        logger.info(f"Vollständig: {self.stats['complete']} ({summary['statistics']['success_rate']}%)")
        logger.info(f"Teilweise: {self.stats['partial']}")
        logger.info(f"Fehlgeschlagen: {self.stats['failed']}")
        logger.info(f"Dauer: {duration:.1f} Sekunden")
        
        return summary
    
    def _save_summary(self, summary: Dict):
        """Speichert die Zusammenfassung"""
        summary_file = self.metadata_path / f"template_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n→ Zusammenfassung gespeichert: {summary_file.name}")


def main():
    """Hauptfunktion für direkten Aufruf"""
    print("🎯 Template Matcher V1.0")
    print("="*60)
    
    try:
        # Initialisiere Matcher
        matcher = TemplateMatcher()
        
        # Verarbeite alle Bilder
        summary = matcher.process_all()
        
        print("\n✅ Template-Matching abgeschlossen!")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())