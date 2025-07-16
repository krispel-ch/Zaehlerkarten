#!/usr/bin/env python3
"""
Textfeld Calculator Modul für Zählerkarten-Pipeline
=================================================
Berechnet Textfeld-Positionen basierend auf Plus-Markern
mittels Barycentric Transform.

Version: 1.0
Autor: KI-System
Datum: 02.07.2025
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


class TextfeldCalculator:
    """
    Berechnet Textfeld-Positionen relativ zu Plus-Markern
    
    Pipeline-Schritt 4: Nach Template-Matching
    Input: Combined JSONs mit Plus-Marker Positionen
    Output: Erweiterte JSONs mit Textfeld-Koordinaten
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den Textfeld Calculator
        
        Args:
            config_path: Pfad zum Config-Verzeichnis (optional)
        """
        # Lade Konfigurationen
        self.global_config, self.module_config = self._load_configs(config_path)
        
        # Setup Pfade
        self._setup_paths()
        
        # Lade Berechnungsmodell
        self.calculation_model = self._load_calculation_model()
        
        # Statistiken
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'missing_markers': 0,
            'low_confidence': 0,
            'errors': []
        }
        
        logger.info(f"✅ Textfeld Calculator V{self.module_config['module_version']} initialisiert")
        logger.info(f"   Berechnungsmodell: {self.module_config['paths']['calculation_model']}")
        logger.info(f"   Textfelder: {', '.join(self.module_config['calculation']['textfelder'])}")
    
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
        module_config_path = config_path / "textfeld_calculator_config.json"
        with open(module_config_path, 'r', encoding='utf-8') as f:
            module_config = json.load(f)
        
        return global_config, module_config
    
    def _setup_paths(self):
        """Richtet alle notwendigen Pfade ein"""
        base_path = Path(self.global_config['base_path'])
        pipeline_data = base_path / self.global_config['paths']['pipeline_data']
        
        # Input/Output Pfade
        self.input_path = pipeline_data / self.module_config['paths']['input_subdir']
        self.output_path = pipeline_data / self.module_config['paths']['output_subdir']
        self.output_metadata_path = pipeline_data / self.module_config['paths']['metadata_subdir']
        
        # Debug Pfad
        if self.module_config['debug']['enabled']:
            self.debug_path = base_path / self.global_config['paths']['debug_root'] / self.module_config['paths']['debug_subdir']
            self.debug_path.mkdir(parents=True, exist_ok=True)
        
        # Erstelle Output-Verzeichnisse
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Berechnungsmodell Pfad
        self.model_path = base_path / self.module_config['paths']['calculation_model']
    
    def _load_calculation_model(self) -> Dict:
        """Lädt das trainierte Berechnungsmodell"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Berechnungsmodell nicht gefunden: {self.model_path}")
        
        with open(self.model_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
        
        logger.info(f"   Modell geladen: {model['model_info']['type']} v{model['model_info']['version']}")
        logger.info(f"   Training Samples: {model['model_info']['training_samples']}")
        
        return model
    
    def _calculate_transform_matrix(self, plus_markers: Dict[str, Dict]) -> Optional[np.ndarray]:
        """
        Berechnet die Transformationsmatrix von normalisierten zu Bildkoordinaten
        
        Args:
            plus_markers: Dictionary mit Plus-Marker Positionen
            
        Returns:
            3x2 Affine Transformationsmatrix oder None bei Fehler
        """
        try:
            # Extrahiere benötigte Marker
            required_markers = self.module_config['calculation']['reference_markers']
            
            # Prüfe ob alle Marker vorhanden
            for marker in required_markers:
                if marker not in plus_markers:
                    logger.warning(f"Fehlender Marker: {marker}")
                    return None
            
            # Quell-Punkte (normalisiertes Dreieck aus Modell)
            src_points = np.array([
                self.calculation_model['reference_system']['normalized_triangle']['links_unten'],
                self.calculation_model['reference_system']['normalized_triangle']['rechts_unten'],
                self.calculation_model['reference_system']['normalized_triangle']['rechts_oben']
            ], dtype=np.float32)
            
            # Ziel-Punkte (aktuelle Plus-Positionen im Bild)
            dst_points = np.array([
                [plus_markers['links_unten'].get('center_x', plus_markers['links_unten']['x']), 
                 plus_markers['links_unten'].get('center_y', plus_markers['links_unten']['y'])],
                [plus_markers['rechts_unten'].get('center_x', plus_markers['rechts_unten']['x']), 
                 plus_markers['rechts_unten'].get('center_y', plus_markers['rechts_unten']['y'])],
                [plus_markers['rechts_oben'].get('center_x', plus_markers['rechts_oben']['x']), 
                 plus_markers['rechts_oben'].get('center_y', plus_markers['rechts_oben']['y'])]
            ], dtype=np.float32)
            
            # Berechne affine Transformation
            transform_matrix = cv2.getAffineTransform(src_points, dst_points)
            
            return transform_matrix
            
        except Exception as e:
            logger.error(f"Fehler bei Transformationsberechnung: {e}")
            return None
    
    def _transform_textfeld_position(self, normalized_pos: Dict, transform_matrix: np.ndarray) -> Tuple[int, int]:
        """
        Transformiert normalisierte Position in Bildkoordinaten
        
        Args:
            normalized_pos: Normalisierte Position {'x': float, 'y': float}
            transform_matrix: Affine Transformationsmatrix
            
        Returns:
            (x, y) Bildkoordinaten
        """
        # Erstelle Punkt-Array
        point = np.array([[normalized_pos['x'], normalized_pos['y']]], dtype=np.float32)
        
        # Transformiere
        transformed = cv2.transform(point.reshape(-1, 1, 2), transform_matrix)
        
        # Extrahiere und runde Koordinaten
        x = int(round(transformed[0][0][0]))
        y = int(round(transformed[0][0][1]))
        
        return x, y
    
    def _calculate_rotation(self, plus_markers: Dict[str, Dict], relative_rotation: float) -> float:
        """
        Berechnet absolute Rotation basierend auf Basis-Linie
        
        Args:
            plus_markers: Plus-Marker Positionen
            relative_rotation: Relative Rotation aus Modell (in Radians)
            
        Returns:
            Absolute Rotation in Grad
        """
        # Berechne Winkel der Basis-Linie (links_unten → rechts_unten)
        lu = np.array([
            plus_markers['links_unten'].get('center_x', plus_markers['links_unten']['x']), 
            plus_markers['links_unten'].get('center_y', plus_markers['links_unten']['y'])
        ])
        ru = np.array([
            plus_markers['rechts_unten'].get('center_x', plus_markers['rechts_unten']['x']), 
            plus_markers['rechts_unten'].get('center_y', plus_markers['rechts_unten']['y'])
        ])
        
        base_vector = ru - lu
        base_angle = np.arctan2(base_vector[1], base_vector[0])
        
        # Absolute Rotation
        absolute_rotation = base_angle + relative_rotation
        
        # Konvertiere zu Grad
        return np.degrees(absolute_rotation)
    
    def _create_visualization(self, image_path: Path, data: Dict, output_path: Path):
        """Erstellt Debug-Visualisierung mit Markern und Textfeldern"""
        if not self.module_config['visualization']['enabled']:
            return
        
        # Lade Bild
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Konnte Bild nicht laden für Visualisierung: {image_path}")
            return
        
        viz_config = self.module_config['visualization']
        
        # Zeichne Plus-Marker
        if viz_config['draw_markers'] and 'plus_zeichen' in data['detections']:
            plus_data = data['detections']['plus_zeichen']
            
            # Konvertiere Liste zu Dictionary falls nötig
            if isinstance(plus_data, list):
                markers_dict = {}
                for marker in plus_data:
                    if 'marker_name' in marker:
                        markers_dict[marker['marker_name']] = marker
                plus_data = markers_dict
            
            # Zeichne Marker
            if isinstance(plus_data, dict):
                for marker_name, marker_data in plus_data.items():
                    center = (int(marker_data.get('center_x', marker_data['x'])), 
                             int(marker_data.get('center_y', marker_data['y'])))
                    cv2.circle(image, center, 10, viz_config['marker_color'], -1)
                    cv2.putText(image, marker_name, 
                               (center[0] + 15, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               viz_config['font_scale'], 
                               viz_config['marker_color'], 2)
        
        # Zeichne Referenz-Dreieck
        if viz_config['draw_reference_triangle'] and 'plus_zeichen' in data['detections']:
            plus_data = data['detections']['plus_zeichen']
            
            # Konvertiere Liste zu Dictionary falls nötig
            if isinstance(plus_data, list):
                markers_dict = {}
                for marker in plus_data:
                    if 'marker_name' in marker:
                        markers_dict[marker['marker_name']] = marker
                plus_data = markers_dict
            
            # Nur wenn Dictionary mit den benötigten Markern
            if isinstance(plus_data, dict):
                markers = plus_data
                if all(m in markers for m in ['links_unten', 'rechts_unten', 'rechts_oben']):
                    pts = np.array([
                        [markers['links_unten'].get('center_x', markers['links_unten']['x']), 
                         markers['links_unten'].get('center_y', markers['links_unten']['y'])],
                        [markers['rechts_unten'].get('center_x', markers['rechts_unten']['x']), 
                         markers['rechts_unten'].get('center_y', markers['rechts_unten']['y'])],
                        [markers['rechts_oben'].get('center_x', markers['rechts_oben']['x']), 
                         markers['rechts_oben'].get('center_y', markers['rechts_oben']['y'])]
                    ], np.int32)
                    cv2.polylines(image, [pts], True, viz_config['triangle_color'], viz_config['line_thickness'])
        
        # Zeichne Textfelder
        if viz_config['draw_textfelder'] and 'textfelder' in data['detections']:
            for field_name, field_data in data['detections']['textfelder'].items():
                # Rechteck zeichnen
                center = (int(field_data['center'][0]), int(field_data['center'][1]))
                size = field_data['size']
                angle = field_data['rotation']
                
                # Berechne Rechteck-Ecken
                rect = ((center[0], center[1]), (size[0], size[1]), angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                cv2.drawContours(image, [box], 0, viz_config['textfeld_color'], viz_config['line_thickness'])
                
                # Beschriftung
                cv2.putText(image, field_name,
                           (center[0] - 50, center[1] - size[1]//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           viz_config['font_scale'],
                           viz_config['textfeld_color'], 2)
                
                # Konfidenz anzeigen
                if 'confidence' in field_data:
                    conf_text = f"{field_data['confidence']:.1%}"
                    cv2.putText(image, conf_text,
                               (center[0] - 30, center[1] + size[1]//2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               viz_config['font_scale'] * 0.8,
                               viz_config['textfeld_color'], 1)
        
        # Speichern
        cv2.imwrite(str(output_path), image)
        logger.debug(f"Visualisierung gespeichert: {output_path}")
    
    def process_image(self, combined_json_path: Path) -> Dict[str, Any]:
        """
        Verarbeitet ein einzelnes Bild
        
        Args:
            combined_json_path: Pfad zur combined JSON Datei
            
        Returns:
            Erweitertes Dictionary mit Textfeld-Positionen
        """
        start_time = datetime.now()
        
        # Lade combined JSON
        with open(combined_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_name = combined_json_path.stem.replace('_combined', '')
        
        # Extrahiere Plus-Marker
        if 'plus_zeichen' not in data.get('detections', {}):
            logger.warning(f"{image_name}: Keine Plus-Marker gefunden")
            self.stats['missing_markers'] += 1
            return data
        
        plus_markers_raw = data['detections']['plus_zeichen']
        
        # Konvertiere Liste zu Dictionary falls nötig
        if isinstance(plus_markers_raw, list):
            plus_markers = {}
            for marker in plus_markers_raw:
                if 'marker_name' in marker:
                    plus_markers[marker['marker_name']] = marker
            logger.debug(f"{image_name}: Konvertierte {len(plus_markers_raw)} Marker von Liste zu Dictionary")
        elif isinstance(plus_markers_raw, dict):
            plus_markers = plus_markers_raw
        else:
            logger.warning(f"{image_name}: Plus-Marker haben unerwartetes Format: {type(plus_markers_raw).__name__}")
            self.stats['missing_markers'] += 1
            return data
        
        # Berechne Transformationsmatrix
        transform_matrix = self._calculate_transform_matrix(plus_markers)
        if transform_matrix is None:
            logger.error(f"{image_name}: Transformation fehlgeschlagen")
            self.stats['failed'] += 1
            return data
        
        # Initialisiere Textfeld-Dictionary
        if 'textfelder' not in data['detections']:
            data['detections']['textfelder'] = {}
        
        # Berechne Position für jedes Textfeld
        for field_name in self.module_config['calculation']['textfelder']:
            if field_name not in self.calculation_model['textfeld_positions']:
                logger.warning(f"Textfeld '{field_name}' nicht im Modell")
                continue
            
            field_model = self.calculation_model['textfeld_positions'][field_name]
            
            # Prüfe Konfidenz-Schwelle
            if (self.module_config['calculation']['use_confidence_threshold'] and 
                field_model['confidence'] < self.module_config['calculation']['min_confidence']):
                logger.warning(f"{image_name}: Konfidenz für '{field_name}' zu niedrig ({field_model['confidence']:.2%})")
                self.stats['low_confidence'] += 1
                continue
            
            # Transformiere Position
            center_x, center_y = self._transform_textfeld_position(
                field_model['normalized_position'],
                transform_matrix
            )
            
            # Berechne Rotation
            rotation = self._calculate_rotation(plus_markers, field_model['relative_rotation'])
            
            # Speichere Textfeld-Daten
            data['detections']['textfelder'][field_name] = {
                'center': [center_x, center_y],
                'size': [field_model['size']['width'], field_model['size']['height']],
                'rotation': rotation,
                'confidence': field_model['confidence'],
                'source': 'calculated_from_markers'
            }
            
            logger.debug(f"{image_name}: {field_name} bei ({center_x}, {center_y}), Rotation: {rotation:.1f}°")
        
        # Update Metadaten
        if '_file_info' not in data:
            data['_file_info'] = {}
        
        data['_file_info'].update({
            'textfeld_calculator_version': self.module_config['module_version'],
            'textfeld_calculation_time': (datetime.now() - start_time).total_seconds(),
            'textfeld_model_version': self.calculation_model['model_info']['version']
        })
        
        # Statistiken updaten
        data['statistics'] = data.get('statistics', {})
        data['statistics']['textfelder'] = {
            'calculated': len(data['detections'].get('textfelder', {})),
            'expected': len(self.module_config['calculation']['textfelder']),
            'complete': len(data['detections'].get('textfelder', {})) == len(self.module_config['calculation']['textfelder'])
        }
        
        return data
    
    def process_all(self) -> Dict[str, Any]:
        """
        Verarbeitet alle Bilder im Input-Verzeichnis
        
        Returns:
            Zusammenfassung der Verarbeitung
        """
        logger.info("="*60)
        logger.info(f"TEXTFELD CALCULATOR - Stapelverarbeitung")
        logger.info("="*60)
        
        # Finde alle combined JSON Dateien
        json_files = sorted(self.input_path.glob("*_combined.json"))
        if not json_files:
            logger.warning(f"Keine combined JSON Dateien gefunden in {self.input_path}")
            return self._create_summary()
        
        logger.info(f"Gefunden: {len(json_files)} Dateien")
        
        # Debug-Zähler
        debug_count = 0
        max_debug = self.module_config['debug']['max_debug_images']
        
        # Verarbeite jede Datei
        for i, json_path in enumerate(json_files, 1):
            logger.info(f"\n[{i}/{len(json_files)}] Verarbeite: {json_path.name}")
            
            try:
                # Verarbeite Bild
                result = self.process_image(json_path)
                
                # Speichere erweiterte JSON
                output_json_path = self.output_metadata_path / f"{json_path.stem.replace('_combined', '')}_textfelder.json"
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    if self.module_config['metadata']['pretty_print']:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    else:
                        json.dump(result, f, ensure_ascii=False)
                
                # Update Statistiken
                if ('statistics' in result and 
                    'textfelder' in result['statistics'] and 
                    result['statistics']['textfelder'].get('complete', False)):
                    self.stats['successful'] += 1
                    
                    # Erstelle Debug-Visualisierung nur bei Erfolg
                    if self.module_config['debug']['enabled'] and debug_count < max_debug:
                        image_name = json_path.stem.replace('_combined', '')
                        image_path = self.input_path.parent.parent / "01_converted" / f"{image_name}.png"
                        
                        if image_path.exists():
                            debug_image_path = self.debug_path / f"{image_name}_textfeld_debug.png"
                            self._create_visualization(image_path, result, debug_image_path)
                            debug_count += 1
                else:
                    self.stats['failed'] += 1
                
            except Exception as e:
                logger.error(f"  ❌ Fehler bei {json_path.name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.stats['errors'].append({
                    'file': json_path.name,
                    'error': str(e)
                })
                self.stats['failed'] += 1
            
            self.stats['total_images'] += 1
        
        # Erstelle und speichere Zusammenfassung
        summary = self._create_summary()
        self._save_summary(summary)
        
        return summary
    
    def _create_summary(self) -> Dict[str, Any]:
        """Erstellt Zusammenfassung der Verarbeitung"""
        success_rate = (self.stats['successful'] / self.stats['total_images'] * 100 
                       if self.stats['total_images'] > 0 else 0)
        
        summary = {
            'module': self.module_config['module_name'],
            'version': self.module_config['module_version'],
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_processed': self.stats['total_images'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'missing_markers': self.stats['missing_markers'],
                'low_confidence': self.stats['low_confidence'],
                'success_rate': f"{success_rate:.1f}%"
            },
            'model_info': {
                'version': self.calculation_model['model_info']['version'],
                'training_samples': self.calculation_model['model_info']['training_samples']
            },
            'errors': self.stats['errors']
        }
        
        # Log Zusammenfassung
        logger.info("\n" + "="*60)
        logger.info("ZUSAMMENFASSUNG")
        logger.info("="*60)
        logger.info(f"✓ Verarbeitet: {self.stats['total_images']} Bilder")
        logger.info(f"✓ Erfolgreich: {self.stats['successful']} ({success_rate:.1f}%)")
        logger.info(f"✗ Fehlgeschlagen: {self.stats['failed']}")
        if self.stats['missing_markers'] > 0:
            logger.info(f"⚠ Fehlende Marker: {self.stats['missing_markers']}")
        if self.stats['low_confidence'] > 0:
            logger.info(f"⚠ Niedrige Konfidenz: {self.stats['low_confidence']}")
        
        return summary
    
    def _save_summary(self, summary: Dict):
        """Speichert Zusammenfassung als JSON"""
        summary_path = self.output_metadata_path / f"textfeld_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📊 Zusammenfassung gespeichert: {summary_path}")


def main():
    """Hauptfunktion für direkten Aufruf"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Textfeld Calculator für Zählerkarten')
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
        # Initialisiere Calculator
        calculator = TextfeldCalculator(config_path=args.config)
        
        if args.single:
            # Einzeldatei verarbeiten
            json_path = Path(args.single)
            if not json_path.exists():
                logger.error(f"Datei nicht gefunden: {json_path}")
                return 1
            
            result = calculator.process_image(json_path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Stapelverarbeitung
            calculator.process_all()
        
        return 0
        
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())