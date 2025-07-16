# pdf_converter.py V2.0
"""
PDF zu PNG Konverter für Zählerkarten Pipeline V2
Konvertiert PDF-Dateien aus Scanner/ in PNG-Bilder für die weitere Verarbeitung.
Identische Konvertierung wie beim YOLO-Training verwendet.
"""

import os
import json
import sys
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from pdf2image import convert_from_path
import traceback


class PDFtoPNGConverter:
    """Konvertiert PDF-Dateien zu PNG-Bildern mit Debug-Funktionalität"""
    
    def __init__(self, config_path: str = None):
        """
        Initialisiert den Konverter
        
        Args:
            config_path: Pfad zum Config-Verzeichnis (optional)
        """
        self.start_time = datetime.now()
        
        # Lade Konfigurationen
        self.global_config, self.module_config = self._load_configs(config_path)
        
        # Setze Pfade
        self._setup_paths()
        
        # Initialisiere Logging
        self._setup_logging()
        
        # Statistiken
        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        self.logger.info(f"PDF zu PNG Konverter V2.0 initialisiert")
        self.logger.info(f"Input: {self.input_path}")
        self.logger.info(f"Output: {self.output_path}")
    
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
        
        # Lade Modul-Config
        module_config_path = config_path / "pdf_to_png_config.json"
        with open(module_config_path, 'r', encoding='utf-8') as f:
            module_config = json.load(f)
        
        return global_config, module_config
    
    def _setup_paths(self):
        """Erstellt alle benötigten Pfade basierend auf den Configs"""
        base_path = Path(self.global_config['base_path'])
        
        # Input/Output Pfade
        self.input_path = base_path / self.global_config['paths']['scanner_dir'] / self.module_config['paths']['input_subdir']
        self.output_path = base_path / self.global_config['paths']['pipeline_data'] / self.module_config['paths']['output_subdir']
        self.metadata_path = base_path / self.global_config['paths']['pipeline_data'] / self.module_config['paths']['metadata_subdir']
        
        # Debug Pfad
        self.debug_path = base_path / self.global_config['paths']['debug_root'] / self.module_config['paths']['debug_subdir']
        
        # Log Pfad
        self.log_path = base_path / self.global_config['paths']['log_dir'] / "pdf_converter"
        
        # Erstelle Verzeichnisse
        for path in [self.output_path, self.metadata_path, self.debug_path, self.log_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Konfiguriert das Logging-System"""
        log_enabled = self.module_config['debug'].get('save_conversion_log', True)
        log_level = self.module_config['debug'].get('log_level', 'INFO')
        
        # Logger erstellen
        self.logger = logging.getLogger('PDFtoPNGConverter')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Vorhandene Handler entfernen
        self.logger.handlers = []
        
        # Console Handler (immer aktiv)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File Handler (optional)
        if log_enabled:
            log_file = self.log_path / f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level))
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging aktiviert: {log_file}")
    
    def find_pdf_files(self) -> List[Path]:
        """Findet alle PDF-Dateien im Input-Verzeichnis"""
        pattern = self.global_config['file_patterns']['pdf_files']
        pdf_files = list(self.input_path.glob(pattern))
        
        self.logger.info(f"Gefunden: {len(pdf_files)} PDF-Dateien")
        return sorted(pdf_files)
    
    def convert_pdf_to_png(self, pdf_path: Path) -> Optional[Dict]:
        """
        Konvertiert eine einzelne PDF zu PNG
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Metadata dict oder None bei Fehler
        """
        start_time = datetime.now()
        pdf_name = pdf_path.stem
        
        try:
            self.logger.info(f"Konvertiere: {pdf_path.name}")
            
            # PDF zu PIL Image konvertieren (identisch zum Training)
            images = convert_from_path(
                pdf_path,
                dpi=self.module_config['conversion']['dpi'],
                first_page=self.module_config['conversion']['first_page'],
                last_page=self.module_config['conversion']['last_page']
            )
            
            if not images:
                raise ValueError("Keine Bilder aus PDF extrahiert")
            
            # PIL zu OpenCV konvertieren (identisch zum Training)
            image_pil = images[0]
            image_rgb = np.array(image_pil)
            
            # Farbformat konvertieren
            if image_rgb.shape[2] == 4:  # RGBA
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Validierung
            height, width, channels = image_bgr.shape
            if self.module_config['processing']['validate_output']:
                min_w = self.module_config['processing']['min_width']
                min_h = self.module_config['processing']['min_height']
                max_w = self.module_config['processing']['max_width']
                max_h = self.module_config['processing']['max_height']
                
                if not (min_w <= width <= max_w and min_h <= height <= max_h):
                    self.logger.warning(f"Bildgröße außerhalb der Grenzen: {width}x{height}")
            
            # PNG speichern
            output_file = self.output_path / f"{pdf_name}.png"
            cv2.imwrite(str(output_file), image_bgr)
            
            # Metadata erstellen
            duration = (datetime.now() - start_time).total_seconds()
            metadata = {
                "source_pdf": pdf_path.name,
                "source_path": str(pdf_path),
                "output_png": output_file.name,
                "output_path": str(output_file),
                "conversion": {
                    "timestamp": datetime.now().isoformat(),
                    "dpi": self.module_config['conversion']['dpi'],
                    "duration_seconds": round(duration, 3)
                },
                "image": {
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "color_format": self.module_config['conversion']['color_format'],
                    "size_bytes": output_file.stat().st_size
                }
            }
            
            # Metadata speichern
            if self.module_config['metadata']['include_file_info']:
                metadata_file = self.metadata_path / f"{pdf_name}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    if self.module_config['metadata']['pretty_print']:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    else:
                        json.dump(metadata, f, ensure_ascii=False)
            
            self.logger.info(f"✓ Konvertiert: {pdf_name} ({width}x{height}) in {duration:.2f}s")
            return metadata
            
        except Exception as e:
            error_msg = f"Fehler bei {pdf_path.name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            self.stats['errors'].append({
                'file': pdf_path.name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def create_debug_overview(self, converted_files: List[Dict]):
        """Erstellt eine Debug-Übersicht mit Thumbnails aller konvertierten Bilder"""
        if not self.module_config['debug']['create_overview']:
            return
        
        self.logger.info("Erstelle Debug-Übersicht...")
        
        # Thumbnail-Einstellungen
        thumb_w, thumb_h = self.module_config['debug']['thumbnail_size']
        max_thumbs = self.module_config['debug']['max_thumbnails']
        
        # Begrenze Anzahl
        files_to_show = converted_files[:max_thumbs]
        
        # Berechne Grid-Layout
        cols = 4
        rows = (len(files_to_show) + cols - 1) // cols
        
        # Erstelle Übersichtsbild
        margin = 10
        overview_w = cols * thumb_w + (cols + 1) * margin
        overview_h = rows * thumb_h + (rows + 1) * margin
        overview = np.ones((overview_h, overview_w, 3), dtype=np.uint8) * 255
        
        # Füge Thumbnails hinzu
        for idx, file_meta in enumerate(files_to_show):
            row = idx // cols
            col = idx % cols
            
            # Lade Bild
            img_path = Path(file_meta['output_path'])
            if img_path.exists():
                img = cv2.imread(str(img_path))
                
                # Erstelle Thumbnail
                aspect = img.shape[1] / img.shape[0]
                if aspect > thumb_w / thumb_h:
                    new_w = thumb_w
                    new_h = int(thumb_w / aspect)
                else:
                    new_h = thumb_h
                    new_w = int(thumb_h * aspect)
                
                thumb = cv2.resize(img, (new_w, new_h))
                
                # Position berechnen
                x = col * thumb_w + (col + 1) * margin + (thumb_w - new_w) // 2
                y = row * thumb_h + (row + 1) * margin + (thumb_h - new_h) // 2
                
                # Thumbnail einfügen
                overview[y:y+new_h, x:x+new_w] = thumb
                
                # Dateiname hinzufügen
                text = file_meta['source_pdf'][:20]  # Kürzen bei Bedarf
                cv2.putText(overview, text, (x, y + new_h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Übersicht speichern
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        overview_file = self.debug_path / f"conversion_overview_{timestamp}.png"
        cv2.imwrite(str(overview_file), overview)
        
        # Statistik-Text hinzufügen
        stats_text = f"Konvertiert: {len(converted_files)} PDFs | Fehler: {self.stats['failed']}"
        cv2.putText(overview, stats_text, (margin, overview_h - margin), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(str(overview_file), overview)
        self.logger.info(f"Debug-Übersicht gespeichert: {overview_file.name}")
    
    def run(self):
        """Führt die komplette Konvertierung durch"""
        self.logger.info("="*60)
        self.logger.info("PDF zu PNG Konvertierung gestartet")
        self.logger.info("="*60)
        
        # PDFs finden
        pdf_files = self.find_pdf_files()
        if not pdf_files:
            self.logger.warning("Keine PDF-Dateien gefunden!")
            return
        
        self.stats['total_pdfs'] = len(pdf_files)
        converted_files = []
        
        # Konvertiere jede PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            self.logger.info(f"\n[{i}/{len(pdf_files)}] Verarbeite: {pdf_path.name}")
            
            metadata = self.convert_pdf_to_png(pdf_path)
            
            if metadata:
                self.stats['successful'] += 1
                converted_files.append(metadata)
            else:
                self.stats['failed'] += 1
        
        # Erstelle Debug-Übersicht
        if converted_files:
            self.create_debug_overview(converted_files)
        
        # Zusammenfassung
        duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.info("\n" + "="*60)
        self.logger.info("KONVERTIERUNG ABGESCHLOSSEN")
        self.logger.info(f"Gesamt: {self.stats['total_pdfs']} PDFs")
        self.logger.info(f"Erfolgreich: {self.stats['successful']}")
        self.logger.info(f"Fehler: {self.stats['failed']}")
        self.logger.info(f"Dauer: {duration:.1f} Sekunden")
        self.logger.info("="*60)
        
        # Speichere Zusammenfassung
        summary_file = self.metadata_path / f"conversion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "statistics": self.stats,
            "converted_files": converted_files
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    """Hauptfunktion"""
    converter = PDFtoPNGConverter()
    converter.run()


if __name__ == "__main__":
    main()