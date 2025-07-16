#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-Korrektur Status Analyzer - Modul 11
==========================================
Analysiert den Status ALLER Karten nach Pipeline + Korrekturen.
Erstellt Prioritätslisten für die finale Ganze-Karte-Korrektur.

Workflow:
1. Lädt alle PDFs aus Scanner-Ordner
2. Analysiert QR, Z-Kästchen, D-Kästchen, Textfelder pro Karte
3. Bewertet Qualität und Vollständigkeit  
4. Erstellt Prioritätslisten (Kritisch → Komplett)
5. Schätzt verbleibende Arbeitszeit
6. Bereitet Daten für Ganze-Karte-GUI vor

Version: 1.0
Autor: Oliver Krispel + KI-System
Datum: 2025-07-15
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostKorrekturAnalyzer:
    """
    Analysiert den Vollständigkeits-Status aller Karten nach Pipeline + Korrekturen
    
    Features:
    - Vollständige Datensammlung pro Karte
    - Intelligente Qualitätsbewertung  
    - Priorisierung nach Dringlichkeit
    - Zeitschätzung für Restarbeit
    - Vorbereitung für Ganze-Karte-GUI
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den Post-Korrektur Analyzer
        
        Args:
            config_path: Pfad zum Config-Verzeichnis (optional)
        """
        # Lade Konfigurationen
        self.global_config, self.module_config = self._load_configs(config_path)
        
        # Setup Pfade
        self._setup_paths()
        
        # Setup Logging
        self._setup_logging()
        
        # Analyseergebnisse
        self.card_status = {}
        self.priority_lists = {}
        self.statistics = {}
        
        logger.info("✅ Post-Korrektur Analyzer initialisiert")
    
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
        module_config_path = config_path / "post_korrektur_analyzer_config.json"
        with open(module_config_path, 'r', encoding='utf-8') as f:
            module_config = json.load(f)
        
        return global_config, module_config
    
    def _setup_paths(self):
        """Richtet alle notwendigen Pfade ein"""
        base_path = Path(self.global_config['base_path'])
        
        # Input Pfade (Datenquellen)
        self.scanner_path = base_path / self.module_config['data_sources']['scanner_dir']
        self.qr_data_path = base_path / self.module_config['data_sources']['qr_data_dir']
        self.text_data_path = base_path / self.module_config['data_sources']['text_data_dir']
        self.kaestchen_data_path = base_path / self.module_config['data_sources']['kaestchen_data_dir']
        self.converted_images_path = base_path / self.module_config['data_sources']['converted_images_dir']
        self.extracted_fields_path = base_path / self.module_config['data_sources']['extracted_fields_dir']
        
        # Output Pfade
        self.output_path = base_path / self.module_config['paths']['output_dir']
        self.reports_path = self.output_path / self.module_config['paths']['reports_subdir']
        self.priority_lists_path = self.output_path / self.module_config['paths']['priority_lists_subdir']
        self.complete_data_path = self.output_path / self.module_config['paths']['complete_data_subdir']
        
        # Erstelle Output-Verzeichnisse
        for path in [self.output_path, self.reports_path, self.priority_lists_path, self.complete_data_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Konfiguriert erweiterte Logging-Ausgabe"""
        self.start_time = datetime.now()
        logger.info(f"🔍 Post-Korrektur Analyzer V{self.module_config['module_version']}")
        logger.info(f"   Scanner-Ordner: {self.scanner_path}")
        logger.info(f"   Output: {self.output_path}")
    
    def analyze_complete_status(self) -> Dict:
        """
        Vollständige Analyse aller Karten
        
        Returns:
            Zusammenfassende Analyse-Ergebnisse
        """
        logger.info("="*60)
        logger.info("POST-KORREKTUR STATUS ANALYSE")
        logger.info("="*60)
        
        # 1. Finde alle PDFs im Scanner-Ordner
        all_basenames = self._get_all_basenames()
        
        if not all_basenames:
            logger.warning("⚠️ Keine PDFs im Scanner-Ordner gefunden!")
            return {'error': 'no_pdfs_found'}
        
        logger.info(f"📁 Gefunden: {len(all_basenames)} Karten zur Analyse")
        
        # 2. Analysiere jede Karte einzeln
        self._analyze_individual_cards(all_basenames)
        
        # 3. Erstelle Prioritätslisten
        self._create_priority_lists()
        
        # 4. Generiere Statistiken
        self._generate_statistics()
        
        # 5. Erstelle Reports
        self._create_reports()
        
        # 6. Bereite Ganze-Karte-GUI Daten vor
        self._prepare_gui_data()
        
        logger.info("✅ Post-Korrektur Analyse abgeschlossen")
        
        return self._create_summary()
    
    def _get_all_basenames(self) -> Set[str]:
        """Extrahiert alle Basisnamen aus PDF-Dateien"""
        pdf_pattern = self.module_config['file_patterns']['pdf_files']
        pdf_files = list(self.scanner_path.glob(pdf_pattern))
        
        # Extrahiere Basisnamen (ohne .pdf)
        basenames = set()
        for pdf_file in pdf_files:
            basename = pdf_file.stem
            basenames.add(basename)
        
        return basenames
    
    def _analyze_individual_cards(self, all_basenames: Set[str]):
        """Analysiert jede Karte einzeln"""
        logger.info(f"\n🔍 ANALYSE EINZELNER KARTEN")
        
        for i, basename in enumerate(sorted(all_basenames), 1):
            if i % 50 == 0:
                logger.info(f"   Fortschritt: {i}/{len(all_basenames)} Karten")
            
            card_status = self._analyze_single_card(basename)
            self.card_status[basename] = card_status
    
    def _analyze_single_card(self, basename: str) -> Dict:
        """Analysiert eine einzelne Karte vollständig"""
        card_status = {
            'basename': basename,
            'qr_code': self._analyze_qr_status(basename),
            'z_kaestchen': self._analyze_z_kaestchen_status(basename),
            'd_kaestchen': self._analyze_d_kaestchen_status(basename),
            'textfelder': self._analyze_textfeld_status(basename),
            'image_paths': self._get_image_paths(basename),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Gesamtbewertung der Karte
        card_status['overall_priority'] = self._calculate_overall_priority(card_status)
        card_status['estimated_work_minutes'] = self._estimate_work_time(card_status)
        card_status['ready_for_export'] = self._is_ready_for_export(card_status)
        
        return card_status
    
    def _analyze_qr_status(self, basename: str) -> Dict:
        """Analysiert QR-Code Status"""
        qr_file = self.qr_data_path / f"{basename}_qr.json"
        
        if not qr_file.exists():
            return {
                'status': 'nicht_verarbeitet',
                'icon': '❌',
                'data': None,
                'message': 'QR-Datei nicht gefunden'
            }
        
        try:
            with open(qr_file, 'r', encoding='utf-8') as f:
                qr_data = json.load(f)
            
            # Prüfe QR-Code Erfolg
            if qr_data.get('qr_success', False):
                return {
                    'status': 'verfügbar',
                    'icon': '✅',
                    'data': qr_data,
                    'zaehlernummer': qr_data.get('zaehlernummer'),
                    'zaehlerart': qr_data.get('zaehlerart'),
                    'message': 'QR-Code erfolgreich gelesen'
                }
            else:
                return {
                    'status': 'nicht_lesbar',
                    'icon': '❌',
                    'data': qr_data,
                    'message': 'QR-Code nicht lesbar'
                }
                
        except Exception as e:
            return {
                'status': 'fehlgeschlagen',
                'icon': '❌',
                'data': None,
                'message': f'Fehler beim Laden: {e}'
            }
    
    def _analyze_z_kaestchen_status(self, basename: str) -> Dict:
        """Analysiert Z-Kästchen Status basierend auf ECHTEN Dateiformaten"""
        
        # 1. Prüfe FEHLT_KAESTCHEN.json
        fehlt_kaestchen_file = self.kaestchen_data_path / "FEHLT_KAESTCHEN.json"
        is_failed_card = False
        
        if fehlt_kaestchen_file.exists():
            try:
                with open(fehlt_kaestchen_file, 'r', encoding='utf-8') as f:
                    fehlt_list = json.load(f)
                is_failed_card = basename in fehlt_list
            except Exception as e:
                logger.debug(f"Fehler beim Laden von FEHLT_KAESTCHEN.json: {e}")
        
        # 2. Prüfe individuelle Kästchen-Datei
        kaestchen_file = self.kaestchen_data_path / f"{basename}.json"
        
        if is_failed_card and not kaestchen_file.exists():
            return {
                'status': 'fehlgeschlagen',
                'icon': '❌',
                'quality_level': 'kritisch',
                'message': 'In FEHLT_KAESTCHEN.json, keine Datei vorhanden'
            }
        
        if not kaestchen_file.exists():
            return {
                'status': 'nicht_verarbeitet',
                'icon': '❌',
                'quality_level': 'kritisch',
                'message': 'Z-Kästchen Datei nicht gefunden'
            }
        
        try:
            with open(kaestchen_file, 'r', encoding='utf-8') as f:
                kaestchen_data = json.load(f)
            
            z_boxes = kaestchen_data.get('z_kaestchen', [])
            
            if not z_boxes:
                return {
                    'status': 'keine_daten',
                    'icon': '❌',
                    'quality_level': 'kritisch',
                    'message': 'Keine Z-Kästchen Daten'
                }
            
            # Analysiere Qualität (berücksichtigt Korrekturen)
            quality_analysis = self._evaluate_kaestchen_quality(z_boxes, 'z_kaestchen')
            
            # Prüfe ob Korrekturen angewendet wurden
            has_corrections = any(box.get('corrected', False) for box in z_boxes)
            needs_manual_review = kaestchen_data.get('needs_manual_review', False)
            
            return {
                'status': 'verarbeitet',
                'icon': quality_analysis['icon'],
                'quality_level': quality_analysis['level'],
                'data': z_boxes,
                'problematic_count': quality_analysis['problematic_count'],
                'average_confidence': quality_analysis['average_confidence'],
                'min_confidence': quality_analysis['min_confidence'],
                'zaehlerstand_text': kaestchen_data.get('zaehlerstand_text'),
                'has_corrections': has_corrections,
                'needs_manual_review': needs_manual_review,
                'was_in_fehlt_liste': is_failed_card,
                'message': quality_analysis['message']
            }
            
        except Exception as e:
            return {
                'status': 'fehlgeschlagen',
                'icon': '❌',
                'quality_level': 'kritisch',
                'message': f'Fehler beim Laden: {e}'
            }
    
    def _analyze_d_kaestchen_status(self, basename: str) -> Dict:
        """Analysiert D-Kästchen Status basierend auf ECHTEN Dateiformaten"""
        
        # 1. Prüfe FEHLT_KAESTCHEN.json  
        fehlt_kaestchen_file = self.kaestchen_data_path / "FEHLT_KAESTCHEN.json"
        is_failed_card = False
        
        if fehlt_kaestchen_file.exists():
            try:
                with open(fehlt_kaestchen_file, 'r', encoding='utf-8') as f:
                    fehlt_list = json.load(f)
                is_failed_card = basename in fehlt_list
            except Exception as e:
                logger.debug(f"Fehler beim Laden von FEHLT_KAESTCHEN.json: {e}")
        
        # 2. Prüfe individuelle Kästchen-Datei
        kaestchen_file = self.kaestchen_data_path / f"{basename}.json"
        
        if is_failed_card and not kaestchen_file.exists():
            return {
                'status': 'fehlgeschlagen',
                'icon': '❌',
                'quality_level': 'kritisch',
                'message': 'In FEHLT_KAESTCHEN.json, keine Datei vorhanden'
            }
        
        if not kaestchen_file.exists():
            return {
                'status': 'nicht_verarbeitet',
                'icon': '❌',
                'quality_level': 'kritisch',
                'message': 'D-Kästchen Datei nicht gefunden'
            }
        
        try:
            with open(kaestchen_file, 'r', encoding='utf-8') as f:
                kaestchen_data = json.load(f)
            
            d_boxes = kaestchen_data.get('d_kaestchen', [])
            
            if not d_boxes:
                return {
                    'status': 'keine_daten',
                    'icon': '❌',
                    'quality_level': 'kritisch',
                    'message': 'Keine D-Kästchen Daten'
                }
            
            # Analysiere Qualität (berücksichtigt Korrekturen)
            quality_analysis = self._evaluate_kaestchen_quality(d_boxes, 'd_kaestchen')
            
            # Prüfe ob Korrekturen angewendet wurden
            has_corrections = any(box.get('corrected', False) for box in d_boxes)
            needs_manual_review = kaestchen_data.get('needs_manual_review', False)
            
            return {
                'status': 'verarbeitet',
                'icon': quality_analysis['icon'],
                'quality_level': quality_analysis['level'],
                'data': d_boxes,
                'problematic_count': quality_analysis['problematic_count'],
                'average_confidence': quality_analysis['average_confidence'],
                'min_confidence': quality_analysis['min_confidence'],
                'ablesedatum': kaestchen_data.get('ablesedatum'),
                'datum_quelle': kaestchen_data.get('datum_quelle'),
                'has_corrections': has_corrections,
                'needs_manual_review': needs_manual_review,
                'was_in_fehlt_liste': is_failed_card,
                'message': quality_analysis['message']
            }
            
        except Exception as e:
            return {
                'status': 'fehlgeschlagen',
                'icon': '❌',
                'quality_level': 'kritisch',
                'message': f'Fehler beim Laden: {e}'
            }
    
    def _evaluate_kaestchen_quality(self, boxes: List[Dict], kaestchen_type: str) -> Dict:
        """Bewertet die Qualität von Z- oder D-Kästchen nach Korrektur"""
        if not boxes:
            return {
                'level': 'kritisch',
                'icon': '🚨',
                'problematic_count': 99,
                'average_confidence': 0.0,
                'min_confidence': 0.0,
                'message': 'Keine Kästchen-Daten'
            }
        
        # Zähle problematische Kästchen
        problematic_codes = self.module_config['analysis_criteria'][kaestchen_type]['problematic_status_codes']
        problematic_count = sum(1 for box in boxes if box.get('status') in problematic_codes)
        
        # Berechne Confidence-Statistiken
        confidences = [box.get('confidence', 0.0) for box in boxes if 'confidence' in box]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0
        
        # Bewerte nach Post-Korrektur Schwellwerten
        thresholds = self.module_config['analysis_criteria'][kaestchen_type]['post_correction_thresholds']
        
        for level_name, criteria in thresholds.items():
            if (avg_confidence >= criteria['min_confidence'] and 
                problematic_count <= criteria['max_problematic_count']):
                
                status_parts = criteria['status'].split(' ', 1)
                icon = status_parts[0]
                level = status_parts[1] if len(status_parts) > 1 else level_name
                
                return {
                    'level': level,
                    'icon': icon,
                    'problematic_count': problematic_count,
                    'average_confidence': round(avg_confidence, 3),
                    'min_confidence': round(min_confidence, 3),
                    'message': f'{problematic_count} problematische von {len(boxes)} Kästchen'
                }
        
        # Fallback
        return {
            'level': 'kritisch',
            'icon': '🚨',
            'problematic_count': problematic_count,
            'average_confidence': round(avg_confidence, 3),
            'min_confidence': round(min_confidence, 3),
            'message': f'{problematic_count} problematische von {len(boxes)} Kästchen'
        }
    
    def _analyze_textfeld_status(self, basename: str) -> Dict:
        """Analysiert Textfeld Status basierend auf ECHTEN Dateiformaten"""
        
        # 1. Prüfe FEHLT_OCR.json für diese Karte
        fehlt_ocr_file = self.text_data_path / "FEHLT_OCR.json"
        failed_fields = []
        
        if fehlt_ocr_file.exists():
            try:
                with open(fehlt_ocr_file, 'r', encoding='utf-8') as f:
                    fehlt_data = json.load(f)
                
                # Finde Fehlschläge für diese Karte
                for item in fehlt_data:
                    if item.get('datei') == basename:
                        failed_fields.append(item.get('feld'))
            except Exception as e:
                logger.debug(f"Fehler beim Laden von FEHLT_OCR.json: {e}")
        
        # 2. Prüfe individuelle Textfeld-Dateien
        status = {}
        textfeld_types = ['zaehlerart', 'zaehlernummer']
        
        for field_type in textfeld_types:
            field_file = self.text_data_path / f"{basename}_{field_type}.json"
            
            if field_type in failed_fields:
                # Explizit als fehlgeschlagen markiert
                status[field_type] = {
                    'available': False,
                    'data': None,
                    'message': 'In FEHLT_OCR.json gelistet'
                }
            else:
                # Prüfe Datei-Existenz und Inhalt
                status[field_type] = self._check_textfeld_file(field_file)
        
        # 3. Gesamtstatus berechnen
        available_count = sum(1 for field_status in status.values() if field_status['available'])
        total_fields = len(status)
        
        if available_count == total_fields:
            overall_status = 'verfügbar'
            icon = '✅'
            message = 'Alle Textfelder verfügbar'
        elif available_count > 0:
            overall_status = 'teilweise'
            icon = '⚠️'
            message = f'{available_count}/{total_fields} Textfelder verfügbar'
        else:
            overall_status = 'manual_ocr_nötig'
            icon = '🔧'
            message = 'Manuelle OCR erforderlich'
        
        return {
            'status': overall_status,
            'icon': icon,
            'fields': status,
            'available_count': available_count,
            'total_fields': total_fields,
            'failed_fields': failed_fields,
            'message': message
        }
    
    def _check_textfeld_file(self, file_path: Path) -> Dict:
        """Prüft einzelne Textfeld-Datei"""
        if not file_path.exists():
            return {'available': False, 'data': None, 'message': 'Datei nicht gefunden'}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Prüfe ob erfolgreich
            success = data.get('success', False)
            text = data.get('text', '').strip()
            
            if success and text:
                return {'available': True, 'data': data, 'text': text, 'message': 'Erfolgreich'}
            else:
                return {'available': False, 'data': data, 'message': 'Kein Text erkannt'}
                
        except Exception as e:
            return {'available': False, 'data': None, 'message': f'Fehler: {e}'}
    
    def _get_image_paths(self, basename: str) -> Dict:
        """Sammelt alle relevanten Bildpfade für eine Karte"""
        return {
            'main_image': f"pipeline_data/01_converted/{basename}.png",
            'z_kaestchen_dir': f"pipeline_data/05_extracted_fields/z_kaestchen",
            'd_kaestchen_dir': f"pipeline_data/05_extracted_fields/d_kaestchen",
            'textfelder_dir': f"pipeline_data/05_extracted_fields/textfelder",
            'qr_dir': f"pipeline_data/05_extracted_fields/QR"
        }
    
    def _calculate_overall_priority(self, card_status: Dict) -> str:
        """Berechnet die Gesamtpriorität einer Karte"""
        # Prüfe Kriterien für jede Prioritätsstufe
        prioritization = self.module_config['prioritization']
        
        # Kritisch
        if (card_status['z_kaestchen']['quality_level'] == 'kritisch' or
            card_status['d_kaestchen']['quality_level'] == 'kritisch' or
            card_status['z_kaestchen']['status'] == 'nicht_verarbeitet' or
            card_status['d_kaestchen']['status'] == 'nicht_verarbeitet'):
            return 'critical_immediate'
        
        # Hoch
        if (card_status['z_kaestchen']['quality_level'] == 'review_nötig' or
            card_status['d_kaestchen']['quality_level'] == 'review_nötig'):
            return 'high_priority'
        
        # Mittel
        if (card_status['z_kaestchen']['quality_level'] == 'akzeptabel' or
            card_status['d_kaestchen']['quality_level'] == 'akzeptabel' or
            card_status['qr_code']['status'] == 'nicht_lesbar' or
            card_status['textfelder']['status'] == 'manual_ocr_nötig'):
            return 'medium_priority'
        
        # Niedrig
        if (card_status['z_kaestchen']['quality_level'] == 'sehr_gut' or
            card_status['d_kaestchen']['quality_level'] == 'sehr_gut'):
            return 'low_priority'
        
        # Komplett
        return 'complete'
    
    def _estimate_work_time(self, card_status: Dict) -> int:
        """Schätzt Arbeitszeit in Minuten für eine Karte"""
        priority = card_status['overall_priority']
        time_estimates = self.module_config['statistics']['time_estimates']
        
        time_mapping = {
            'critical_immediate': time_estimates['per_critical_card_minutes'],
            'high_priority': time_estimates['per_high_priority_card_minutes'],
            'medium_priority': time_estimates['per_medium_priority_card_minutes'],
            'low_priority': time_estimates['per_low_priority_card_minutes'],
            'complete': 0
        }
        
        return time_mapping.get(priority, 3)
    
    def _is_ready_for_export(self, card_status: Dict) -> bool:
        """Prüft ob Karte bereit für Excel-Export ist"""
        return (card_status['overall_priority'] == 'complete' and
                card_status['z_kaestchen']['status'] == 'verarbeitet' and
                card_status['d_kaestchen']['status'] == 'verarbeitet')
    
    def _create_priority_lists(self):
        """Erstellt Prioritätslisten für verschiedene Bearbeitungsstufen"""
        logger.info(f"\n📋 ERSTELLE PRIORITÄTSLISTEN")
        
        # Gruppiere nach Priorität
        priority_groups = defaultdict(list)
        
        for basename, card_status in self.card_status.items():
            priority = card_status['overall_priority']
            priority_groups[priority].append({
                'basename': basename,
                'card_status': card_status
            })
        
        # Sortiere innerhalb jeder Gruppe
        for priority, cards in priority_groups.items():
            # Sortiere kritische nach schlechtester Qualität zuerst
            if priority == 'critical_immediate':
                cards.sort(key=lambda x: (
                    x['card_status']['z_kaestchen'].get('problematic_count', 0) +
                    x['card_status']['d_kaestchen'].get('problematic_count', 0)
                ), reverse=True)
            else:
                # Alphabetisch
                cards.sort(key=lambda x: x['basename'])
        
        self.priority_lists = dict(priority_groups)
        
        # Logging
        for priority, cards in self.priority_lists.items():
            priority_config = self.module_config['prioritization'][priority]
            icon = priority_config['icon']
            description = priority_config['description']
            logger.info(f"   {icon} {priority.upper()}: {len(cards)} Karten - {description}")
    
    def _generate_statistics(self):
        """Generiert umfassende Statistiken"""
        logger.info(f"\n📊 GENERIERE STATISTIKEN")
        
        total_cards = len(self.card_status)
        
        # Prioritäts-Statistiken
        priority_stats = {}
        total_work_time = 0
        
        for priority, cards in self.priority_lists.items():
            count = len(cards)
            work_time = sum(card['card_status']['estimated_work_minutes'] for card in cards)
            
            priority_stats[priority] = {
                'count': count,
                'percentage': round((count / total_cards) * 100, 1),
                'estimated_work_minutes': work_time,
                'estimated_work_hours': round(work_time / 60, 1)
            }
            total_work_time += work_time
        
        # Qualitäts-Statistiken
        quality_stats = {
            'qr_code': Counter(),
            'z_kaestchen': Counter(),
            'd_kaestchen': Counter(),
            'textfelder': Counter()
        }
        
        for card_status in self.card_status.values():
            quality_stats['qr_code'][card_status['qr_code']['status']] += 1
            quality_stats['z_kaestchen'][card_status['z_kaestchen']['quality_level']] += 1
            quality_stats['d_kaestchen'][card_status['d_kaestchen']['quality_level']] += 1
            quality_stats['textfelder'][card_status['textfelder']['status']] += 1
        
        # Export-Bereitschaft
        ready_for_export = sum(1 for card_status in self.card_status.values() 
                              if card_status['ready_for_export'])
        
        self.statistics = {
            'total_cards': total_cards,
            'analysis_timestamp': datetime.now().isoformat(),
            'priority_breakdown': priority_stats,
            'quality_breakdown': quality_stats,
            'export_readiness': {
                'ready_count': ready_for_export,
                'ready_percentage': round((ready_for_export / total_cards) * 100, 1),
                'remaining_count': total_cards - ready_for_export
            },
            'work_estimates': {
                'total_minutes': total_work_time,
                'total_hours': round(total_work_time / 60, 1),
                'total_days': round(total_work_time / (8 * 60), 1)  # 8h Arbeitstag
            }
        }
        
        # Logging der wichtigsten Statistiken
        logger.info(f"   📊 {total_cards} Karten analysiert")
        logger.info(f"   ✅ {ready_for_export} Karten export-bereit ({ready_for_export/total_cards*100:.1f}%)")
        logger.info(f"   ⏱️ Geschätzte Restarbeit: {total_work_time/60:.1f} Stunden")
    
    def _create_reports(self):
        """Erstellt detaillierte Reports"""
        logger.info(f"\n📄 ERSTELLE REPORTS")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Haupt-Report mit allen Daten
        main_report = {
            'analysis_info': {
                'timestamp': self.statistics['analysis_timestamp'],
                'total_cards': self.statistics['total_cards'],
                'analyzer_version': self.module_config['module_version']
            },
            'statistics': self.statistics,
            'priority_lists': self.priority_lists,
            'individual_cards': self.card_status
        }
        
        main_report_file = self.reports_path / f"post_korrektur_analysis_{timestamp}.json"
        with open(main_report_file, 'w', encoding='utf-8') as f:
            json.dump(main_report, f, indent=2, ensure_ascii=False)
        
        # 2. Kompakte Übersicht
        summary_report = {
            'summary': self.statistics,
            'priority_counts': {
                priority: len(cards) for priority, cards in self.priority_lists.items()
            },
            'next_actions': self._generate_next_actions()
        }
        
        summary_file = self.reports_path / f"summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   📄 Haupt-Report: {main_report_file.name}")
        logger.info(f"   📄 Zusammenfassung: {summary_file.name}")
    
    def _generate_next_actions(self) -> List[str]:
        """Generiert Handlungsempfehlungen"""
        actions = []
        
        # Prüfe Prioritätslisten
        critical_count = len(self.priority_lists.get('critical_immediate', []))
        high_count = len(self.priority_lists.get('high_priority', []))
        
        if critical_count > 0:
            actions.append(f"🚨 SOFORT: {critical_count} kritische Karten bearbeiten")
        
        if high_count > 0:
            actions.append(f"🔧 HOCH: {high_count} Karten mit vielen Unsicherheiten prüfen")
        
        # Textfeld-Probleme
        textfeld_issues = sum(1 for card_status in self.card_status.values() 
                             if card_status['textfelder']['status'] == 'manual_ocr_nötig')
        if textfeld_issues > 0:
            actions.append(f"📝 TEXTFELDER: {textfeld_issues} Karten brauchen manuelle OCR")
        
        # QR-Code Probleme
        qr_issues = sum(1 for card_status in self.card_status.values() 
                       if card_status['qr_code']['status'] == 'nicht_lesbar')
        if qr_issues > 0:
            actions.append(f"❌ QR-CODES: {qr_issues} Karten mit unlesbaren QR-Codes")
        
        # Export-Bereitschaft
        ready_count = self.statistics['export_readiness']['ready_count']
        total_count = self.statistics['total_cards']
        
        if ready_count < total_count:
            remaining = total_count - ready_count
            actions.append(f"📊 ZIEL: {remaining} Karten für Excel-Export fertigstellen")
        
        return actions
    
    def _prepare_gui_data(self):
        """Bereitet Daten für das Ganze-Karte-GUI vor"""
        logger.info(f"\n🎮 BEREITE GUI-DATEN VOR")
        
        # Erstelle separate Listen für GUI
        gui_data = {}
        
        for priority, cards in self.priority_lists.items():
            if priority in ['critical_immediate', 'high_priority', 'medium_priority']:
                gui_cards = []
                
                for card in cards:
                    basename = card['basename']
                    card_status = card['card_status']
                    
                    gui_card = {
                        'basename': basename,
                        'priority': priority,
                        'main_image_path': card_status['image_paths']['main_image'],
                        'correction_needed': {
                            'qr_code': card_status['qr_code']['status'] != 'verfügbar',
                            'z_kaestchen': card_status['z_kaestchen']['quality_level'] not in ['perfekt', 'sehr_gut'],
                            'd_kaestchen': card_status['d_kaestchen']['quality_level'] not in ['perfekt', 'sehr_gut'],
                            'textfelder': card_status['textfelder']['status'] != 'verfügbar'
                        },
                        'current_data': {
                            'qr_zaehlernummer': card_status['qr_code'].get('zaehlernummer'),
                            'qr_zaehlerart': card_status['qr_code'].get('zaehlerart'),
                            'zaehlerstand': card_status['z_kaestchen'].get('zaehlerstand_text'),
                            'ablesedatum': card_status['d_kaestchen'].get('ablesedatum'),
                            'textfeld_zaehlerart': card_status['textfelder']['fields'].get('zaehlerart', {}).get('text'),
                            'textfeld_zaehlernummer': card_status['textfelder']['fields'].get('zaehlernummer', {}).get('text')
                        },
                        'estimated_minutes': card_status['estimated_work_minutes']
                    }
                    
                    gui_cards.append(gui_card)
                
                gui_data[priority] = gui_cards
        
        # Speichere GUI-Daten
        gui_file = self.complete_data_path / "ganze_karte_gui_data.json"
        with open(gui_file, 'w', encoding='utf-8') as f:
            json.dump(gui_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   🎮 GUI-Daten: {gui_file.name}")
        
        # Erstelle Prioritätslisten für GUI
        for priority, cards in gui_data.items():
            priority_file = self.priority_lists_path / f"{priority}_cards.json"
            with open(priority_file, 'w', encoding='utf-8') as f:
                json.dump(cards, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   📋 {priority}: {len(cards)} Karten → {priority_file.name}")
    
    def _create_summary(self) -> Dict:
        """Erstellt Zusammenfassung für Return"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'success': True,
            'analysis_duration_seconds': round(duration, 1),
            'statistics': self.statistics,
            'priority_summary': {
                priority: len(cards) for priority, cards in self.priority_lists.items()
            },
            'files_created': {
                'reports_dir': str(self.reports_path),
                'priority_lists_dir': str(self.priority_lists_path),
                'gui_data_dir': str(self.complete_data_path)
            },
            'next_steps': self._generate_next_actions()
        }
    
    def print_summary_report(self):
        """Druckt zusammenfassenden Bericht in die Konsole"""
        stats = self.statistics
        
        print("\n" + "="*60)
        print("POST-KORREKTUR ANALYSE - ZUSAMMENFASSUNG")
        print("="*60)
        
        print(f"📊 ANALYSIERTE KARTEN: {stats['total_cards']}")
        
        print(f"\n🎯 PRIORITÄTEN:")
        for priority, data in stats['priority_breakdown'].items():
            priority_config = self.module_config['prioritization'][priority]
            icon = priority_config['icon']
            print(f"   {icon} {priority.replace('_', ' ').upper()}: {data['count']} Karten ({data['percentage']}%)")
        
        print(f"\n✅ EXPORT-BEREITSCHAFT:")
        export_stats = stats['export_readiness']
        print(f"   Bereit: {export_stats['ready_count']} Karten ({export_stats['ready_percentage']}%)")
        print(f"   Verbleibend: {export_stats['remaining_count']} Karten")
        
        print(f"\n⏱️ ARBEITSAUFWAND:")
        work_stats = stats['work_estimates']
        print(f"   Geschätzte Restarbeit: {work_stats['total_hours']} Stunden")
        print(f"   Bei 8h/Tag: {work_stats['total_days']} Arbeitstage")
        
        print(f"\n🎯 NÄCHSTE SCHRITTE:")
        for action in self._generate_next_actions():
            print(f"   • {action}")


def main():
    """Hauptfunktion"""
    try:
        # Analyzer erstellen und ausführen
        analyzer = PostKorrekturAnalyzer()
        results = analyzer.analyze_complete_status()
        
        if 'error' in results:
            logger.error(f"❌ Analyse fehlgeschlagen: {results['error']}")
            return 1
        
        # Zusammenfassung ausgeben
        analyzer.print_summary_report()
        
        logger.info(f"\n✅ Post-Korrektur Analyse erfolgreich abgeschlossen!")
        logger.info(f"   Dauer: {results['analysis_duration_seconds']} Sekunden")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())