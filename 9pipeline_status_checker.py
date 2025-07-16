#!/usr/bin/env python3
"""
Dynamic Pipeline Analyzer - ERWEITERT für GUI-Korrektur-Tool
===========================================================
Analysiert ALLE PDFs und erstellt GUI-optimierte JSON-Ausgabe
für Kästchen-Korrektur-Interface.

Version: 1.2 - GUI Output
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class DynamicPipelineAnalyzer:
    """Config-basierte dynamische Pipeline-Analyse mit GUI-Output"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialisiert Analyzer mit Config-System"""
        # Lade Konfigurationen
        self.global_config, self.paths = self._load_configs(config_path)
        
        # Setup Pfade dynamisch aus Config
        self._setup_paths()
        
        logger.info(f"🔍 Dynamic Pipeline Analyzer initialisiert")
        logger.info(f"   Scanner-Ordner: {self.scanner_path}")
        logger.info(f"   Pipeline-Daten: {self.pipeline_data_path}")
        logger.info(f"   GUI-Output: {self.gui_output_path}")
        
    def _load_configs(self, config_path: Optional[str]) -> Tuple[Dict, Dict]:
        """Lädt Pipeline-Config"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "Config"
        else:
            config_path = Path(config_path)
        
        # Lade globale Pipeline-Config
        global_config_path = config_path / "pipeline_config.json"
        with open(global_config_path, 'r', encoding='utf-8') as f:
            global_config = json.load(f)
        
        # Extrahiere relevante Pfade
        paths = {
            'scanner_dir': global_config['paths']['scanner_dir'],
            'pipeline_data': global_config['paths']['pipeline_data'],
            'file_patterns': global_config['file_patterns']
        }
        
        return global_config, paths
    
    def _setup_paths(self):
        """Setup aller Pfade dynamisch aus Config"""
        base_path = Path(self.global_config['base_path'])
        
        # Scanner-Pfad (dynamischer Input)
        self.scanner_path = base_path / self.paths['scanner_dir']
        
        # Pipeline-Daten Pfad
        self.pipeline_data_path = base_path / self.paths['pipeline_data']
        
        # GUI-Output Pfad - NEU
        self.gui_output_path = self.pipeline_data_path / "09_pipeline_status"
        self.gui_output_path.mkdir(parents=True, exist_ok=True)
        
        # Alle Pipeline-Stufen-Pfade
        self.stage_paths = {
            'converted_images': self.pipeline_data_path / "01_converted",
            'yolo_metadata': self.pipeline_data_path / "02_yolo" / "metadata",
            'template_metadata': self.pipeline_data_path / "03_template" / "metadata",
            'textfield_metadata': self.pipeline_data_path / "04_textfelder" / "metadata",
            'extracted_metadata': self.pipeline_data_path / "05_extracted_fields" / "metadata",
            'qr_data': self.pipeline_data_path / "06_data_QR",
            'text_data': self.pipeline_data_path / "07_data_textfelder",
            'digit_data': self.pipeline_data_path / "08_data_kaestchen"
        }
        
        # File Patterns aus Config
        self.pdf_pattern = self.paths['file_patterns']['pdf_files']
        
    def analyze_complete_pipeline(self) -> Dict:
        """Vollständige Pipeline-Analyse mit GUI-Output"""
        
        print("🏭 DYNAMIC PIPELINE ANALYZER - GUI VERSION")
        print("="*60)
        
        # 1. Scanner-Ordner analysieren
        pdf_analysis = self._analyze_scanner_folder()
        
        if pdf_analysis['total_pdfs'] == 0:
            print("⚠️  KEINE PDFs im Scanner-Ordner gefunden!")
            return {'error': 'no_pdfs_found', 'scanner_path': str(self.scanner_path)}
        
        # 2. Pipeline-Flow analysieren  
        pipeline_flow = self._analyze_pipeline_flow(pdf_analysis['all_basenames'])
        
        # 3. Datenqualität analysieren
        data_quality = self._analyze_data_quality(pdf_analysis['all_basenames'])
        
        # 4. ERWEITERT: Detaillierte Kästchen-Analyse
        detailed_boxes = self._analyze_detailed_boxes(pdf_analysis['all_basenames'])
        
        # 5. GUI-JSON erstellen
        gui_data = self._create_gui_correction_data(
            pdf_analysis['all_basenames'], 
            detailed_boxes,
            data_quality
        )
        
        # 6. Korrektur-Bedarf bewerten
        correction_analysis = self._analyze_correction_needs(
            pdf_analysis['all_basenames'], 
            pipeline_flow, 
            data_quality
        )
        
        # 7. Training-Potenzial bewerten
        training_analysis = self._analyze_training_potential(data_quality, detailed_boxes)
        
        # 8. Produktions-Bereitschaft bewerten
        production_readiness = self._assess_production_readiness(
            pdf_analysis, 
            correction_analysis
        )
        
        # 9. SAVE GUI-JSON
        self._save_gui_data(gui_data)
        
        return {
            'scanner_analysis': pdf_analysis,
            'pipeline_flow': pipeline_flow,
            'data_quality': data_quality,
            'detailed_boxes': detailed_boxes,
            'gui_data_path': str(self.gui_output_path / "correction_gui_data.json"),
            'correction_analysis': correction_analysis,
            'training_analysis': training_analysis,
            'production_readiness': production_readiness,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_scanner_folder(self) -> Dict:
        """Analysiert Scanner-Ordner dynamisch"""
        
        print(f"📁 SCANNER-ORDNER: {self.scanner_path}")
        
        if not self.scanner_path.exists():
            return {
                'error': 'scanner_folder_not_found',
                'path': str(self.scanner_path),
                'total_pdfs': 0,
                'all_basenames': set()
            }
        
        # PDFs dynamisch finden
        pdf_files = list(self.scanner_path.glob(self.pdf_pattern))
        total_pdfs = len(pdf_files)
        all_basenames = {pdf.stem for pdf in pdf_files}
        
        print(f"   📊 Gefundene PDFs: {total_pdfs}")
        if total_pdfs > 0:
            print(f"   📝 Beispiele: {list(all_basenames)[:3]}...")
        
        return {
            'total_pdfs': total_pdfs,
            'all_basenames': all_basenames,
            'pdf_files': [str(pdf) for pdf in pdf_files],
            'scanner_path': str(self.scanner_path)
        }
    
    def _analyze_pipeline_flow(self, all_basenames: Set[str]) -> Dict:
        """Analysiert Pipeline-Durchlauf aller Basenames"""
        
        print(f"\n🔄 PIPELINE-FLOW-ANALYSE")
        
        flow_analysis = {}
        
        # Stufe 1: PDF → PNG Konvertierung
        converted_images = self._count_converted_images(all_basenames)
        flow_analysis['converted'] = converted_images
        
        # Stufe 2: YOLO Detection
        yolo_processed = self._count_yolo_processed(all_basenames)
        flow_analysis['yolo_detected'] = yolo_processed
        
        # Stufe 3: Template Matching
        template_processed = self._count_template_processed(all_basenames)
        flow_analysis['template_matched'] = template_processed
        
        # Stufe 4: Textfield Calculation
        textfield_processed = self._count_textfield_processed(all_basenames)
        flow_analysis['textfield_calculated'] = textfield_processed
        
        # Stufe 5: Field Extraction
        extracted_processed = self._count_extracted_processed(all_basenames)
        flow_analysis['fields_extracted'] = extracted_processed
        
        # Print Flow-Statistiken
        total = len(all_basenames)
        print(f"   1️⃣ Konvertiert: {len(converted_images['successful'])}/{total}")
        print(f"   2️⃣ YOLO erkannt: {len(yolo_processed['successful'])}/{total}")
        print(f"   3️⃣ Template matched: {len(template_processed['successful'])}/{total}")
        print(f"   4️⃣ Textfelder berechnet: {len(textfield_processed['successful'])}/{total}")
        print(f"   5️⃣ Felder extrahiert: {len(extracted_processed['successful'])}/{total}")
        
        return flow_analysis
    
    def _analyze_data_quality(self, all_basenames: Set[str]) -> Dict:
        """Analysiert finale Datenqualität"""
        
        print(f"\n🎯 DATENQUALITÄTS-ANALYSE")
        
        # Finale Daten laden
        qr_data = self._load_qr_data()
        text_data = self._load_text_data()
        digit_data = self._load_digit_data()
        
        # QR-Code Qualität
        qr_quality = self._analyze_qr_quality(qr_data, all_basenames)
        
        # Text-OCR Qualität  
        text_quality = self._analyze_text_quality(text_data, all_basenames)
        
        # Kästchen-OCR Qualität - ERWEITERT
        digit_quality = self._analyze_digit_quality_enhanced(digit_data, all_basenames)
        
        print(f"   📱 QR-Codes: {qr_quality['success_rate']:.1f}%")
        print(f"   📝 Text-OCR: {text_quality['success_rate']:.1f}%")
        print(f"   🔢 Kästchen-OCR: {digit_quality['box_level_success_rate']:.1f}%")
        
        return {
            'qr_quality': qr_quality,
            'text_quality': text_quality,
            'digit_quality': digit_quality
        }
    
    def _analyze_detailed_boxes(self, all_basenames: Set[str]) -> Dict:
        """NEU: Detaillierte Analyse aller einzelnen Kästchen"""
        
        print(f"\n🔍 DETAILLIERTE KÄSTCHEN-ANALYSE")
        
        digit_data = self._load_digit_data()
        
        box_stats = {
            'total_boxes': 0,
            'z_boxes': 0,
            'd_boxes': 0,
            'status_counts': defaultdict(int),
            'position_analysis': {
                'z_positions': {str(i): defaultdict(int) for i in range(9)},
                'd_positions': {str(i): defaultdict(int) for i in range(4)}
            },
            'problematic_boxes': [],
            'cards_with_details': {}
        }
        
        for basename in all_basenames:
            if basename in digit_data:
                card_data = digit_data[basename]
                
                # Analysiere Z-Kästchen
                if 'z_kaestchen' in card_data:
                    for box in card_data['z_kaestchen']:
                        box_stats['total_boxes'] += 1
                        box_stats['z_boxes'] += 1
                        status = box.get('status', 'fehler')
                        box_stats['status_counts'][status] += 1
                        
                        # Position-spezifische Statistiken
                        pos = str(box.get('position', 0))
                        box_stats['position_analysis']['z_positions'][pos][status] += 1
                        
                        # Problematische Kästchen sammeln
                        if status in ['unsicher', 'fehler']:
                            box_stats['problematic_boxes'].append({
                                'basename': basename,
                                'type': 'z',
                                'position': box.get('position'),
                                'status': status,
                                'confidence': box.get('confidence', 0.0),
                                'predicted': box.get('predicted', 'x'),
                                'image_path': box.get('image_path')
                            })
                
                # Analysiere D-Kästchen
                if 'd_kaestchen' in card_data:
                    for box in card_data['d_kaestchen']:
                        box_stats['total_boxes'] += 1
                        box_stats['d_boxes'] += 1
                        status = box.get('status', 'fehler')
                        box_stats['status_counts'][status] += 1
                        
                        # Position-spezifische Statistiken
                        pos = str(box.get('position', 0))
                        box_stats['position_analysis']['d_positions'][pos][status] += 1
                        
                        # Problematische Kästchen sammeln
                        if status in ['unsicher', 'fehler']:
                            box_stats['problematic_boxes'].append({
                                'basename': basename,
                                'type': 'd',
                                'position': box.get('position'),
                                'status': status,
                                'confidence': box.get('confidence', 0.0),
                                'predicted': box.get('predicted', 'x'),
                                'image_path': box.get('image_path')
                            })
                
                # Speichere Karten-Details
                box_stats['cards_with_details'][basename] = {
                    'problematic_boxes_count': card_data.get('problematic_boxes_count', 0),
                    'average_confidence_z': card_data.get('average_confidence_z', 0.0),
                    'min_confidence_z': card_data.get('min_confidence_z', 0.0),
                    'needs_manual_review': card_data.get('needs_manual_review', True),
                    'zaehlerstand_text': card_data.get('zaehlerstand_text', ''),
                    'ablesedatum': card_data.get('ablesedatum', '')
                }
        
        problematic_count = len(box_stats['problematic_boxes'])
        success_rate = ((box_stats['total_boxes'] - problematic_count) / box_stats['total_boxes'] * 100) if box_stats['total_boxes'] > 0 else 0
        
        print(f"   📦 Gesamte Kästchen: {box_stats['total_boxes']}")
        print(f"   🔢 Problematische: {problematic_count} ({100-success_rate:.1f}%)")
        print(f"   ✅ Erfolgsrate: {success_rate:.1f}%")
        
        return dict(box_stats)
    
    def _create_gui_correction_data(self, all_basenames: Set[str], detailed_boxes: Dict, data_quality: Dict) -> Dict:
        """NEU: Erstellt GUI-optimierte JSON für Korrektur-Interface"""
        
        print(f"\n🖥️ GUI-DATEN ERSTELLEN")
        
        digit_data = self._load_digit_data()
        
        gui_data = {
            'metadata': {
                'created_timestamp': datetime.now().isoformat(),
                'total_cards': len(all_basenames),
                'total_boxes': detailed_boxes['total_boxes'],
                'problematic_boxes_count': len(detailed_boxes['problematic_boxes']),
                'version': '1.0'
            },
            'cards': [],
            'statistics': {
                'box_level': dict(detailed_boxes['status_counts']),
                'position_analysis': detailed_boxes['position_analysis'],
                'priority_counts': {
                    'high_priority': 0,    # fehler + <0.5 confidence
                    'medium_priority': 0,  # unsicher + 0.5-0.7 confidence  
                    'low_priority': 0      # leicht_unsicher + 0.7-0.8 confidence
                }
            },
            'correction_queue': {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': []
            }
        }
        
        # Erstelle Karten-Liste für GUI
        for basename in all_basenames:
            if basename in digit_data:
                card_data = digit_data[basename]
                
                # Sammle problematische Kästchen für diese Karte
                card_problematic_boxes = [
                    box for box in detailed_boxes['problematic_boxes'] 
                    if box['basename'] == basename
                ]
                
                # Priorität bestimmen
                priority = self._determine_card_priority(card_problematic_boxes)
                
                gui_card = {
                    'basename': basename,
                    'priority': priority,
                    'needs_review': len(card_problematic_boxes) > 0,
                    'zaehlerstand_text': card_data.get('zaehlerstand_text', ''),
                    'ablesedatum': card_data.get('ablesedatum', ''),
                    'problematic_boxes_count': len(card_problematic_boxes),
                    'average_confidence': card_data.get('average_confidence_z', 0.0),
                    'min_confidence': card_data.get('min_confidence_z', 0.0),
                    'problematic_boxes': card_problematic_boxes,
                    'z_kaestchen': card_data.get('z_kaestchen', []),
                    'd_kaestchen': card_data.get('d_kaestchen', []),
                    'image_path': f"pipeline_data/01_converted/{basename}.png"  # Für GUI
                }
                
                gui_data['cards'].append(gui_card)
                
                # Zu Korrektur-Queue hinzufügen
                if len(card_problematic_boxes) > 0:
                    gui_data['correction_queue'][f'{priority}_priority'].append(gui_card)
                    gui_data['statistics']['priority_counts'][f'{priority}_priority'] += 1
        
        # Sortiere Karten nach Priorität
        gui_data['cards'].sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2, 'none': 3}.get(x['priority'], 3))
        
        print(f"   📋 GUI-Karten erstellt: {len(gui_data['cards'])}")
        print(f"   🔥 Hohe Priorität: {gui_data['statistics']['priority_counts']['high_priority']}")
        print(f"   🔶 Mittlere Priorität: {gui_data['statistics']['priority_counts']['medium_priority']}")
        print(f"   ⚠️ Niedrige Priorität: {gui_data['statistics']['priority_counts']['low_priority']}")
        
        return gui_data
    
    def _determine_card_priority(self, problematic_boxes: List[Dict]) -> str:
        """Bestimmt Priorität einer Karte basierend auf problematischen Kästchen"""
        if not problematic_boxes:
            return 'none'
        
        # Zähle verschiedene Problemtypen
        fehler_count = sum(1 for box in problematic_boxes if box['status'] == 'fehler')
        low_conf_count = sum(1 for box in problematic_boxes if box['confidence'] < 0.5)
        unsicher_count = sum(1 for box in problematic_boxes if box['status'] == 'unsicher')
        
        # Priorität bestimmen
        if fehler_count > 0 or low_conf_count > 2:
            return 'high'
        elif unsicher_count > 2 or low_conf_count > 0:
            return 'medium'
        else:
            return 'low'
    
    def _save_gui_data(self, gui_data: Dict):
        """Speichert GUI-Daten in verschiedenen Formaten"""
        
        # Haupt-GUI-JSON
        gui_file = self.gui_output_path / "correction_gui_data.json"
        with open(gui_file, 'w', encoding='utf-8') as f:
            json.dump(gui_data, f, indent=2, ensure_ascii=False)
        
        # Kompakte Version für schnelle Übersicht
        summary_file = self.gui_output_path / "correction_summary.json"
        summary = {
            'timestamp': gui_data['metadata']['created_timestamp'],
            'total_cards': gui_data['metadata']['total_cards'],
            'cards_needing_review': len([c for c in gui_data['cards'] if c['needs_review']]),
            'priority_counts': gui_data['statistics']['priority_counts'],
            'top_problematic_cards': [
                {
                    'basename': card['basename'],
                    'priority': card['priority'],
                    'problematic_count': card['problematic_boxes_count'],
                    'min_confidence': card['min_confidence']
                }
                for card in gui_data['cards'][:20] if card['needs_review']
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 GUI-DATEIEN GESPEICHERT:")
        print(f"   📋 Haupt-GUI-Daten: {gui_file}")
        print(f"   📊 Zusammenfassung: {summary_file}")
    
    # ===== BESTEHENDE METHODEN (gekürzt) =====
    
    def _analyze_digit_quality_enhanced(self, digit_data: Dict, all_basenames: Set[str]) -> Dict:
        """Erweiterte Kästchen-OCR Qualitätsanalyse"""
        processed = len(digit_data)
        total_boxes = 0
        successful_boxes = 0
        problematic_cards = set()
        
        for basename, data in digit_data.items():
            if isinstance(data, dict):
                # Zähle Kästchen-Level Erfolg
                z_boxes = data.get('z_kaestchen', [])
                d_boxes = data.get('d_kaestchen', [])
                
                for box in z_boxes + d_boxes:
                    total_boxes += 1
                    if box.get('status') == 'ok':
                        successful_boxes += 1
                
                # Karten-Level Status
                if data.get('needs_manual_review', True):
                    problematic_cards.add(basename)
        
        not_processed = all_basenames - digit_data.keys()
        problematic_cards.update(not_processed)
        
        return {
            'processed': processed,
            'successful_cards': processed - len(problematic_cards),
            'problematic_cards': problematic_cards,
            'card_success_rate': ((processed - len(problematic_cards)) / len(all_basenames) * 100) if all_basenames else 0,
            'total_boxes': total_boxes,
            'successful_boxes': successful_boxes,
            'box_level_success_rate': (successful_boxes / total_boxes * 100) if total_boxes > 0 else 0
        }
    
    # ===== HELPER FUNCTIONS (vereinfacht) =====
    
    def _load_qr_data(self) -> Dict[str, Dict]:
        """Lädt QR-Code Daten"""
        qr_data = {}
        qr_path = self.stage_paths['qr_data']
        
        if not qr_path.exists():
            return qr_data
        
        for qr_file in qr_path.glob("*_qr.json"):
            basename = "_".join(qr_file.stem.split("_")[:-1])
            try:
                with open(qr_file, 'r', encoding='utf-8') as f:
                    qr_data[basename] = json.load(f)
            except Exception as e:
                logger.debug(f"QR-Datei Fehler {qr_file}: {e}")
        
        return qr_data
    
    def _load_text_data(self) -> Dict[str, Dict]:
        """Lädt Text-OCR Daten"""
        text_data = {}
        text_path = self.stage_paths['text_data']
        
        if not text_path.exists():
            return text_data
        
        # Sammle alle Basenames aus beiden Dateitypen
        all_files = list(text_path.glob("*_zaehlerart.json")) + list(text_path.glob("*_zaehlernummer.json"))
        basenames = set()
        for file in all_files:
            if "_zaehlerart" in file.name:
                basename = file.stem.replace("_zaehlerart", "")
            elif "_zaehlernummer" in file.name:
                basename = file.stem.replace("_zaehlernummer", "")
            else:
                continue
            basenames.add(basename)
        
        # Für jeden Basename beide Dateien laden
        for basename in basenames:
            zaehlerart_file = text_path / f"{basename}_zaehlerart.json"
            zaehlernummer_file = text_path / f"{basename}_zaehlernummer.json"
            
            combined_data = {}
            
            # Lade Zählerart
            if zaehlerart_file.exists():
                try:
                    with open(zaehlerart_file, 'r', encoding='utf-8') as f:
                        combined_data['zaehlerart'] = json.load(f)
                except Exception as e:
                    logger.debug(f"Zählerart-Datei Fehler {zaehlerart_file}: {e}")
                    combined_data['zaehlerart'] = {'status': 'fehler'}
            else:
                combined_data['zaehlerart'] = {'status': 'fehler'}
            
            # Lade Zählernummer 
            if zaehlernummer_file.exists():
                try:
                    with open(zaehlernummer_file, 'r', encoding='utf-8') as f:
                        combined_data['zaehlernummer'] = json.load(f)
                except Exception as e:
                    logger.debug(f"Zählernummer-Datei Fehler {zaehlernummer_file}: {e}")
                    combined_data['zaehlernummer'] = {'status': 'fehler'}
            else:
                combined_data['zaehlernummer'] = {'status': 'fehler'}
            
            text_data[basename] = combined_data
        
        return text_data
    
    def _load_digit_data(self) -> Dict[str, Dict]:
        """Lädt erweiterte Kästchen-OCR Daten"""
        digit_data = {}
        digit_path = self.stage_paths['digit_data']
        
        if not digit_path.exists():
            return digit_data
        
        for digit_file in digit_path.glob("*_kaestchen.json"):
            basename = "_".join(digit_file.stem.split("_")[:-1])
            try:
                with open(digit_file, 'r', encoding='utf-8') as f:
                    digit_data[basename] = json.load(f)
            except Exception as e:
                logger.debug(f"Kästchen-Datei Fehler {digit_file}: {e}")
        
        return digit_data
    
    # ===== SIMPLIFIED HELPER METHODS =====
    
    def _count_converted_images(self, basenames: Set[str]) -> Dict:
        converted_path = self.stage_paths['converted_images']
        if not converted_path.exists():
            return {'successful': set(), 'missing': basenames}
        png_files = list(converted_path.glob("*.png"))
        converted_basenames = {png.stem.rsplit('_', 1)[0] for png in png_files}
        successful = basenames & converted_basenames
        missing = basenames - converted_basenames
        return {'successful': successful, 'missing': missing}
    
    def _count_yolo_processed(self, basenames: Set[str]) -> Dict:
        yolo_path = self.stage_paths['yolo_metadata']
        if not yolo_path.exists():
            return {'successful': set(), 'missing': basenames}
        yolo_files = list(yolo_path.glob("*_yolo.json"))
        yolo_basenames = {"_".join(f.stem.split("_")[:-1]) for f in yolo_files}
        successful = basenames & yolo_basenames
        missing = basenames - yolo_basenames
        return {'successful': successful, 'missing': missing}
    
    def _count_template_processed(self, basenames: Set[str]) -> Dict:
        template_path = self.stage_paths['template_metadata']
        if not template_path.exists():
            return {'successful': set(), 'missing': basenames}
        template_files = list(template_path.glob("*_combined.json"))
        template_basenames = {"_".join(f.stem.split("_")[:-1]) for f in template_files}
        successful = basenames & template_basenames
        missing = basenames - template_basenames
        return {'successful': successful, 'missing': missing}
    
    def _count_textfield_processed(self, basenames: Set[str]) -> Dict:
        textfield_path = self.stage_paths['textfield_metadata']
        if not textfield_path.exists():
            return {'successful': set(), 'missing': basenames}
        textfield_files = list(textfield_path.glob("*_textfelder.json"))
        textfield_basenames = {"_".join(f.stem.split("_")[:-1]) for f in textfield_files}
        successful = basenames & textfield_basenames
        missing = basenames - textfield_basenames
        return {'successful': successful, 'missing': missing}
    
    def _count_extracted_processed(self, basenames: Set[str]) -> Dict:
        extracted_path = self.stage_paths['extracted_metadata']
        if not extracted_path.exists():
            return {'successful': set(), 'missing': basenames}
        extracted_files = list(extracted_path.glob("*_extracted.json"))
        extracted_basenames = {"_".join(f.stem.split("_")[:-2]) for f in extracted_files}
        successful = basenames & extracted_basenames
        missing = basenames - extracted_basenames
        return {'successful': successful, 'missing': missing}
    
    def _analyze_qr_quality(self, qr_data: Dict, all_basenames: Set[str]) -> Dict:
        processed = len(qr_data)
        successful = 0
        failed_cards = set()
        
        for basename, data in qr_data.items():
            qr_content = data.get('qr', '').strip()
            if qr_content and qr_content != "FEHLT_MANUELL":
                successful += 1
            else:
                failed_cards.add(basename)
        
        not_processed = all_basenames - qr_data.keys()
        failed_cards.update(not_processed)
        
        return {
            'processed': processed,
            'successful': successful,
            'failed': len(failed_cards),
            'success_rate': (successful / len(all_basenames) * 100) if all_basenames else 0,
            'failed_cards': failed_cards
        }
    
    def _analyze_text_quality(self, text_data: Dict, all_basenames: Set[str]) -> Dict:
        processed = len(text_data)
        successful = 0
        problematic_cards = set()
        
        for basename, data in text_data.items():
            art_status = data.get('zaehlerart', {}).get('status', 'fehler')
            num_status = data.get('zaehlernummer', {}).get('status', 'fehler')
            
            if art_status == 'ok' and num_status == 'ok':
                successful += 1
            else:
                problematic_cards.add(basename)
        
        not_processed = all_basenames - text_data.keys()
        problematic_cards.update(not_processed)
        
        return {
            'processed': processed,
            'successful': successful,
            'problematic': len(problematic_cards),
            'success_rate': (successful / len(all_basenames) * 100) if all_basenames else 0,
            'problematic_cards': problematic_cards
        }
    
    def _analyze_correction_needs(self, all_basenames: Set[str], pipeline_flow: Dict, data_quality: Dict) -> Dict:
        print(f"\n🔧 KORREKTUR-ANALYSE")
        
        pipeline_failures = set()
        final_extracted = pipeline_flow['fields_extracted']['successful']
        pipeline_failures = all_basenames - final_extracted
        
        qr_problems = data_quality['qr_quality']['failed_cards']
        text_problems = data_quality['text_quality']['problematic_cards']
        digit_problems = data_quality['digit_quality']['problematic_cards']
        
        critical_immediate = pipeline_failures | digit_problems
        high_priority = qr_problems & text_problems
        medium_priority = text_problems - qr_problems
        low_priority = qr_problems - text_problems
        
        total_corrections = len(critical_immediate | high_priority | medium_priority | low_priority)
        
        print(f"   🚨 KRITISCH: {len(critical_immediate)} Karten")
        print(f"   ⚠️  HOCH: {len(high_priority)} Karten")
        print(f"   📝 MITTEL: {len(medium_priority)} Karten") 
        print(f"   ℹ️  NIEDRIG: {len(low_priority)} Karten")
        print(f"   📊 GESAMT Korrekturen: {total_corrections}/{len(all_basenames)}")
        
        return {
            'critical_immediate': critical_immediate,
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority,
            'total_corrections_needed': total_corrections
        }
    
    def _analyze_training_potential(self, data_quality: Dict, detailed_boxes: Dict) -> Dict:
        problematic_boxes_count = len(detailed_boxes['problematic_boxes'])
        
        return {
            'problematic_boxes_count': problematic_boxes_count,
            'estimated_training_samples': problematic_boxes_count,
            'keras_improvement_potential': problematic_boxes_count > 50,
            'priority_distribution': {
                'high_priority_samples': len([b for b in detailed_boxes['problematic_boxes'] if b['status'] == 'fehler']),
                'medium_priority_samples': len([b for b in detailed_boxes['problematic_boxes'] if b['status'] == 'unsicher'])
            }
        }
    
    def _assess_production_readiness(self, pdf_analysis: Dict, correction_analysis: Dict) -> Dict:
        total_pdfs = pdf_analysis['total_pdfs']
        critical_issues = len(correction_analysis['critical_immediate'])
        
        ready = critical_issues == 0
        completeness_rate = ((total_pdfs - critical_issues) / total_pdfs * 100) if total_pdfs > 0 else 0
        
        return {
            'ready_for_production': ready,
            'critical_blocking_issues': critical_issues,
            'completeness_rate': completeness_rate,
            'total_pdfs': total_pdfs
        }
    
    def print_summary_report(self, analysis: Dict):
        """Druckt erweiterten Zusammenfassungs-Bericht"""
        
        scanner = analysis['scanner_analysis']
        detailed = analysis['detailed_boxes']
        correction = analysis['correction_analysis']
        production = analysis['production_readiness']
        training = analysis['training_analysis']
        
        print(f"\n🎯 PIPELINE-ZUSAMMENFASSUNG")
        print("="*60)
        
        print(f"📁 Scanner-Ordner: {scanner['total_pdfs']} PDFs")
        print(f"🔢 Gesamte Kästchen: {detailed['total_boxes']}")
        print(f"🔧 Problematische Kästchen: {len(detailed['problematic_boxes'])}")
        
        if production['ready_for_production']:
            print("✅ BEREIT FÜR PRODUKTION!")
        else:
            print(f"🚨 {production['critical_blocking_issues']} KRITISCHE PROBLEME")
        
        print(f"📊 Vollständigkeit: {production['completeness_rate']:.1f}%")
        print(f"🔧 Korrekturen nötig: {correction['total_corrections_needed']}")
        print(f"🧠 Training-Potenzial: {training['estimated_training_samples']} Kästchen")
        
        print(f"\n💾 GUI-Daten gespeichert in: {analysis['gui_data_path']}")


def main():
    """Hauptfunktion"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        analyzer = DynamicPipelineAnalyzer()
        analysis = analyzer.analyze_complete_pipeline()
        
        if 'error' in analysis:
            print(f"❌ Fehler: {analysis['error']}")
            return 1
        
        analyzer.print_summary_report(analysis)
        
        # Speichere vollständige Analyse
        output_file = Path(r"C:\ZaehlerkartenV2\pipeline_data\09_pipeline_status\dynamic_pipeline_analysis_gui.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert sets to lists for JSON
            def convert_sets(obj):
                if isinstance(obj, set):
                    return list(obj)
                raise TypeError
            
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=convert_sets)
        
        print(f"\n💾 Vollständige Analyse gespeichert: {output_file}")
        return 0
        
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())