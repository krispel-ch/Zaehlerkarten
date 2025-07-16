#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kästchen-Korrektur GUI - Modul 10
==================================
GUI-Tool zur manuellen Korrektur problematischer Kästchen-OCR Ergebnisse.
Lädt fehlerhafte Kästchen, zeigt sie zur Korrektur an, speichert Korrekturen
in Pipeline-Ergebnisse und sammelt Training-Daten für Keras-Verbesserung.

Version: 1.0
Autor: Oliver Krispel + KI-System  
Datum: 2025-07-15
"""

import json
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from PIL import Image, ImageTk

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaestchenKorrekturGUI:
    """
    GUI für manuelle Korrektur problematischer Kästchen-OCR Ergebnisse
    
    Workflow:
    1. Lädt problematische Kästchen aus Pipeline-Status
    2. Zeigt Kästchen-Bild + OCR-Vorhersage zur Korrektur
    3. Speichert Korrekturen in Pipeline-Ergebnisse (08_data_kaestchen)
    4. Sammelt Training-Daten (KI/Piplinetraining)
    5. Bietet inkrementelles Keras-Training an
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert das Korrektur-GUI
        
        Args:
            config_path: Pfad zum Config-Verzeichnis (optional)
        """
        # Lade Konfigurationen
        self.global_config, self.module_config = self._load_configs(config_path)
        
        # Setup Pfade
        self._setup_paths()
        
        # Setup Logging
        self._setup_logging()
        
        # Prüfe auf vorherigen Fortschritt
        self.previous_progress = self._load_previous_progress()
        
        # Lade problematische Kästchen
        self.problematic_boxes = self._load_problematic_boxes()
        
        # GUI State
        self.current_index = 0
        self.corrections_made = 0
        self.training_queue = []
        self.current_image = None
        self.current_box_data = None
        
        # Lade vorherigen Fortschritt falls vorhanden
        if self.previous_progress and self._ask_resume_progress():
            self.current_index = min(self.previous_progress.get('current_index', 0), len(self.problematic_boxes) - 1)
            self.corrections_made = self.previous_progress.get('corrections_made', 0)
            logger.info(f"📂 Fortschritt geladen: Index {self.current_index}, {self.corrections_made} Korrekturen")
        
        # Statistiken für Analyse
        self.correction_statistics = {
            'total_corrections': 0,
            'true_positives': 0,    # Wirklich falsch vorhergesagt
            'false_positives': 0,   # Richtig vorhergesagt, aber als problematisch markiert
            'empty_fields_found': 0, # Leere Felder die OCR übersehen hat
            'confidence_analysis': [], # [(original_conf, was_correct)]
            'correction_details': []   # Detaillierte Korrektur-Info
        }
        
        # Erstelle GUI
        self._create_gui()
        
        # Setup Keyboard-Navigation
        self._setup_keyboard_navigation()
        
        # Zeige erstes Kästchen
        if self.problematic_boxes:
            self._display_current_box()
        else:
            messagebox.showinfo("Info", "Keine problematischen Kästchen gefunden!")
            self.root.quit()
        
        logger.info(f"✅ Korrektur-GUI initialisiert: {len(self.problematic_boxes)} Kästchen zu korrigieren")
        
        # Prüfe bereits vorhandene Training-Daten
        existing_training_images = len(list(self.training_images_path.glob("*.png")))
        if existing_training_images > 0:
            logger.info(f"📊 Vorhandene Training-Daten: {existing_training_images} Bilder")
    
    def _load_previous_progress(self) -> Optional[Dict]:
        """Lädt vorherigen Fortschritt falls vorhanden"""
        progress_file = self.problem_data_path / "korrektur_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Fehler beim Laden des Fortschritts: {e}")
        return None
    
    def _ask_resume_progress(self) -> bool:
        """Fragt ob vorheriger Fortschritt fortgesetzt werden soll"""
        if not self.previous_progress:
            return False
        
        # Erstelle GUI Root temporär für MessageBox
        temp_root = tk.Tk()
        temp_root.withdraw()  # Verstecke Fenster
        
        current_index = self.previous_progress.get('current_index', 0)
        corrections_made = self.previous_progress.get('corrections_made', 0)
        timestamp = self.previous_progress.get('timestamp', 'Unbekannt')
        
        response = messagebox.askyesno(
            "Fortschritt gefunden",
            f"Vorheriger Fortschritt gefunden:\n\n"
            f"• Letzter Stand: Kästchen {current_index + 1}/{len(self.problematic_boxes)}\n"
            f"• Korrekturen: {corrections_made}\n"
            f"• Zeitpunkt: {timestamp[:19].replace('T', ' ')}\n\n"
            f"Möchten Sie dort fortfahren?\n\n"
            f"[Ja] = Fortsetzen | [Nein] = Von vorne beginnen",
            icon='question'
        )
        
        temp_root.destroy()
        return response
    
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
        module_config_path = config_path / "korrektur_kaestchen_config.json"
        with open(module_config_path, 'r', encoding='utf-8') as f:
            module_config = json.load(f)
        
        return global_config, module_config
    
    def _setup_paths(self):
        """Richtet alle notwendigen Pfade ein"""
        base_path = Path(self.global_config['base_path'])
        pipeline_data = base_path / self.global_config['paths']['pipeline_data']
        
        # Input Pfade
        self.problem_data_path = pipeline_data / self.module_config['paths']['problem_data_subdir']
        self.problem_data_file = self.problem_data_path / self.module_config['paths']['problem_data_file']
        self.kaestchen_z_path = pipeline_data / self.module_config['paths']['kaestchen_z_subdir']
        self.kaestchen_d_path = pipeline_data / self.module_config['paths']['kaestchen_d_subdir']
        
        # Output Pfade
        self.pipeline_results_path = pipeline_data / self.module_config['paths']['pipeline_results_subdir']
        self.training_base_path = base_path / self.module_config['paths']['training_base_dir']
        self.training_images_path = self.training_base_path / self.module_config['paths']['training_images_subdir']
        self.training_json_path = self.training_base_path / self.module_config['paths']['training_json_subdir']
        self.training_labels_path = base_path / self.module_config['paths']['training_labels_dir']
        
        # Keras Training Script
        self.keras_trainer_path = base_path / self.module_config['paths']['keras_trainer_script']
        
        # Erstelle Output-Verzeichnisse
        self.training_images_path.mkdir(parents=True, exist_ok=True)
        self.training_json_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup Logging falls aktiviert"""
        if self.module_config['logging']['enabled']:
            log_level = getattr(logging, self.module_config['logging']['log_level'])
            log_file = Path(self.module_config['logging']['log_file'])
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def _load_problematic_boxes(self) -> List[Dict]:
        """Lädt problematische Kästchen aus Pipeline-Status"""
        if not self.problem_data_file.exists():
            logger.error(f"Problem-Datei nicht gefunden: {self.problem_data_file}")
            return []
        
        try:
            with open(self.problem_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extrahiere problematische Kästchen
            problematic_boxes = []
            
            # Korrekte Struktur: data['cards'] (nicht 'gui_cards')
            if 'cards' in data:
                for card_data in data['cards']:
                    basename = card_data['basename']
                    
                    # Problematische Kästchen direkt aus 'problematic_boxes'
                    if 'problematic_boxes' in card_data:
                        for box_info in card_data['problematic_boxes']:
                            # Extrahiere Dateiname aus image_path
                            image_path = Path(box_info['image_path'])
                            image_filename = image_path.name
                            
                            box_entry = {
                                'basename': basename,
                                'type': box_info['type'],
                                'box_index': box_info['position'],  # 'position' nicht 'box_index'
                                'predicted_digit': box_info['predicted'],  # 'predicted' nicht 'predicted_digit'
                                'confidence': box_info['confidence'],
                                'status': box_info['status'],
                                'image_filename': image_filename
                            }
                            problematic_boxes.append(box_entry)
            
            logger.info(f"📊 Gefunden: {len(problematic_boxes)} problematische Kästchen")
            return problematic_boxes
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der problematischen Kästchen: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_gui(self):
        """Erstellt das Hauptfenster"""
        self.root = tk.Tk()
        self.root.title(self.module_config['gui_settings']['window_title'])
        self.root.geometry(f"{self.module_config['gui_settings']['window_width']}x{self.module_config['gui_settings']['window_height']}")
        
        # Hauptframe
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfiguriere Grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Linke Seite - Bild
        image_frame = ttk.LabelFrame(main_frame, text="Kästchen-Bild", padding="5")
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="Kein Bild geladen")
        self.image_label.pack()
        
        # Rechte Seite - Information und Korrektur
        info_frame = ttk.LabelFrame(main_frame, text="Korrektur", padding="5")
        info_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=(10, 0))
        info_frame.columnconfigure(1, weight=1)
        
        # Datei-Info
        ttk.Label(info_frame, text="Datei:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.file_label = ttk.Label(info_frame, text="", foreground="blue")
        self.file_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Typ:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.type_label = ttk.Label(info_frame, text="")
        self.type_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="OCR-Vorhersage:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.prediction_label = ttk.Label(info_frame, text="", font=("Arial", 14, "bold"))
        self.prediction_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Confidence:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.confidence_label = ttk.Label(info_frame, text="")
        self.confidence_label.grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Separator
        ttk.Separator(info_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Korrektur-Hinweise
        ttk.Label(info_frame, text="Tastatur-Navigation:", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        hints_frame = ttk.Frame(info_frame)
        hints_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(hints_frame, text="• 0-9: Ziffer eingeben", font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(hints_frame, text="• X: Leeres/durchgestrichenes Feld", font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(hints_frame, text="• B: Zurück zum vorherigen", font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(hints_frame, text="• ESC: Beenden", font=("Arial", 10)).pack(anchor=tk.W)
        
        # Aktuelle Eingabe anzeigen
        ttk.Label(info_frame, text="Eingabe:", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky=tk.W, pady=5)
        self.input_display = ttk.Label(info_frame, text="[Warten auf Eingabe...]", 
                                     font=("Arial", 20, "bold"), foreground="red")
        self.input_display.grid(row=7, column=1, sticky=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(info_frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=20)
        
        # Training Button (nur anzeigen wenn konfiguriert)
        if self.module_config['training_settings']['show_training_button']:
            self.training_button = ttk.Button(button_frame, text="🧠 Training starten", 
                                            command=self._start_keras_training, state='disabled')
            self.training_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="📊 Statistiken", command=self._show_statistics).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="❌ Beenden", command=self._on_closing).pack(side=tk.LEFT)
        
        # Navigation (vereinfacht)
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=1, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        nav_frame.columnconfigure(2, weight=1)
        
        ttk.Button(nav_frame, text="< Zurück (B)", command=self._previous_box).grid(row=0, column=0, padx=(0, 20))
        ttk.Button(nav_frame, text="Weiter >", command=self._next_box).grid(row=0, column=1, padx=(0, 20))
        
        # Springe zu (behalten für Navigation)
        ttk.Label(nav_frame, text="Springe zu #:").grid(row=0, column=3, padx=(20, 5))
        self.jump_var = tk.StringVar()
        jump_entry = ttk.Entry(nav_frame, textvariable=self.jump_var, width=6)
        jump_entry.grid(row=0, column=4, padx=(0, 5))
        jump_entry.bind('<Return>', lambda e: self._jump_to_index())
        ttk.Button(nav_frame, text="Los", command=self._jump_to_index).grid(row=0, column=5)
        
        # Status
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.progress_label = ttk.Label(status_frame, text="")
        self.progress_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(status_frame, text="")
        self.status_label.pack(side=tk.RIGHT)
        
        # Exit Handler
        if self.module_config['gui_settings']['confirm_before_exit']:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_keyboard_navigation(self):
        """Setup Keyboard-Navigation für direkte Eingabe"""
        if self.module_config['gui_settings']['keyboard_navigation']:
            self.root.bind('<Key>', self._handle_keypress)
            self.root.focus_set()  # Hauptfenster fokussieren für Keyboard-Events
    
    def _display_current_box(self):
        """Zeigt das aktuelle Kästchen an"""
        if not self.problematic_boxes or self.current_index >= len(self.problematic_boxes):
            messagebox.showinfo("Fertig", "Alle Kästchen bearbeitet!")
            self.root.quit()
            return
        
        self.current_box_data = self.problematic_boxes[self.current_index]
        
        # Lade und zeige Bild
        self._load_and_display_image()
        
        # Update Labels
        self.file_label.config(text=self.current_box_data['image_filename'])
        
        box_type = "Z-Kästchen" if self.current_box_data['type'] == 'z' else "D-Kästchen"
        self.type_label.config(text=f"{box_type} #{self.current_box_data['box_index']}")
        
        self.prediction_label.config(text=str(self.current_box_data['predicted_digit']))
        
        confidence_text = f"{self.current_box_data['confidence']:.3f} ({self.current_box_data['confidence']*100:.1f}%)"
        self.confidence_label.config(text=confidence_text)
        
        # Update Progress
        progress_text = f"Kästchen {self.current_index + 1} von {len(self.problematic_boxes)} ({((self.current_index + 1)/len(self.problematic_boxes)*100):.1f}%)"
        self.progress_label.config(text=progress_text)
        
        status_text = f"Korrekturen: {self.corrections_made} | Training-Queue: {len(self.training_queue)}"
        self.status_label.config(text=status_text)
        
        # Update Training Button
        if hasattr(self, 'training_button'):
            training_images_count = len(list(self.training_images_path.glob("*.png")))
            if training_images_count > 0:
                self.training_button.config(state='normal', text=f"🧠 Training starten ({training_images_count} Daten)")
            else:
                self.training_button.config(state='disabled', text="🧠 Training starten")
        
        # Reset Input Display
        self.input_display.config(text="[Warten auf Eingabe...]", foreground="red")
    
    def _load_and_display_image(self):
        """Lädt und zeigt das Kästchen-Bild"""
        image_filename = self.current_box_data['image_filename']
        
        # Bestimme Pfad basierend auf Typ
        if self.current_box_data['type'] == 'z':
            image_path = self.kaestchen_z_path / image_filename
        else:
            image_path = self.kaestchen_d_path / image_filename
        
        if not image_path.exists():
            self.image_label.config(text=f"Bild nicht gefunden:\n{image_filename}")
            logger.warning(f"Bild nicht gefunden: {image_path}")
            return
        
        try:
            # Lade Bild mit OpenCV
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Skaliere Bild
            display_size = self.module_config['gui_settings']['image_display_size']
            zoom_factor = self.module_config['gui_settings']['image_zoom_factor']
            
            new_height = int(display_size * zoom_factor)
            new_width = int(img_rgb.shape[1] * (new_height / img_rgb.shape[0]))
            
            img_resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Konvertiere für Tkinter
            img_pil = Image.fromarray(img_resized)
            self.current_image = ImageTk.PhotoImage(img_pil)
            
            self.image_label.config(image=self.current_image, text="")
            
        except Exception as e:
            self.image_label.config(text=f"Fehler beim Laden:\n{e}")
            logger.error(f"Fehler beim Laden des Bildes {image_path}: {e}")
    
    def _handle_keypress(self, event):
        """Behandelt Keyboard-Navigation"""
        key = event.char.lower()
        
        # Nur verarbeiten wenn gültiger Key
        valid_digits = self.module_config['correction_settings']['valid_digits']
        empty_symbol = self.module_config['correction_settings']['empty_field_symbol'].lower()
        back_key = self.module_config['correction_settings']['back_key'].lower()
        
        if key in [d for d in valid_digits]:  # Ziffern 0-9
            self._apply_correction_from_key(key)
        elif key == empty_symbol:  # X für leeres Feld
            self._apply_correction_from_key(self.module_config['correction_settings']['empty_field_symbol'])
        elif key == back_key:  # B für zurück
            self._previous_box()
        elif event.keysym == 'Escape':  # ESC für beenden
            self._on_closing()
        
        return "break"  # Verhindert weitere Event-Behandlung
    
    def _apply_correction_from_key(self, correction: str):
        """Wendet Korrektur direkt von Tastendruck an"""
        # Zeige Eingabe kurz an
        display_text = "LEERES FELD" if correction == self.module_config['correction_settings']['empty_field_symbol'] else correction
        self.input_display.config(text=f"✓ {display_text}", foreground="green")
        
        # Aktualisiere GUI
        self.root.update()
        
        # Statistiken sammeln
        self._collect_correction_statistics(correction)
        
        # Speichere Korrektur
        self._save_correction(correction)
        
        # Zur Training-Queue hinzufügen
        self._add_to_training_queue(correction)
        
        # Statistiken
        self.corrections_made += 1
        
        # Auto-Advance mit längerem Delay für bessere UX
        if self.module_config['gui_settings']['auto_advance_after_keypress']:
            self.root.after(800, self._next_box)  # 800ms Pause für bessere Sichtbarkeit
        
        logger.info(f"Korrektur von Tastatur: {self.current_box_data['image_filename']} → {correction}")
    
    def _collect_correction_statistics(self, correction: str):
        """Sammelt Statistiken für Analyse"""
        if not self.module_config['statistics']['enable_correction_statistics']:
            return
        
        original_prediction = str(self.current_box_data['predicted_digit'])
        original_confidence = self.current_box_data['confidence']
        
        # Kategorisiere Korrektur
        is_correct_prediction = (original_prediction == correction)
        is_empty_field = (correction == self.module_config['correction_settings']['empty_field_symbol'])
        
        if is_correct_prediction:
            self.correction_statistics['false_positives'] += 1  # War richtig, aber als problematisch markiert
        else:
            self.correction_statistics['true_positives'] += 1   # War wirklich falsch
        
        if is_empty_field:
            self.correction_statistics['empty_fields_found'] += 1
        
        # Confidence-Analyse
        self.correction_statistics['confidence_analysis'].append({
            'original_confidence': original_confidence,
            'was_correct': is_correct_prediction,
            'was_empty_field': is_empty_field
        })
        
        # Detaillierte Info
        correction_detail = {
            'basename': self.current_box_data['basename'],
            'type': self.current_box_data['type'],
            'box_index': self.current_box_data['box_index'],
            'original_prediction': original_prediction,
            'original_confidence': original_confidence,
            'corrected_to': correction,
            'was_correct_prediction': is_correct_prediction,
            'was_empty_field': is_empty_field,
            'timestamp': datetime.now().isoformat()
        }
        
        self.correction_statistics['correction_details'].append(correction_detail)
        self.correction_statistics['total_corrections'] += 1
    
    def _save_correction(self, correction: str):
        """Speichert Korrektur in Pipeline-Ergebnisse"""
        basename = self.current_box_data['basename']
        box_type = self.current_box_data['type']
        box_index = self.current_box_data['box_index']
        
        # Lade bestehende Pipeline-Ergebnisse
        result_file = self.pipeline_results_path / f"{basename}_kaestchen.json"
        
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        else:
            logger.warning(f"Pipeline-Ergebnis nicht gefunden: {result_file}")
            return
        
        # Update Korrektur
        if box_type == 'z' and 'z_kaestchen' in result_data:
            for box in result_data['z_kaestchen']:
                if box.get('box_index') == box_index:
                    box['predicted_digit'] = correction
                    box['confidence'] = 1.0  # Manuelle Korrektur = 100% sicher
                    box['status'] = 'korrigiert'
                    box['correction_timestamp'] = datetime.now().isoformat()
                    break
        elif box_type == 'd' and 'd_kaestchen' in result_data:
            for box in result_data['d_kaestchen']:
                if box.get('box_index') == box_index:
                    box['predicted_digit'] = correction
                    box['confidence'] = 1.0
                    box['status'] = 'korrigiert'
                    box['correction_timestamp'] = datetime.now().isoformat()
                    break
        
        # Speichere zurück
        if self.module_config['correction_settings']['backup_corrections']:
            # Fix: Backup korrekt erstellen (ohne .json im suffix)
            backup_name = result_file.stem + self.module_config['correction_settings']['backup_suffix'] + ".json"
            backup_file = result_file.parent / backup_name
            shutil.copy2(result_file, backup_file)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    def _add_to_training_queue(self, correction: str):
        """Fügt Korrektur zur Training-Queue hinzu UND speichert sofort"""
        if not self.module_config['training_settings']['enable_training_collection']:
            return
        
        training_entry = {
            'basename': self.current_box_data['basename'],
            'type': self.current_box_data['type'],
            'box_index': self.current_box_data['box_index'],
            'image_filename': self.current_box_data['image_filename'],
            'original_prediction': self.current_box_data['predicted_digit'],
            'original_confidence': self.current_box_data['confidence'],
            'corrected_label': correction,
            'correction_timestamp': datetime.now().isoformat()
        }
        
        self.training_queue.append(training_entry)
        
        # SOFORT Training-Daten speichern (nicht warten bis Training-Button)
        self._save_single_training_data(training_entry)
        
        logger.info(f"Training-Daten gespeichert: {training_entry['image_filename']} → {correction}")
    
    def _save_single_training_data(self, training_entry: Dict):
        """Speichert einzelne Training-Daten sofort"""
        try:
            # 1. Kopiere Bild ins Training-Verzeichnis
            source_path = (self.kaestchen_z_path if training_entry['type'] == 'z' else self.kaestchen_d_path) / training_entry['image_filename']
            dest_path = self.training_images_path / training_entry['image_filename']
            
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                logger.debug(f"Bild kopiert: {source_path} → {dest_path}")
            else:
                logger.warning(f"Quell-Bild nicht gefunden: {source_path}")
                return
            
            # 2. Erstelle JSON für detaillierte Info
            json_data = {
                'image_filename': training_entry['image_filename'],
                'original_prediction': training_entry['original_prediction'],
                'original_confidence': training_entry['original_confidence'],
                'corrected_label': training_entry['corrected_label'],
                'correction_timestamp': training_entry['correction_timestamp'],
                'basename': training_entry['basename'],
                'box_type': training_entry['type'],
                'box_index': training_entry['box_index']
            }
            
            json_file = self.training_json_path / f"{Path(training_entry['image_filename']).stem}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # 3. Erweitere Label-CSV sofort
            label_entry = f"{training_entry['image_filename']},{training_entry['corrected_label']}\n"
            
            if training_entry['type'] == 'z':
                z_csv = self.training_labels_path / "z_kaestchen_labels.csv"
                with open(z_csv, 'a', encoding='utf-8') as f:
                    f.write(label_entry)
            else:
                d_csv = self.training_labels_path / "d_kaestchen_labels.csv"
                with open(d_csv, 'a', encoding='utf-8') as f:
                    f.write(label_entry)
            
            logger.debug(f"Training-Daten komplett gespeichert für: {training_entry['image_filename']}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Training-Daten: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_statistics(self):
        """Zeigt Korrektur-Statistiken"""
        if self.correction_statistics['total_corrections'] == 0:
            messagebox.showinfo("Statistiken", "Noch keine Korrekturen vorgenommen.")
            return
        
        stats = self.correction_statistics
        
        # Berechne Prozentuale Werte
        total = stats['total_corrections']
        false_positive_rate = (stats['false_positives'] / total * 100) if total > 0 else 0
        true_positive_rate = (stats['true_positives'] / total * 100) if total > 0 else 0
        empty_field_rate = (stats['empty_fields_found'] / total * 100) if total > 0 else 0
        
        # Confidence-Analyse
        correct_confidences = [item['original_confidence'] for item in stats['confidence_analysis'] if item['was_correct']]
        incorrect_confidences = [item['original_confidence'] for item in stats['confidence_analysis'] if not item['was_correct']]
        
        avg_conf_correct = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
        avg_conf_incorrect = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
        
        # Statistik-Text
        stats_text = f"""KORREKTUR-STATISTIKEN
{"="*50}

Gesamte Korrekturen: {total}

KATEGORIE-ANALYSE:
• Falsche Alarme: {stats['false_positives']} ({false_positive_rate:.1f}%)
  → OCR war richtig, aber als problematisch markiert
  
• Echte Fehler: {stats['true_positives']} ({true_positive_rate:.1f}%)
  → OCR war wirklich falsch
  
• Leere Felder: {stats['empty_fields_found']} ({empty_field_rate:.1f}%)
  → Durchgestrichene/übermalte Felder

CONFIDENCE-ANALYSE:
• Durchschnittliche Confidence bei RICHTIGEN Vorhersagen: {avg_conf_correct:.3f}
• Durchschnittliche Confidence bei FALSCHEN Vorhersagen: {avg_conf_incorrect:.3f}

EMPFEHLUNG FÜR SCHWELLWERT-TUNING:
→ Confidence-Schwelle könnte auf {avg_conf_correct:.2f} gesenkt werden
→ Das würde {stats['false_positives']} richtige Vorhersagen aus der Korrektur nehmen
"""
        
        # Zeige in Popup
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Korrektur-Statistiken")
        stats_window.geometry("600x500")
        
        text_widget = tk.Text(stats_window, wrap=tk.WORD, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state=tk.DISABLED)
        
        # Export Button
        export_button = ttk.Button(stats_window, text="📊 Statistiken exportieren", 
                                 command=self._export_statistics)
        export_button.pack(pady=10)
    
    def _export_statistics(self):
        """Exportiert Statistiken als JSON"""
        if not self.module_config['statistics']['export_statistics']:
            return
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'module_version': self.module_config['module_version'],
            'total_problematic_boxes': len(self.problematic_boxes),
            'corrections_completed': self.corrections_made,
            'statistics': self.correction_statistics,
            'recommendations': {
                'suggested_confidence_threshold': None,
                'false_positive_reduction': None
            }
        }
        
        # Berechne Empfehlungen
        if self.correction_statistics['confidence_analysis']:
            correct_confidences = [item['original_confidence'] for item in self.correction_statistics['confidence_analysis'] if item['was_correct']]
            if correct_confidences:
                suggested_threshold = sum(correct_confidences) / len(correct_confidences)
                export_data['recommendations']['suggested_confidence_threshold'] = suggested_threshold
                export_data['recommendations']['false_positive_reduction'] = self.correction_statistics['false_positives']
        
        # Speichere Export
        export_file = self.problem_data_path / self.module_config['statistics']['statistics_file']
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        messagebox.showinfo("Export", f"Statistiken exportiert nach:\n{export_file}")
        logger.info(f"Statistiken exportiert: {export_file}")
    
    def _start_keras_training(self):
        """Startet inkrementelles Keras-Training"""
        if not self.training_queue:
            messagebox.showwarning("Kein Training", "Keine Training-Daten gesammelt.")
            return
        
        # Bestätige Training
        response = messagebox.askyesno(
            "Training bestätigen",
            f"Training mit {len(self.training_queue)} neuen Beispielen starten?\n\n"
            f"Dies kann einige Minuten dauern und das Keras-Modell verbessern.",
            icon='question'
        )
        
        if not response:
            return
        
        try:
            # Prüfe ob Training-Daten vorhanden sind
            if not self._prepare_training_data():
                messagebox.showwarning("Keine Training-Daten", "Keine Training-Daten zum Trainieren vorhanden.")
                return
            
            # Führe Training aus
            messagebox.showinfo("Training", "Training wird gestartet...\nBitte warten Sie.")
            
            if self.module_config['training_settings']['show_training_progress']:
                # TODO: Progress-Dialog für Training
                pass
            
            result = subprocess.run([sys.executable, str(self.keras_trainer_path)], 
                                  capture_output=True, text=True, timeout=600)  # 10 Min Timeout
            
            if result.returncode == 0:
                training_count = len(list(self.training_images_path.glob("*.png")))
                messagebox.showinfo("Training erfolgreich", 
                                  f"🎯 Keras-Modell wurde mit {training_count} neuen Beispielen trainiert!\n\n"
                                  f"✅ Das verbesserte Modell wird bei der nächsten Pipeline-Ausführung verwendet.\n\n"
                                  f"🔄 Empfehlung: Pipeline komplett neu laufen lassen um die Verbesserung zu testen.\n\n"
                                  f"📊 Training-Daten bleiben in C:\\ZaehlerkartenV2\\KI\\Piplinetraining erhalten.")
                
                if self.module_config['training_settings']['auto_clear_queue_after_training']:
                    self.training_queue.clear()
                    if hasattr(self, 'training_button'):
                        self.training_button.config(state='normal', text=f"🧠 Training starten ({len(list(self.training_images_path.glob('*.png')))} Daten)")
                
                logger.info(f"Training erfolgreich abgeschlossen mit {training_count} Beispielen")
            else:
                messagebox.showerror("Training fehlgeschlagen", 
                                   f"❌ Fehler beim Training:\n\n{result.stderr}")
                logger.error(f"Training fehlgeschlagen: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            messagebox.showerror("Training Timeout", "Training dauerte zu lange und wurde abgebrochen.")
        except Exception as e:
            messagebox.showerror("Training Fehler", f"Unerwarteter Fehler:\n{e}")
            logger.error(f"Training Fehler: {e}")
    
    def _prepare_training_data(self):
        """Bereitet Training-Daten vor (wird nur noch für CSV-Vervollständigung genutzt)"""
        # Training-Daten werden jetzt sofort in _save_single_training_data() gespeichert
        # Diese Funktion prüft nur noch ob alle Daten korrekt sind
        
        z_count = len(list(self.training_images_path.glob("*_z_*.png")))
        d_count = len(list(self.training_images_path.glob("*_d_*.png")))
        total_images = z_count + d_count
        
        logger.info(f"Training-Daten Übersicht: {total_images} Bilder ({z_count} Z-Kästchen, {d_count} D-Kästchen)")
        
        if total_images != len(self.training_queue):
            logger.warning(f"Mismatch: {len(self.training_queue)} Queue-Einträge vs {total_images} gespeicherte Bilder")
        
        return total_images > 0
    
    def _skip_current(self):
        """Überspringt aktuelles Kästchen (entfernt da nicht mehr nötig)"""
        self._next_box()
    
    def _next_box(self):
        """Geht zum nächsten Kästchen"""
        if self.current_index < len(self.problematic_boxes) - 1:
            self.current_index += 1
            self._display_current_box()
        else:
            # Alle Kästchen bearbeitet - zeige Abschluss-Screen
            self._show_completion_screen()
    
    def _show_completion_screen(self):
        """Zeigt Abschluss-Screen nach Bearbeitung aller Kästchen"""
        # Verstecke Hauptfenster-Inhalte
        for widget in self.root.winfo_children():
            widget.pack_forget()
            widget.grid_forget()
        
        # Erstelle Abschluss-Screen
        completion_frame = ttk.Frame(self.root, padding="20")
        completion_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titel
        title_label = ttk.Label(completion_frame, text="🎉 KORREKTUR ABGESCHLOSSEN!", 
                               font=("Arial", 18, "bold"), foreground="green")
        title_label.pack(pady=20)
        
        # Zusammenfassung
        summary_frame = ttk.LabelFrame(completion_frame, text="Zusammenfassung", padding="15")
        summary_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(summary_frame, text=f"Bearbeitete Kästchen: {len(self.problematic_boxes)}", 
                 font=("Arial", 12)).pack(anchor=tk.W, pady=2)
        ttk.Label(summary_frame, text=f"Korrekturen vorgenommen: {self.corrections_made}", 
                 font=("Arial", 12)).pack(anchor=tk.W, pady=2)
        
        # Training-Daten Info
        training_images_count = len(list(self.training_images_path.glob("*.png")))
        ttk.Label(summary_frame, text=f"Training-Daten gesammelt: {training_images_count} Bilder", 
                 font=("Arial", 12)).pack(anchor=tk.W, pady=2)
        
        # Statistiken Übersicht
        if self.correction_statistics['total_corrections'] > 0:
            stats_frame = ttk.LabelFrame(completion_frame, text="Schnell-Statistik", padding="10")
            stats_frame.pack(fill=tk.X, pady=10)
            
            false_positive_rate = (self.correction_statistics['false_positives'] / 
                                 self.correction_statistics['total_corrections'] * 100)
            
            ttk.Label(stats_frame, text=f"• Falsche Alarme: {self.correction_statistics['false_positives']} ({false_positive_rate:.1f}%)", 
                     font=("Arial", 10)).pack(anchor=tk.W)
            ttk.Label(stats_frame, text=f"• Echte Fehler: {self.correction_statistics['true_positives']}", 
                     font=("Arial", 10)).pack(anchor=tk.W)
            ttk.Label(stats_frame, text=f"• Leere Felder: {self.correction_statistics['empty_fields_found']}", 
                     font=("Arial", 10)).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(completion_frame)
        button_frame.pack(pady=30)
        
        # Training Button (falls Daten vorhanden)
        if training_images_count > 0:
            ttk.Button(button_frame, text=f"🧠 Training starten ({training_images_count} Daten)", 
                      command=self._start_keras_training, style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        
        # Statistiken Button
        ttk.Button(button_frame, text="📊 Detaillierte Statistiken", 
                  command=self._show_statistics).pack(side=tk.LEFT, padx=10)
        
        # Export Button
        ttk.Button(button_frame, text="💾 Statistiken exportieren", 
                  command=self._export_statistics).pack(side=tk.LEFT, padx=10)
        
        # Beenden Button
        ttk.Button(button_frame, text="✅ Fertig & Beenden", 
                  command=self._final_exit, style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        
        # Empfehlungen
        recommendations_frame = ttk.LabelFrame(completion_frame, text="🎯 Empfehlungen", padding="10")
        recommendations_frame.pack(fill=tk.X, pady=10)
        
        if self.correction_statistics['false_positives'] > 0:
            correct_confidences = [item['original_confidence'] for item in self.correction_statistics['confidence_analysis'] if item['was_correct']]
            if correct_confidences:
                avg_correct_conf = sum(correct_confidences) / len(correct_confidences)
                ttk.Label(recommendations_frame, 
                         text=f"💡 Tipp: Confidence-Schwelle von 0.8 auf {avg_correct_conf:.2f} senken", 
                         font=("Arial", 11, "bold"), foreground="blue").pack(anchor=tk.W)
                ttk.Label(recommendations_frame, 
                         text=f"   Das würde {self.correction_statistics['false_positives']} falsche Alarme vermeiden!", 
                         font=("Arial", 10)).pack(anchor=tk.W)
        
        if training_images_count > 0:
            ttk.Label(recommendations_frame, 
                     text="🔄 Nach dem Training: Pipeline komplett neu laufen lassen!", 
                     font=("Arial", 11, "bold"), foreground="green").pack(anchor=tk.W, pady=(5, 0))
        
        # Speichere Fortschritt automatisch
        self._save_progress()
    
    def _final_exit(self):
        """Finaler Exit mit Bestätigung"""
        training_images_count = len(list(self.training_images_path.glob("*.png")))
        
        if training_images_count > 0:
            response = messagebox.askyesno(
                "Training noch nicht gestartet",
                f"Sie haben {training_images_count} Training-Daten gesammelt, aber noch nicht trainiert.\n\n"
                f"Möchten Sie das Training jetzt starten?\n\n"
                f"[Ja] = Training starten\n"
                f"[Nein] = Ohne Training beenden",
                icon='question'
            )
            
            if response:
                self._start_keras_training()
                return  # Nicht beenden, User kann nach Training selbst beenden
        
        # Endgültig beenden
        messagebox.showinfo("Auf Wiedersehen", 
                          f"✅ Korrektur-Session beendet!\n\n"
                          f"📊 {self.corrections_made} Korrekturen gespeichert\n"
                          f"🧠 {training_images_count} Training-Daten bereit\n\n"
                          f"Die Daten stehen für die nächste Pipeline-Ausführung bereit.")
        
        self.root.destroy()
    
    def _previous_box(self):
        """Geht zum vorherigen Kästchen"""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current_box()
    
    def _jump_to_index(self):
        """Springt zu spezifischem Index"""
        try:
            target_index = int(self.jump_var.get()) - 1  # 1-basiert zu 0-basiert
            if 0 <= target_index < len(self.problematic_boxes):
                self.current_index = target_index
                self._display_current_box()
                self.jump_var.set("")
            else:
                messagebox.showwarning("Ungültiger Index", f"Index muss zwischen 1 und {len(self.problematic_boxes)} liegen.")
        except ValueError:
            messagebox.showwarning("Ungültige Eingabe", "Bitte geben Sie eine gültige Zahl ein.")
    
    def _save_progress(self):
        """Speichert Fortschritt (ohne automatischen Export)"""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'total_boxes': len(self.problematic_boxes),
            'current_index': self.current_index,
            'corrections_made': self.corrections_made,
            'training_queue_size': len(self.training_queue),
            'module_version': self.module_config['module_version'],
            'correction_statistics': self.correction_statistics,
            'completed': self.current_index >= len(self.problematic_boxes) - 1
        }
        
        progress_file = self.problem_data_path / "korrektur_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fortschritt gespeichert: {progress_file}")
        
        # Statistiken nur exportieren wenn explizit gewünscht (nicht automatisch)
        # Das passiert jetzt nur noch über Button oder _final_exit
    
    def _on_closing(self):
        """Behandelt Fenster-Schließen"""
        if self.current_index >= len(self.problematic_boxes) - 1:
            # Alle Kästchen bearbeitet - zeige Abschluss-Screen statt direkt zu beenden
            self._show_completion_screen()
            return
        
        if self.corrections_made > 0:
            response = messagebox.askyesnocancel(
                "Beenden bestätigen",
                f"Sie haben {self.corrections_made} Korrekturen vorgenommen.\n\n"
                f"Fortschritt: {self.current_index + 1}/{len(self.problematic_boxes)} Kästchen bearbeitet\n\n"
                f"Möchten Sie vor dem Beenden speichern?",
                icon='question'
            )
            
            if response is True:  # Ja
                self._save_progress()
                self.root.destroy()
            elif response is False:  # Nein
                self.root.destroy()
            # Bei Cancel passiert nichts
        else:
            self.root.destroy()
    
    def run(self):
        """Startet die GUI"""
        self.root.mainloop()


def main():
    """Hauptfunktion"""
    try:
        app = KaestchenKorrekturGUI()
        app.run()
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Kritischer Fehler", f"Die Anwendung konnte nicht gestartet werden:\n\n{e}")


if __name__ == "__main__":
    main()