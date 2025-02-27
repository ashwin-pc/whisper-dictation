#!/usr/bin/env python3
import os
import time
import tempfile
import threading
import pyaudio
import wave
import numpy as np
import subprocess
import rumps
from pynput import keyboard
from pynput.keyboard import Key, Controller
import faster_whisper

class WhisperDictationApp(rumps.App):
    def __init__(self):
        super(WhisperDictationApp, self).__init__("🎙️", quit_button=rumps.MenuItem("Quit"))
        
        # Add menu items
        self.menu = ["Start/Stop Listening", None, "Settings"]
        
        # Register signal handlers for graceful shutdown
        import signal
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        # Recording state
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.keyboard_controller = Controller()
        
        # Status item
        self.status_item = rumps.MenuItem("Status: Ready")
        self.menu.insert_before("Settings", self.status_item)
        
        # Initialize Whisper model
        self.model = None
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()
        
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        
        # Hotkey configuration - we'll listen for globe/fn key (vk=63)
        self.trigger_key = 63  # Key code for globe/fn key
        self.setup_global_monitor()
        
        # Show initial message
        print("Started WhisperDictation app. Look for 🎙️ in your menu bar.")
        print("Press and hold the Globe/Fn key (vk=63) to record. Release to transcribe.")
        print("You may need to grant this app accessibility permissions in System Preferences.")
        print("Go to System Preferences → Security & Privacy → Privacy → Accessibility")
        print("and add your terminal or the built app to the list.")
    
    def load_model(self):
        self.title = "🎙️ (Loading...)"
        self.status_item.title = "Status: Loading Whisper model..."
        try:
            self.model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
            self.title = "🎙️"
            self.status_item.title = "Status: Ready"
            print("Whisper model loaded successfully!")
        except Exception as e:
            self.title = "🎙️ (Error)"
            self.status_item.title = "Status: Error loading model"
            print(f"Error loading model: {e}")
    
    def setup_global_monitor(self):
        # Create a separate thread to monitor for global key events
        self.key_monitor_thread = threading.Thread(target=self.monitor_keys)
        self.key_monitor_thread.daemon = True
        self.key_monitor_thread.start()
    
    def monitor_keys(self):
        # Track state of key 63 (Globe/Fn key)
        self.is_recording_with_key63 = False
        
        def on_press(key):
            # Print all key presses for debugging
            print(f"DEBUG: Key pressed: {key} (type: {type(key)})")
            
            # Check if it has a virtual key code
            if hasattr(key, 'vk'):
                print(f"DEBUG: Key with vk={key.vk} pressed")
                
                # Skip key 63 here since we'll handle it in on_release
                if key.vk != self.trigger_key and not self.recording:
                    # Handle other keys with vk codes
                    # Common virtual key codes: Space=49, Tab=48, Command=55, Option=58, Control=59
                    if key.vk == 49 or key.vk == 55:  # Space or Command
                        print(f"Alternative key (vk={key.vk}) pressed - Starting recording")
                        self.start_recording()
            
            # Also check for special keys like F-keys
            try:
                if key in [Key.f12, Key.f13, Key.f17, Key.f18, Key.cmd, Key.space] and not self.recording:
                    print(f"Special key {key} pressed - Starting recording")
                    self.start_recording()
            except:
                pass
        
        def on_release(key):
            # Print key releases for debugging
            if hasattr(key, 'vk'):
                print(f"DEBUG: Key with vk={key.vk} released")
                
                # Special handling for key 63 (Globe/Fn key)
                if key.vk == self.trigger_key:
                    if not self.recording and not self.is_recording_with_key63:
                        # Start recording on first release
                        print(f"TARGET KEY RELEASED! Globe/Fn key (vk={key.vk}) released - STARTING recording")
                        self.is_recording_with_key63 = True
                        self.start_recording()
                    elif self.recording and self.is_recording_with_key63:
                        # Stop recording on second release
                        print(f"TARGET KEY RELEASED AGAIN! Globe/Fn key (vk={key.vk}) released - STOPPING recording")
                        self.is_recording_with_key63 = False
                        self.stop_recording()
                
                # Handle other keys normally
                elif self.recording:
                    if key.vk == 49 or key.vk == 55:  # Space or Command
                        print(f"Alternative key (vk={key.vk}) released - Stopping recording")
                        self.stop_recording()
            
            # Also check for special keys
            try:
                if key in [Key.f12, Key.f13, Key.f17, Key.f18, Key.cmd, Key.space] and self.recording:
                    print(f"Special key {key} released - Stopping recording")
                    self.stop_recording()
            except:
                pass
        
        # Start the listener with full debugging
        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                print(f"Keyboard listener started - listening for key events")
                print(f"Target key is Globe/Fn key (vk={self.trigger_key})")
                print(f"Alternative keys also enabled: Space (vk=49), Command (vk=55), F12, F13")
                print(f"Press any of these keys to start recording, release to stop")
                print(f"DEBUG MODE: All key presses will be logged for debugging purposes")
                listener.join()
        except Exception as e:
            print(f"Error with keyboard listener: {e}")
            print("You may need to grant accessibility permissions in System Preferences")
            print("Go to System Preferences → Privacy & Security → Privacy → Accessibility")
            print("and add Terminal or iTerm to the list of apps that can control your computer")
    
    @rumps.clicked("Start Recording")
    def start_recording_menu(self, _):
        if not self.recording:
            self.start_recording()
    
    @rumps.clicked("Stop Recording")
    def stop_recording_menu(self, _):
        if self.recording:
            self.stop_recording()
    
    def start_recording(self):
        if not hasattr(self, 'model') or self.model is None:
            print("Model not loaded. Please wait for the model to finish loading.")
            self.status_item.title = "Status: Waiting for model to load"
            return
            
        self.frames = []
        self.recording = True
        
        # Update UI
        self.title = "🎙️ (Recording)"
        self.status_item.title = "Status: Recording..."
        print("Recording started. Speak now...")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
    
    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
        # Update UI
        self.title = "🎙️ (Transcribing)"
        self.status_item.title = "Status: Transcribing..."
        print("Recording stopped. Transcribing...")
        
        # Process in background
        transcribe_thread = threading.Thread(target=self.process_recording)
        transcribe_thread.start()
    
    def process_recording(self):
        # Transcribe and insert text
        try:
            self.transcribe_audio()
        except Exception as e:
            print(f"Error during transcription: {e}")
            self.status_item.title = "Status: Error during transcription"
        finally:
            self.title = "🎙️"  # Reset title
    
    def record_audio(self):
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        while self.recording:
            data = stream.read(self.chunk)
            self.frames.append(data)
            
        stream.stop_stream()
        stream.close()
    
    def transcribe_audio(self):
        if not self.frames:
            self.title = "🎙️"
            self.status_item.title = "Status: No audio recorded"
            print("No audio recorded")
            return
            
        # Save the recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        
        print(f"Audio saved to temporary file. Transcribing...")
        
        # Transcribe with Whisper
        try:
            segments, info = self.model.transcribe(temp_filename, beam_size=5)
            
            text = ""
            for segment in segments:
                text += segment.text
            
            if text:
                # Insert text at cursor position
                self.insert_text(text)
                print(f"Transcription: {text}")
                self.status_item.title = f"Status: Transcribed: {text[:30]}..."
            else:
                print("No speech detected")
                self.status_item.title = "Status: No speech detected"
        except Exception as e:
            print(f"Transcription error: {e}")
            self.status_item.title = "Status: Transcription error"
            raise
        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)
    
    def insert_text(self, text):
        # Use pbpaste/pbcopy to insert text at cursor position
        process = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
        
        # Simulate Command+V to paste
        print("Pasting text at cursor position...")
        self.keyboard_controller.press(Key.cmd)
        self.keyboard_controller.press('v')
        self.keyboard_controller.release('v')
        self.keyboard_controller.release(Key.cmd)
        print("Text pasted successfully")
    
    @rumps.clicked("Settings")
    def settings(self, _):
        response = rumps.Window(
            message="Whisper Dictation Settings",
            title="Settings",
            default_text="No settings available yet",
            ok="OK",
            cancel=None
        ).run()
    
    def handle_shutdown(self, signal, frame):
        """Handle graceful shutdown when Ctrl+C is pressed"""
        print("\nShutting down Whisper Dictation...")
        
        # Stop recording if in progress
        if self.recording:
            self.stop_recording()
        
        # Close PyAudio
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        print("Goodbye!")
        rumps.quit_application()

if __name__ == "__main__":
    WhisperDictationApp().run()