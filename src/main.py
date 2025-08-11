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
import signal
import queue

# Set up a global flag for handling SIGINT
exit_flag = False

def signal_handler(sig, frame):
    """Global signal handler for graceful shutdown"""
    global exit_flag
    print("\nShutdown signal received, exiting gracefully...")
    exit_flag = True
    # Try to force exit if the app doesn't respond quickly
    import threading
    threading.Timer(2.0, lambda: os._exit(0)).start()

# Register the global signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class WhisperDictationApp(rumps.App):
    def __init__(self):
        super(WhisperDictationApp, self).__init__("🎙️", quit_button=rumps.MenuItem("Quit"))
        
        # Status item
        self.status_item = rumps.MenuItem("Status: Ready")
        
        # Add menu items - use a single menu item for toggling recording
        self.recording_menu_item = rumps.MenuItem("Start Recording")
        self.menu = [self.recording_menu_item, None, self.status_item]
        
        # Recording state
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.keyboard_controller = Controller()
        # Toggle for streaming transcription
        self.enable_streaming = False
        self.audio_queue = None
        
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
        print("Press Ctrl+C to quit the application.")
        print("You may need to grant this app accessibility permissions in System Preferences.")
        print("Go to System Preferences → Security & Privacy → Privacy → Accessibility")
        print("and add your terminal or the built app to the list.")
        
        # Start a watchdog thread to check for exit flag
        self.watchdog = threading.Thread(target=self.check_exit_flag, daemon=True)
        self.watchdog.start()
    
    def check_exit_flag(self):
        """Monitor the exit flag and terminate the app when set"""
        while True:
            if exit_flag:
                print("Watchdog detected exit flag, shutting down...")
                self.cleanup()
                rumps.quit_application()
                os._exit(0)
                break
            time.sleep(0.5)
    
    def cleanup(self):
        """Clean up resources before exiting"""
        print("Cleaning up resources...")
        # Stop recording if in progress
        if self.recording:
            self.recording = False
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
        
        # Close PyAudio
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except:
                pass
    
    def load_model(self):
        self.title = "🎙️ (Loading...)"
        self.status_item.title = "Status: Loading Whisper model..."
        try:
            self.model = faster_whisper.WhisperModel("small.en")
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
            # Removed logging for every key press; log only when target key is pressed
            if hasattr(key, 'vk') and key.vk == self.trigger_key:
                print(f"DEBUG: Target key (vk={key.vk}) pressed")
        
        def on_release(key):
            if hasattr(key, 'vk'):
                print(f"DEBUG: Key with vk={key.vk} released")
                if key.vk == self.trigger_key:
                    if not self.recording and not self.is_recording_with_key63:
                        print(f"TARGET KEY RELEASED! Globe/Fn key (vk={key.vk}) released - STARTING recording")
                        self.is_recording_with_key63 = True
                        self.start_recording()
                    elif self.recording and self.is_recording_with_key63:
                        print(f"TARGET KEY RELEASED AGAIN! Globe/Fn key (vk={key.vk}) released - STOPPING recording")
                        self.is_recording_with_key63 = False
                        self.stop_recording()
        
        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                print(f"Keyboard listener started - listening for key events")
                print(f"Target key is Globe/Fn key (vk={self.trigger_key})")
                print(f"Press and release the target key to control recording")
                listener.join()
        except Exception as e:
            print(f"Error with keyboard listener: {e}")
            print("Please check accessibility permissions in System Preferences")
    
    @rumps.clicked("Start Recording")  # This will be matched by title
    def toggle_recording(self, sender):
        if not self.recording:
            self.start_recording()
            sender.title = "Stop Recording"
        else:
            self.stop_recording()
            sender.title = "Start Recording"
    
    def start_recording(self):
        if not hasattr(self, 'model') or self.model is None:
            print("Model not loaded. Please wait for the model to finish loading.")
            self.status_item.title = "Status: Waiting for model to load"
            return

        self.frames = []
        self.recording = True

        if self.enable_streaming:
            self.audio_queue = queue.Queue()
            self.transcribe_thread = threading.Thread(target=self.transcribe_stream)
            self.transcribe_thread.start()

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
        if self.enable_streaming:
            if self.audio_queue is not None:
                self.audio_queue.put(None)
            if hasattr(self, 'transcribe_thread'):
                self.transcribe_thread.join()
            self.audio_queue = None
            self.title = "🎙️"
            self.status_item.title = "Status: Ready"
            print("Recording stopped.")
        else:
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
            if self.enable_streaming and self.audio_queue is not None:
                self.audio_queue.put(data)
            else:
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
        # Type text at cursor position without altering the clipboard
        print("Typing text at cursor position...")
        self.keyboard_controller.type(text)
        print("Text typed successfully")

    def transcribe_stream(self):
        def generator():
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    break
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                yield samples

        try:
            for segment in self.model.transcribe(generator, beam_size=5):
                if segment.text:
                    self.insert_text(segment.text)
        except Exception as e:
            print(f"Transcription error: {e}")
            self.status_item.title = "Status: Transcription error"
        finally:
            self.title = "🎙️"
            self.status_item.title = "Status: Ready"
    
    def handle_shutdown(self, signal, frame):
        """This method is no longer used with the global handler approach"""
        pass

# Wrap the main execution in a try-except to ensure clean exit
if __name__ == "__main__":
    try:
        WhisperDictationApp().run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, exiting...")
        os._exit(0)