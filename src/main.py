#!/usr/bin/env python3
import os
import platform
import sys
import time
import threading
import pyaudio
import numpy as np
import rumps
from pynput import keyboard
from pynput.keyboard import Key, Controller
if platform.machine() != "arm64":
    print("Error: mlx-whisper requires Apple Silicon (arm64). This Mac appears to be Intel-based.")
    print("Please use the 'faster-whisper' backend instead.")
    sys.exit(1)
import mlx_whisper
import signal
from text_selection import TextSelection
from bedrock_client import BedrockClient
from recording_indicator import RecordingIndicator
from logger_config import setup_logging

logger = setup_logging()

# Set up a global flag for handling SIGINT
exit_flag = False

def signal_handler(sig, frame):
    """Global signal handler for graceful shutdown"""
    global exit_flag
    logger.info("Shutdown signal received, exiting gracefully...")
    exit_flag = True
    # Try to force exit if the app doesn't respond quickly
    threading.Timer(2.0, lambda: os._exit(0)).start()

# Set up graceful shutdown handling for interrupt and termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class WhisperDictationApp(rumps.App):
    def __init__(self):
        super(WhisperDictationApp, self).__init__("🎙️", quit_button=rumps.MenuItem("Quit"))

        # Status item
        self.status_item = rumps.MenuItem("Status: Ready")

        # Add menu items - use a single menu item for toggling recording
        self.recording_menu_item = rumps.MenuItem("Start Recording")

        # Recording state
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.frames_lock = threading.Lock()
        self.keyboard_controller = Controller()

        # Microphone selection (None = use default)
        self.selected_input_device = None

        # Create microphone selection submenu
        self.mic_menu = {}
        self.mic_menu_mapping = {}  # Maps menu title to device index
        self.setup_microphone_menu()

        self.menu = [self.recording_menu_item, self.mic_submenu, None, self.status_item]

        # Initialize text selection handler
        self.text_selector = TextSelection()

        # Initialize Bedrock client
        self.bedrock_client = BedrockClient()

        # Initialize recording indicator
        self.indicator = RecordingIndicator()
        self.indicator.set_app_reference(self)

        # Initialize Whisper model
        self.model_loaded = False
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024

        # Chunked transcription state
        self.partial_text = ""
        self.last_transcribed_frame_count = 0
        self.chunked_stop = threading.Event()
        self.CHUNK_INTERVAL_SECONDS = 5

        # Hotkey configuration - we'll listen for globe/fn key (vk=63)
        self.trigger_key = 63  # Key code for globe/fn key

        # Right Shift trigger state (optimistic recording with discard threshold)
        self.shift_press_time = None
        self.shift_held = False
        self.shift_threshold = 0.75  # Minimum hold time in seconds

        self.setup_global_monitor()

        # Show initial message
        logger.info("Started WhisperDictation app. Look for 🎙️ in your menu bar.")
        logger.info("Press and hold the Globe/Fn key (vk=63) to record. Release to transcribe.")
        logger.info("Alternatively, hold Right Shift to record (release after 0.75s to process, before to discard).")
        logger.info("Press Ctrl+C to quit the application.")
        logger.info("You may need to grant this app accessibility permissions in System Preferences.")
        logger.info("Go to System Preferences → Security & Privacy → Privacy → Accessibility")
        logger.info("and add your terminal or the built app to the list.")

        # Test Bedrock connection
        if self.bedrock_client.is_available():
            logger.info("✓ Bedrock client initialized successfully")
        else:
            logger.warning("⚠ Bedrock client not available - text enhancement features disabled")

        # Start a watchdog thread to check for exit flag
        self.watchdog = threading.Thread(target=self.check_exit_flag, daemon=True)
        self.watchdog.start()

    def check_exit_flag(self):
        """Monitor the exit flag and terminate the app when set"""
        while True:
            if exit_flag:
                logger.info("Watchdog detected exit flag, shutting down...")
                self.cleanup()
                rumps.quit_application()
                os._exit(0)
                break
            time.sleep(0.5)

    def cleanup(self):
        """Clean up resources before exiting"""
        logger.info("Cleaning up resources...")
        # Stop recording if in progress
        if self.recording:
            self.recording = False
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)

        # Stop chunked transcription thread if it is running
        if hasattr(self, 'chunked_stop'):
            self.chunked_stop.set()
        if hasattr(self, 'chunk_thread') and self.chunk_thread.is_alive():
            self.chunk_thread.join(timeout=1.0)

        # Close PyAudio
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except Exception:
                pass

    def load_model(self):
        self.title = "🎙️ (Loading...)"
        self.status_item.title = "Status: Loading Whisper model..."
        self.model_path = "mlx-community/distil-whisper-large-v3"
        self.initial_prompt = (
            "This is a voice dictation for software engineering work. "
            "Common terms include: AWS, Amazon Web Services, Bedrock, Lambda, S3, EC2, ECS, EKS, "
            "CloudFormation, CloudWatch, DynamoDB, SQS, SNS, IAM, VPC, API Gateway, "
            "Python, JavaScript, TypeScript, React, Node.js, Docker, Kubernetes, "
            "Git, GitHub, pull request, commit, merge, branch, CI/CD, "
            "JSON, YAML, REST, GraphQL, HTTP, HTTPS, SSL, TLS, OAuth, JWT, "
            "Redis, PostgreSQL, MongoDB, Kafka, Terraform, CDK, "
            "VS Code, IntelliJ, PyCharm, Xcode, macOS, Linux, Ubuntu, "
            "Claude, Anthropic, OpenAI, GPT, LLM, MLX, Whisper, "
            "oncall, ticket, deployment, rollback, pipeline, microservice, "
            "async, await, callback, promise, thread, mutex, semaphore, "
            "OTEL, OpenTelemetry, observability, tracing, metrics, spans."
        )
        try:
            # Warm up the model by running a short silent transcription
            load_start = time.time()
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            mlx_whisper.transcribe(dummy_audio, path_or_hf_repo=self.model_path)
            load_elapsed = time.time() - load_start
            self.model_loaded = True
            self.title = "🎙️"
            self.status_item.title = "Status: Ready"
            logger.info(f"[METRICS] Model load + warm-up: {load_elapsed*1000:.0f}ms")
            logger.info("Whisper model loaded successfully!")
        except Exception as e:
            self.model_loaded = False
            self.title = "🎙️ (Error)"
            self.status_item.title = "Status: Error loading model"
            logger.error(f"Error loading model: {e}")

    def setup_microphone_menu(self):
        """Setup the microphone selection submenu"""
        self.mic_submenu = rumps.MenuItem("Microphone")
        devices = self.get_input_devices()

        for device in devices:
            title = device['name']
            if device['is_default']:
                title += " (Default)"

            menu_item = rumps.MenuItem(title, callback=self.select_microphone)
            # Mark the default device as selected initially
            if device['is_default']:
                menu_item.state = True

            self.mic_menu[title] = menu_item
            self.mic_menu_mapping[title] = device['index']
            self.mic_submenu.add(menu_item)

    def setup_global_monitor(self):
        # Create a separate thread to monitor for global key events
        self.key_monitor_thread = threading.Thread(target=self.monitor_keys)
        self.key_monitor_thread.daemon = True
        self.key_monitor_thread.start()

    def get_input_devices(self):
        """Get list of available audio input devices"""
        devices = []
        default_index = self.audio.get_default_input_device_info()['index']
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': info['index'],
                    'name': info['name'],
                    'is_default': info['index'] == default_index
                })
        return devices

    def select_microphone(self, sender):
        """Callback when a microphone is selected from the menu"""
        # Uncheck all items in the microphone menu
        for item in self.mic_menu.values():
            item.state = False
        # Check the selected item
        sender.state = True
        # Store the device index (stored in the menu item's title parsing or we use a mapping)
        device_index = self.mic_menu_mapping.get(sender.title)
        self.selected_input_device = device_index
        device_name = sender.title.replace(" (Default)", "")
        logger.info(f"Microphone changed to: {device_name}")

    def discard_recording(self):
        """Discard current recording without processing (held too short)"""
        self.recording = False
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0)
        # Stop chunked transcription and wait for it to fully exit
        self.chunked_stop.set()
        if hasattr(self, 'chunk_thread') and self.chunk_thread.is_alive():
            self.chunk_thread.join(timeout=10.0)
        with self.frames_lock:
            self.frames = []
        self.indicator.stop()
        self.title = "🎙️"
        self.status_item.title = "Status: Recording discarded (too short)"
        logger.info("Recording discarded - held for less than threshold")

    def monitor_keys(self):
        # Track state of key 63 (Globe/Fn key)
        self.is_recording_with_key63 = False

        def on_press(key):
            # If Right Shift is held and another key is pressed, cancel recording (user is typing)
            if self.shift_held and key != Key.shift_r:
                logger.info("Other key pressed while Right Shift held - canceling recording")
                self.shift_held = False
                self.discard_recording()
                return

            # Removed logging for every key press; log only when target key is pressed
            if hasattr(key, 'vk') and key.vk == self.trigger_key:
                logger.debug(f"Target key (vk={key.vk}) pressed")

            # Right Shift handling - start recording immediately (optimistic)
            if key == Key.shift_r:
                if not self.recording:
                    logger.info("Right Shift pressed - starting recording immediately")
                    self.shift_press_time = time.time()
                    self.shift_held = True
                    self.start_recording()

        def on_release(key):
            if hasattr(key, 'vk'):
                logger.debug(f"Key with vk={key.vk} released")
                if key.vk == self.trigger_key:
                    if not self.recording and not self.is_recording_with_key63:
                        logger.debug(f"Globe/Fn key (vk={key.vk}) released - STARTING recording")
                        self.is_recording_with_key63 = True
                        self.start_recording()
                    elif self.recording and self.is_recording_with_key63:
                        logger.debug(f"Globe/Fn key (vk={key.vk}) released - STOPPING recording")
                        self.is_recording_with_key63 = False
                        self.stop_recording()

            # Right Shift handling - check duration and discard or process
            if key == Key.shift_r and self.shift_held:
                hold_duration = time.time() - self.shift_press_time
                self.shift_held = False

                if hold_duration < self.shift_threshold:
                    logger.info(f"Right Shift released after {hold_duration:.2f}s - discarding (< {self.shift_threshold}s)")
                    self.discard_recording()
                else:
                    logger.info(f"Right Shift released after {hold_duration:.2f}s - processing")
                    self.stop_recording()

        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release, suppress=False) as listener:
                logger.debug(f"Keyboard listener started - listening for key events")
                logger.debug(f"Target key is Globe/Fn key (vk={self.trigger_key})")
                logger.debug(f"Press and release the target key to control recording")
                listener.join()
        except Exception as e:
            logger.error(f"Error with keyboard listener: {e}")
            logger.error("Please check accessibility permissions in System Preferences")

    @rumps.clicked("Start Recording")  # This will be matched by title
    def toggle_recording(self, sender):
        if not self.recording:
            self.start_recording()
            sender.title = "Stop Recording"
        else:
            self.stop_recording()
            sender.title = "Start Recording"

    def start_recording(self):
        if not getattr(self, 'model_loaded', False):
            logger.warning("Model not loaded. Please wait for the model to finish loading.")
            self.status_item.title = "Status: Waiting for model to load"
            return

        with self.frames_lock:
            self.frames = []
        self.recording = True
        self.partial_text = ""
        self.last_transcribed_frame_count = 0
        self.chunked_stop.clear()

        # Update UI
        self.title = "🎙️ (Recording)"
        self.status_item.title = "Status: Recording..."
        logger.info("Recording started. Speak now...")

        # Show recording indicator
        self.indicator.start()

        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

        # Start chunked transcription thread
        self.chunk_thread = threading.Thread(target=self.chunked_transcribe_loop)
        self.chunk_thread.start()

    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()

        # Stop chunked transcription and wait for it to fully finish
        # before starting the final transcription pass to avoid concurrent
        # model execution and races on shared transcription state
        self.chunked_stop.set()
        if hasattr(self, 'chunk_thread'):
            self.chunk_thread.join()

        # Hide recording indicator
        self.indicator.stop()

        # Update UI
        self.title = "🎙️ (Transcribing)"
        self.status_item.title = "Status: Transcribing..."
        logger.info("Recording stopped. Transcribing...")

        # Process in background
        transcribe_thread = threading.Thread(target=self.process_recording)
        transcribe_thread.start()

    def process_recording(self):
        # Transcribe and insert text
        try:
            self.transcribe_audio()
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            self.status_item.title = "Status: Error during transcription"
        finally:
            self.title = "🎙️"  # Reset title

    def record_audio(self):
        # Build kwargs for audio stream
        stream_kwargs = {
            'format': self.format,
            'channels': self.channels,
            'rate': self.rate,
            'input': True,
            'frames_per_buffer': self.chunk
        }
        # Use selected input device if specified
        if self.selected_input_device is not None:
            stream_kwargs['input_device_index'] = self.selected_input_device

        stream = self.audio.open(**stream_kwargs)
        try:
            while self.recording:
                data = stream.read(self.chunk)
                with self.frames_lock:
                    self.frames.append(data)

                # Update indicator with audio level
                self.indicator.update_audio_level(data)
        finally:
            stream.stop_stream()
            stream.close()

    def chunked_transcribe_loop(self):
        """Periodically transcribe accumulated audio while recording."""
        frames_per_interval = (self.rate * self.CHUNK_INTERVAL_SECONDS) // self.chunk

        while not self.chunked_stop.is_set():
            self.chunked_stop.wait(timeout=0.5)
            if self.chunked_stop.is_set():
                break

            with self.frames_lock:
                current_frame_count = len(self.frames)
                new_frames = current_frame_count - self.last_transcribed_frame_count
                if new_frames < frames_per_interval:
                    continue
                frames_snapshot = list(self.frames[:current_frame_count])
            if not frames_snapshot:
                continue

            try:
                audio_data = np.frombuffer(b''.join(frames_snapshot), dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                audio_duration = len(audio_float) / self.rate

                chunk_start = time.time()
                result = mlx_whisper.transcribe(
                    audio_float,
                    path_or_hf_repo=self.model_path,
                    language="en",
                    initial_prompt=self.initial_prompt,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
                chunk_elapsed = time.time() - chunk_start

                text = result.get("text", "").strip()
                rtf = chunk_elapsed / audio_duration if audio_duration > 0 else 0
                logger.info(f"[METRICS] Chunk transcription: {chunk_elapsed*1000:.0f}ms "
                            f"(audio: {audio_duration:.2f}s, RTF: {rtf:.2f}x)")

                self.partial_text = text
                self.last_transcribed_frame_count = current_frame_count
            except Exception as e:
                logger.error(f"Error in chunked transcription: {e}")

    def transcribe_audio(self):
        with self.frames_lock:
            if not self.frames:
                self.title = "🎙️"
                self.status_item.title = "Status: No audio recorded"
                logger.warning("No audio recorded")
                return
            total_frame_count = len(self.frames)
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)

        pipeline_start = time.time()
        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_duration = len(audio_float) / self.rate

        convert_elapsed = time.time() - pipeline_start
        logger.info(f"[METRICS] Audio duration: {audio_duration:.2f}s | Buffer conversion: {convert_elapsed*1000:.0f}ms")

        try:
            has_partial = bool(self.partial_text) and self.last_transcribed_frame_count > 0

            if has_partial and self.last_transcribed_frame_count >= total_frame_count:
                # All audio already transcribed by chunked loop
                text = self.partial_text
                logger.info(f"[METRICS] Using cached transcription (no tail to process)")
            elif has_partial:
                tail_samples = total_frame_count - self.last_transcribed_frame_count
                tail_duration = (tail_samples * self.chunk) / self.rate
                if tail_duration < 0.5:
                    text = self.partial_text
                    logger.info(f"[METRICS] Tail too short ({tail_duration:.2f}s), using partial result")
                else:
                    # Re-transcribe full audio for best accuracy at boundaries
                    transcribe_start = time.time()
                    result = mlx_whisper.transcribe(
                        audio_float,
                        path_or_hf_repo=self.model_path,
                        language="en",
                        initial_prompt=self.initial_prompt,
                        temperature=0.0,
                        condition_on_previous_text=True,
                    )
                    transcribe_elapsed = time.time() - transcribe_start
                    text = result.get("text", "").strip()
                    rtf = transcribe_elapsed / audio_duration if audio_duration > 0 else 0
                    logger.info(f"[METRICS] Final transcription (with tail): {transcribe_elapsed*1000:.0f}ms | RTF: {rtf:.2f}x")
            else:
                # No partial results (short recording) — full transcription
                transcribe_start = time.time()
                result = mlx_whisper.transcribe(
                    audio_float,
                    path_or_hf_repo=self.model_path,
                    language="en",
                    initial_prompt=self.initial_prompt,
                    temperature=0.0,
                    condition_on_previous_text=True,
                )
                transcribe_elapsed = time.time() - transcribe_start
                text = result.get("text", "").strip()
                rtf = transcribe_elapsed / audio_duration if audio_duration > 0 else 0
                logger.info(f"[METRICS] Full transcription (no partials): {transcribe_elapsed*1000:.0f}ms | RTF: {rtf:.2f}x")

            if text:
                # Only attempt selected-text detection when text enhancement
                # is explicitly enabled.  The detection simulates Cmd+C which
                # can leak a literal 'c' into the target app, so it must be
                # gated behind an opt-in flag.
                enable_enhance = os.getenv('ENABLE_TEXT_ENHANCEMENT', 'false').lower() == 'true'
                selected_text = None
                if enable_enhance and self.bedrock_client.is_available():
                    selected_text = self.text_selector.get_selected_text()
                    logger.debug(f"Selected text: {selected_text}")

                if selected_text and self.bedrock_client.is_available():
                    logger.info(f"Selected text detected: {selected_text[:50]}...")
                    logger.info(f"Voice instruction: {text}")

                    try:
                        # Use Bedrock to enhance the selected text
                        self.status_item.title = "Status: Enhancing text with AI..."
                        enhance_start = time.time()
                        enhanced_text = self.bedrock_client.enhance_text(text, selected_text)
                        enhance_elapsed = time.time() - enhance_start
                        logger.info(f"[METRICS] AI enhancement: {enhance_elapsed*1000:.0f}ms")

                        # Replace selected text with enhanced version
                        self.text_selector.replace_selected_text(enhanced_text)
                        logger.info(f"Enhanced text: {enhanced_text}")
                        self.status_item.title = f"Status: Enhanced: {enhanced_text[:30]}..."

                    except Exception as e:
                        logger.error(f"Error enhancing text: {e}")
                        # Fallback to normal text insertion
                        self.insert_text(text)
                        logger.info(f"Transcription (fallback): {text}")
                        self.status_item.title = f"Status: Transcribed: {text[:30]}..."
                else:
                    # No selected text or Bedrock unavailable - normal insertion
                    insert_start = time.time()
                    self.insert_text(text)
                    insert_elapsed = time.time() - insert_start
                    logger.info(f"[METRICS] Text insertion: {insert_elapsed*1000:.0f}ms")
                    logger.info(f"Transcription: {text}")
                    self.status_item.title = f"Status: Transcribed: {text[:30]}..."
            else:
                logger.warning("No speech detected")
                self.status_item.title = "Status: No speech detected"

            total_elapsed = time.time() - pipeline_start
            logger.info(f"[METRICS] Total pipeline: {total_elapsed*1000:.0f}ms (audio: {audio_duration:.2f}s)")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.status_item.title = "Status: Transcription error"
            raise

    def insert_text(self, text):
        # Small delay to let any pending key events (e.g. Globe/Fn) fully propagate
        # before typing, preventing stray characters from being prepended
        time.sleep(0.05)
        # Type text at cursor position without altering the clipboard
        logger.debug("Typing text at cursor position...")
        self.keyboard_controller.type(text)
        logger.debug("Text typed successfully")

    def handle_shutdown(self, _signal, _frame):
        """This method is no longer used with the global handler approach"""
        pass

# Wrap the main execution in a try-except to ensure clean exit
if __name__ == "__main__":
    try:
        WhisperDictationApp().run()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received, exiting...")
        os._exit(0)
