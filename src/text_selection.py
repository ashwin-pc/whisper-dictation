#!/usr/bin/env python3
import pyperclip
import time
from AppKit import NSPasteboard, NSStringPboardType
from pynput.keyboard import Key, Controller
from logger_config import setup_logging

logger = setup_logging()


class TextSelection:
    def __init__(self):
        self.keyboard_controller = Controller()
    
    def get_selected_text(self):
        """
        Get currently selected text using clipboard manipulation.
        Returns the selected text or None if no text is selected.
        """
        try:
            # Save current clipboard content
            original_clipboard = pyperclip.paste()
            
            # Clear clipboard to detect if copy operation succeeds
            pyperclip.copy("")
            time.sleep(0.1)
            
            # Copy selected text to clipboard
            with self.keyboard_controller.pressed(Key.cmd):
                time.sleep(0.05)
                self.keyboard_controller.press('c')
                self.keyboard_controller.release('c')
                time.sleep(0.05)

            # Small delay to ensure copy operation completes
            time.sleep(0.2)
            
            # Get the copied text
            selected_text = pyperclip.paste()
            logger.debug(f"Copied text: {selected_text}")
            
            # Restore original clipboard content
            pyperclip.copy(original_clipboard)
            
            # Return selected text if it's not empty and different from original
            if selected_text and selected_text != original_clipboard:
                return selected_text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting selected text: {e}")
            return None
    
    def replace_selected_text(self, new_text):
        """
        Replace currently selected text with new text.
        If no text is selected, inserts at cursor position.
        """
        try:
            # Type the replacement text
            self.keyboard_controller.type(new_text)
            return True
            
        except Exception as e:
            logger.error(f"Error replacing selected text: {e}")
            return False
    
    def select_all_and_replace(self, new_text):
        """
        Select all text in current field and replace with new text.
        Fallback method when specific text selection fails.
        """
        try:
            # Select all text
            with self.keyboard_controller.pressed(Key.cmd):
                time.sleep(0.05)
                self.keyboard_controller.press('a')
                self.keyboard_controller.release('a')
                time.sleep(0.05)
            
            time.sleep(0.1)
            
            # Type replacement text
            self.keyboard_controller.type(new_text)
            return True
            
        except Exception as e:
            logger.error(f"Error in select all and replace: {e}")
            return False
    
    def get_selected_text_native(self):
        """
        Alternative method using NSPasteboard directly.
        This is a backup method that might work better in some scenarios.
        """
        try:
            # Get the general pasteboard
            pasteboard = NSPasteboard.generalPasteboard()
            
            # Save current pasteboard content
            original_content = pasteboard.stringForType_(NSStringPboardType)
            
            # Clear pasteboard
            pasteboard.clearContents()
            
            # Copy selected text
            with self.keyboard_controller.pressed(Key.cmd):
                time.sleep(0.05)
                self.keyboard_controller.press('c')
                self.keyboard_controller.release('c')
                time.sleep(0.05)

            time.sleep(0.2)
            
            # Get copied text
            selected_text = pasteboard.stringForType_(NSStringPboardType)
            
            # Restore original content
            if original_content:
                pasteboard.clearContents()
                pasteboard.setString_forType_(original_content, NSStringPboardType)
            
            return selected_text.strip() if selected_text else None
            
        except Exception as e:
            logger.error(f"Error with native text selection: {e}")
            return None