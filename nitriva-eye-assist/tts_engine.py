import subprocess
import threading

class NitrivaVoiceEngine:
    """
    Text-to-Speech audio synthesizer in Python.
    Uses native Windows System.Speech synthesizer via PowerShell in a background thread.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def speak(self, text):
        if not text:
            return

        def _run():
            with self._lock:
                clean_text = text.replace("'", "").replace('"', "")
                ps_script = f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_text}')"
                cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"Add-Type -AssemblyName System.Speech; {ps_script}"]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print("TTS error:", e)

        threading.Thread(target=_run, daemon=True).start()
