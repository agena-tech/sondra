import wave
import subprocess
from piper import PiperVoice, SynthesisConfig

# 📌 Model yolu
model_path = "tr_TR-fettah-medium.onnx"

# 📌 Metin
text = "Merhaba Anezatra, bu ses artık yaşlı değil, daha genç ve ince."

# 📌 Piper modeli yükle
voice = PiperVoice.load(model_path)

# 📌 Ses ayarları (senin CLI ayarların)
syn_config = SynthesisConfig(
    length_scale=0.9,
    noise_scale=0.7,
    noise_w_scale=0.8,
)

# 📌 WAV üret
with wave.open("out.wav", "wb") as wav_file:
    voice.synthesize_wav(text, wav_file, syn_config=syn_config)

print("Piper çıktı üretildi: out.wav")

# 📌 ffmpeg ile pitch + echo (senin sweet spot ayarın)
subprocess.run([
    "ffmpeg",
    "-i", "out.wav",
    "-af", "asetrate=22050*1.12,aresample=22050,aecho=0.4:0.5:350:0.15",
    "final.wav"
])

print("Final ses hazır: final.wav")