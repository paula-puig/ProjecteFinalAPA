import numpy as np
import soundfile as sf
import sounddevice as sd
from pydub import AudioSegment
#import librosa

class AudioProcessor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.audio, self.samplerate = sf.read(input_file)

    def save_audio(self, output_file):
        sf.write(output_file, self.audio, self.samplerate)

    def play_audio(self):
        sd.play(self.audio, self.samplerate)
        sd.wait()

    def convert_to_wav(self, output_file):
        audio = AudioSegment.from_mp3(self.input_file)
        audio.export(output_file, format="wav")

    def trim_audio(self, output_file, start_time, duration):
        start_sample = int(start_time * self.samplerate)
        end_sample = int((start_time + duration) * self.samplerate)
        trimmed_audio = self.audio[start_sample:end_sample]
        sf.write(output_file, trimmed_audio, self.samplerate)


class RobotEffect(AudioProcessor):
    def apply_robot_effect(self, output_file, modulation_factor=0.8, pitch_factor=0.9):
        mod = np.sin(2 * np.pi * np.arange(0, len(self.audio)) * (1.0 / self.samplerate) * modulation_factor)
        adjusted_audio = self.audio * mod[:, np.newaxis]
        max_value = np.max(np.abs(adjusted_audio))
        modulated_audio = adjusted_audio / max_value
        robot_audio = modulated_audio * pitch_factor
        sf.write(output_file, robot_audio, self.samplerate)


class EchoEffect(AudioProcessor):
    def apply_echo_effect(self, output_file, delay=0.7, decay=0.5):
        delay_samples = int(delay * self.samplerate)
        output_audio = np.zeros_like(self.audio)
        for i in range(len(self.audio)):
            if i >= delay_samples:
                output_audio[i] = self.audio[i] + decay * output_audio[i - delay_samples]
            else:
                output_audio[i] = self.audio[i]
        sf.write(output_file, output_audio, self.samplerate)
        
class FlangerEffect(AudioProcessor):
    def apply_flanger_effect(self, output_file, delay=0.003, depth=0.002, rate=0.2):
        delay_samples = int(delay * self.samplerate)
        depth_samples = int(depth * self.samplerate)
        modulator = depth_samples * np.sin(2 * np.pi * rate * np.arange(len(self.audio)) / self.samplerate)
        flanger_audio = np.zeros_like(self.audio)
        for i in range(len(self.audio)):
            if i >= delay_samples:
                index = int(i - delay_samples + modulator[i])
                flanger_audio[i] = self.audio[i] + self.audio[index]
            else:
                flanger_audio[i] = self.audio[i]
        sf.write(output_file, flanger_audio, self.samplerate)

class PitufoEffect(AudioProcessor):
    def apply_pitufo_effect(self, output_file, modulation_factor=0.1, pitch_factor=0.9):
        num_channels = self.audio.shape[1]
        mod = (2 * np.pi * np.arange(0, len(self.audio)) * (1.0 / self.samplerate) * modulation_factor)

        for channel in range(num_channels):
            audio_1d = self.audio[:, channel]    
            mod_channel = mod[:len(audio_1d)]  
            indices = np.arange(len(audio_1d)) + mod_channel * self.samplerate * pitch_factor
            modulated_audio = np.interp(indices, np.arange(len(audio_1d)), audio_1d)
            max_value = np.max(np.abs(modulated_audio))
            modulated_audio /= max_value
            self.audio[:, channel] = modulated_audio
        sf.write(output_file, self.audio, self.samplerate)

class LowEffect(AudioProcessor):
    def apply_low_effect(self, output_file, modulation_factor=0.1, pitch_factor=0.9):
        num_channels = self.audio.shape[1]
        mod = -(2 * np.pi * np.arange(0, len(self.audio)) * (1.0 / self.samplerate) * modulation_factor)

        for channel in range(num_channels):
            audio_1d = self.audio[:, channel]    
            mod_channel = mod[:len(audio_1d)]  
            indices = np.arange(len(audio_1d)) + mod_channel * self.samplerate * pitch_factor
            modulated_audio = np.interp(indices, np.arange(len(audio_1d)), audio_1d)
            max_value = np.max(np.abs(modulated_audio))
            modulated_audio /= max_value
            self.audio[:, channel] = modulated_audio
        sf.write(output_file, self.audio, self.samplerate)

class LowPassFilter(AudioProcessor):
    def apply_lowpass_filter(self, output_file, cutoff_frequency):
        filtered_audio = self.audio.low_pass_filter(cutoff_frequency)
        filtered_audio.export(output_file, format='wav')
    
# Ejemplo de uso de las clases y la interfaz
input_file = "Segment.wav"

# Crear instancia de la clase AudioProcessor y convertir a WAV
#audio_processor = AudioProcessor(input_file)
#output_file_wav = "Summer_Wine.wav"
#audio_processor.convert_to_wav(output_file_wav)

# Crear instancia de la clase RobotEffect y aplicar efecto de robot
robot_effect = RobotEffect(input_file)  ##CANVIAAR
output_file_robot = "Robot.wav"
robot_effect.apply_robot_effect(output_file_robot)

# Crear instancia de la clase EchoEffect y aplicar efecto de eco
echo_effect = EchoEffect(input_file)
output_file_echo = "Echo.wav"
echo_effect.apply_echo_effect(output_file_echo)

flanger_effect = FlangerEffect(input_file)
output_file_flanger = "Flanger.wav"
flanger_effect.apply_flanger_effect(output_file_flanger)

pitufo_effect = PitufoEffect(input_file)
output_file_pitufo = "Pitufo.wav"
pitufo_effect.apply_pitufo_effect(output_file_pitufo)

low_effect = LowEffect(input_file)
output_file_low = "Low.wav"
low_effect.apply_low_effect(output_file_low)

lowPass_effect = LowPassFilter(input_file)
output_file_lowPass = "Low.wav"
lowPass_effect.apply_lowpass_filter(output_file_lowPass)

# Reproducir el audio original
#audio_processor.play_audio()

# Reproducir el audio con el efecto de robot
#robot_effect.play_audio()

# Reproducir el audio con el efecto de eco
#echo_effect.play_audio()

