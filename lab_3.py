import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

## Ruta para los audios, saber estan los audios

time_Dayanna, Audio_Dayanna = wav.read("Audios/Audio_Dayanna.wav")
time_David, Audio_David = wav.read("Audios/Audio_David.wav")
time_Sofia, Audio_Sofia = wav.read("Audios/Audio_Sofia.wav")
                                  
fs = 44100 ##frecuencia de la aplicacion
time_demora_Dayanna = len (Audio_Dayanna)/time_Dayanna
time_demora_David = len (Audio_David)/time_David
time_demora_Sofia = len (Audio_Sofia)/time_Sofia

##Tiempos Mostrados en la terminal

print (f"Tiempo Duracion del Audio de Dayanna: {time_demora_Dayanna: 3f}s")
print (f"Tiempo Duracion del Audio de David: {time_demora_David: 3f}s")
print (f"Tiempo Duracion del Audio de Sofia: {time_demora_Sofia: 3f}s")

escala = 10
Dayanna = Audio_Dayanna[:: escala]
David = Audio_David[:: escala]
Sofia = Audio_Sofia[:: escala]
tiempo_Dayanna = np.linspace(0, time_demora_Dayanna , len (Dayanna))
tiempo_David = np.linspace(0, time_demora_David , len (David))
tiempo_Sofia = np.linspace(0, time_demora_Sofia , len (Sofia))

plt.figure(figsize=(12,6))
plt.subplot(3, 1, 1)
plt.plot(tiempo_Dayanna, Dayanna, color="orange")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (dB)")
plt.title("Audio Dayanna")
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 2)
plt.plot(tiempo_David, David, color="red")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (dB)")
plt.title("Audio David")
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.plot(tiempo_Sofia, Sofia, color="violet")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (dB)")
plt.title("Audio Sofia")
plt.legend()
plt.tight_layout()
plt.show()

##Calculo Ruido SNR en el terminal

time_ruido, audio_ruido = wav.read("Audios/Ruido_Ambiente.wav")
ambiente = audio_ruido [::escala]

def calculate_snr(audio, ambiente):
    audio_power = np.mean(audio ** 2)
    noise_power = np.var(audio - np.mean(ambiente)) 
    snr = 10 * np.log10(audio_power / noise_power)
    return snr

snr_Dayanna = calculate_snr(Dayanna, ambiente)
snr_David = calculate_snr(David, ambiente)
snr_Sofia = calculate_snr(Sofia, ambiente)

print (f"SNR de Dayanna: {snr_Dayanna:.2f} dB")
print (f"SNR de David: {snr_David:.2f} dB") ##Mayor numero: menor ruido ambiente
print (f"SNR de Sofia: {snr_Sofia:.2f} dB")

##Transformada de fourier discreta

t = np.linspace(0, 1, fs, endpoint=False) ## numero de datos
N = len(t)

frecuencia = np.fft.fftfreq(N, 1/fs) #muestreo
spectrum_Dayanna = np.fft.fft(Dayanna) / N
magnitud_Dayanna = np.abs(spectrum_Dayanna[:N//2]) ##cantidad de canales

spectrum_David = np.fft.fft(David) / N
magnitud_David = np.abs(spectrum_David[:N//2])

spectrum_Sofia = np.fft.fft(Sofia) / N
magnitud_Sofia = np.abs(spectrum_Sofia[:N//2])


plt.figure(figsize=(12,6))
plt.subplot (3,1,1)
plt.plot(frecuencia[:N//2], magnitud_Dayanna, color="blue")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro señal normalizado Dayanna")
plt.grid()
plt.tight_layout()

plt.subplot (3,1,2)
plt.plot(frecuencia[:N//2], magnitud_David, color="green")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro señal normalizado David")
plt.grid()
plt.tight_layout()

plt.subplot (3,1,3)
plt.plot(frecuencia[:N//2], magnitud_Sofia, color="yellow")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro señal normalizado Sofia")
plt.grid()
plt.tight_layout()
plt.show


##Audios mejorados

audio_ICA = "C:/Users/User/Documents/PDS_LAB3/Audios_mejorados/"

minutos = min(len(Audio_Dayanna), len (David), len (Sofia))
nuevo_audio_Dayanna = Audio_Dayanna[:minutos]
nuevo_audio_David = Audio_David[:minutos]
nuevo_audio_Sofia = Audio_Sofia[:minutos]

audios_mixtos = np.column_stack((nuevo_audio_Dayanna, nuevo_audio_David, nuevo_audio_Sofia))
ICA = FastICA (n_components=3) ##3 porque son 3 audios
audios_separados = ICA.fit_transform(audios_mixtos)
##Normalizar para evitar saturación
filtro = np.int16 (audios_separados / np.max(np.abs(audios_separados))*3276)

wav.write(audio_ICA + "audio_ICA_Dayanna.wav", time_Dayanna, filtro[:,0])
wav.write(audio_ICA + "audio_ICA_David.wav", time_David, filtro[:,1])
wav.write(audio_ICA + "audio_ICA_Sofia.wav", time_Sofia, filtro[:,2])

print ("Se guardo las señales filtradas con ICA")




