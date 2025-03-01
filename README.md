# PDS_LAB3

Para la práctica número 3, se abordó el desafío de separar y analizar señales de audio provenientes de los tres integrantes de nuestro grupo hablando simultáneamente en una sala, capturadas mediante tres micrófonos ubicados a cierta distancia de cada uno. Posteriormente, se realizó un análisis temporal y espectral de las grabaciones, incluyendo la aplicación de la Transformada de Fourier para identificar las componentes de frecuencia de cada voz, posteriormente, se implementó el algoritmo de Análisis de Componentes Independientes (ICA) con el objetivo de aislar las fuentes individuales a partir de las señales mezcladas. A lo largo del proceso, se calcularon métricas clave como la relación señal-ruido (SNR) para evaluar la calidad de las grabaciones y se generaron visualizaciones que permitieron comprender mejor el comportamiento de las señales en los dominios del tiempo y la frecuencia. Este trabajo sienta las bases para futuras mejoras en la separación de fuentes y el filtrado de ruido en entornos acústicos complejos.

## 1. Importación de Librerías

    import scipy.io.wavfile as wav
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import FastICA

En esta sección, se importan las librerías necesarias para el procesamiento de señales, específicamente las siguientes dos son funamentales: scipy.io.wavfile se utiliza para leer y escribir archivos de audio en formato .wav, lo que es esencial para trabajar con señales de voz grabadas; sklearn.decomposition.FastICA se emplea para implementar el algoritmo de Análisis de Componentes Independientes (ICA), que es clave para separar las fuentes de audio mezcladas. Estas herramientas son la base para el análisis y procesamiento de señales en el proyecto.

## 2. Lectura y Cálculo del Tiempo de Duración de los Audios

    time_Dayanna, Audio_Dayanna = wav.read("Audios/Audio_Dayanna.wav")
    time_David, Audio_David = wav.read("Audios/Audio_David.wav")
    time_Sofia, Audio_Sofia = wav.read("Audios/Audio_Sofia.wav")
                                  
    fs = 44100 ##frecuencia de la aplicacion
    time_demora_Dayanna = len (Audio_Dayanna)/time_Dayanna
    time_demora_David = len (Audio_David)/time_David
    time_demora_Sofia = len (Audio_Sofia)/time_Sofia

Cada archivo contiene dos datos principales: la frecuencia de muestreo y las amplitudes de la señal en el dominio del tiempo. La frecuencia de muestreo es crucial para entender la resolución temporal de la señal, mientras que las amplitudes representan la intensidad de la voz en cada instante. Posteriormente se calcula la duración de cada audio en segundos, esto se hace dividiendo la longitud de cada arreglo de audio entre la frecuencia de muestreo, este cálculo es importante porque permite conocer la extensión temporal de cada señal, lo que es útil para sincronizar y comparar los audios. Los resultados muestran que los audios tienen duraciones de: 

Dayanna: 8.404 s 
David: 7.848 s 
Sofia: 6.505 s 

Lo que puede causar problemas al procesarlos juntos, especialmente en técnicas como ICA, que requieren que todas las señales tengan la misma longitud.

## 3. Reducción de Resolución de Audios y Creación Vector Tiempo

    escala = 10
    Dayanna = Audio_Dayanna[:: escala]
    David = Audio_David[:: escala]
    Sofia = Audio_Sofia[:: escala]
    tiempo_Dayanna = np.linspace(0, time_demora_Dayanna , len (Dayanna))
    tiempo_David = np.linspace(0, time_demora_David , len (David))
    tiempo_Sofia = np.linspace(0, time_demora_Sofia , len (Sofia))

Aquí se reduce la resolución de los audios tomando una muestra cada 10 puntos (escala = 10), esto disminuye la frecuencia de muestreo efectiva de 44,100 Hz a 4,410 Hz, lo que facilita el procesamiento al reducir la cantidad de datos. Sin embargo, esta reducción puede causar pérdida de información, especialmente en frecuencias altas, lo que afecta la calidad de la señal. Esta técnica es útil para pruebas iniciales. Seguido, se crea un vector de tiempo para cada audio, que va desde 0 hasta la duración del audio, con un número de puntos igual a la longitud del audio reducido, este vector es esencial para graficar las señales en el dominio del tiempo, ya que proporciona una referencia temporal para las amplitudes.

## 4. Graficación Señales en Dominio del Tiempo

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(tiempo_Dayanna, Dayanna, color="orange")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (dB)")
    plt.title("Audio Dayanna")
    plt.legend()
    plt.tight_layout()

(El código anterior se repitió con cada uno de los integrantes)

![Image](Imagenes/audios_usados.png)

En esta imagen encontramos tres gráficas de onda donde cada una corresponde a un integrante, vamos a comenzar por el análisis de cada una:

La gráfica de color naranja la cual corresponde al audio de Dayanna esta tiene una amplitud relativamente moderada con picos distribuidas a lo largo del tiempo, lo que significa un tono de voz estable con variación de intensidad, este audio se demoró más de ocho segundos y llego a tener una amplitud de 10.000 db.

La gráfica de color rojo la cual pertenece al audio de David presenta una mayor amplitud, con picos más pronunciados, especialmente al inicio por lo que se puede decir que es una voz más fuerte, este audio se demoro mas de siete segundos y llego a tener una amplitud mayor a 10.000 db.

La gráfica rosada la cual pertenece al audio de Sofia nos permite ver que es menos intensa en términos de amplitud, con una menos variabilidad en los picos, por lo que se puede decir que es un tono de voz muy suave, este audio se demora mas de seis segundos y llego a tener una amplitud de 10.000 db pero al final.

Para las tres gráficas también se tienen factores que la afectan como la ubicación del micrófono ya que si está lejos se tiene menos amplitud, al tener ruido de fondo también se puede ver afectada ya que este se puede reflejar en las amplitudes pequeñas y por último el micrófono ya que algunos tienen una sensibilidad más alta que otros.

## 5. Cálculo SNR (Relación Señal-Ruido)

    time_ruido, audio_ruido = wav.read("Audios/Ruido_Ambiente.wav")
    ambiente = audio_ruido[::escala]

    def calculate_snr(audio, ambiente):
        audio_power = np.mean(audio ** 2)
        noise_power = np.var(audio - np.mean(ambiente))
        snr = 10 * np.log10(audio_power / noise_power)
        return snr

    snr_Dayanna = calculate_snr(Dayanna, ambiente)
    snr_David = calculate_snr(David, ambiente)
    snr_Sofia = calculate_snr(Sofia, ambiente)

    print(f"SNR de Dayanna: {snr_Dayanna:.2f} dB")
    print(f"SNR de David: {snr_David:.2f} dB")
    print(f"SNR de Sofia: {snr_Sofia:.2f} dB")

### Resultados del SNR: 

Dayanna: -27.23 dB (mejor SNR)

David: -32.32 dB (peor SNR)

Sofía: -32.01 dB*

Aquí se calcula la relación señal-ruido (SNR) para cada audio. El SNR es una métrica clave para evaluar la calidad de la señal, ya que indica cuánto más fuerte es la señal en comparación con el ruido ambiental. En este caso, los valores de SNR son negativos lo que indica que el ruido es más fuerte que la señal. Esto sugiere que el ruido ambiental es significativo y afecta la calidad de los audios; un valor más bajo indica una mayor proporción de ruido en comparación con la señal, guiandonos de esto y con los resultados de cada audio pudimos concluir que: 

Dayanna al tener un SNR de -27.23 dB el cual se podría decir que a comparación de losotros dos fue el mejor nos indica que su voz es más clara en la grabación, con menos interferencia de ruido. Esto se refleja en su espectro, que muestra una mayor presencia de frecuencias altas.

David y Sofía tienen un SNR más bajo (-32.32 dB y -32.01 dB), lo que indica que su grabación tiene más ruido en comparación con la señal de voz. Esto podría explicar por qué sus espectros tienen menos energía en altas frecuencias.

## 6. Transformada de Fourier Discreta (DFT)

En esta sección, se aplica la Transformada de Fourier Discreta (DFT) para analizar las señales en el dominio de la frecuencia. La DFT convierte las señales del dominio del tiempo al dominio de la frecuencia, lo que permite identificar las frecuencias dominantes en cada voz.

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

![Image](Imagenes/espectro_señales_normalizadas.png)

Como se puede ver las gráficas nos muestran el espectro de magnitud de señales de audionormalizadas para tres personas: Dayanna, David y Sofía, en el cual cada gráficorepresenta la magnitud de la Transformada de Fourier de la señal de voz de cada persona. El eje x representa la frecuencia en Hz , mostrando las componentes espectrales de la señal mientras que en el eje y encontramos representa la magnitud indicando la energía decada frecuencia.

Un mayor valor en una frecuencia específica sugiere que dicha frecuencia está máspresente en la voz de la persona.

Dayanna Gráfica azul: En la gráfica de Dayana podemos observar que se presentan picospronunciados en bajas y medias frecuencias por debajo de 5000 Hz, también hay una distribución de energía en altas frecuencias mayores a 10000 Hz aunque con menorintensidad, esto nos permite observar una estructura rica en frecuencias, lo que sugiere una voz con timbre variado y posible mayor presencia de armónicos.

David Gráfica verde: En la gráfica de David encontramos un espectro el cual nos muestra una concentración de energía más marcada en bajas y medias frecuencias 0 - 5000 Hz, también se observa una menor presencia de altas frecuencias en comparación con Dayanna, lo que nos podría indicar que la voz de David tiene un tono más grave y menos armónicos en frecuencias altas.

Sofía Gráfica amarilla : En la gráfica de Sofía ya observamos que la mayor parte de la energía está concentrada en frecuencias muy bajas 0 - 3000 Hz y también se presenta muy poca energía en altas frecuencias, lo que sugiere una voz más suave o con menos variabilidad espectral esto puede indicar una voz más monótona o menos rica en armónicos altos

### Relación entre SNR y Espectro

El espectro de Dayanna es más amplio y rico en armónicos, lo cual se alinea con su mejor SNR: su señal de voz es más fuerte que el ruido. David y Sofía tienen SNR más bajo y menos contenido en altas frecuencias, lo que nos indica que sus voces pueden estar más enmascaradas por ruido o que las frecuencias altas se han perdido debido a la calidad de la grabación.

## 7. Separación de Fuentes con ICA

    audios_mixtos = np.column_stack((nuevo_audio_Dayanna, nuevo_audio_David, nuevo_audio_Sofia))
    ICA = FastICA(n_components=3)
    audios_separados = ICA.fit_transform(audios_mixtos)
    filtro = np.int16(audios_separados / np.max(np.abs(audios_separados)) * 3276)

    wav.write(audio_ICA + "audio_ICA_Dayanna.wav", time_Dayanna, filtro[:, 0])
    wav.write(audio_ICA + "audio_ICA_David.wav", time_David, filtro[:, 1])
    wav.write(audio_ICA + "audio_ICA_Sofia.wav", time_Sofia, filtro[:, 2])

    print("Se guardo las señales filtradas con ICA")

En esta parte, se aplica el algoritmo de Análisis de Componentes Independientes (ICA) para separar las fuentes de audio. El audio ICA (Análisis de componentes independientes) este es una técnica que permite el procesamiento de señales que usa como un filtro para separar fuentes de sonido contaminas de ruido o para mejorar la calidad del audio.

Como análisis ante el mejoramiento del audio tomado por cada intégrate con el audio ICA nos reduce en gran cantidad el audio principalmente tomado ya que este nos elimina ruidos o señales irrelevantes dejando la parte más importante y al separarlos la señal se vuelve más corta y algo muy importante que se tiene que tener en cuanta es que al utilizar mal el audio ICA este puede quitar parte importante del audio original.

## 8. Conclusión

En este proyecto, se abordó el desafío de separar y analizar señales de audio provenientes de tres personas hablando simultáneamente en una sala. Se realizó un análisis temporal y espectral de las grabaciones, incluyendo la aplicación de la Transformada de Fourier para identificar las componentes de frecuencia de cada voz. Además, se implementó el algoritmo de Análisis de Componentes Independientes (ICA) con el objetivo de aislar las fuentes individuales a partir de las señales mezcladas. A lo largo del proceso, se calcularon métricas clave como la relación señal-ruido (SNR) para evaluar la calidad de las grabaciones y se generaron visualizaciones que permitieron comprender mejor el comportamiento de las señales en los dominios del tiempo y la frecuencia. Este trabajo sienta las bases para futuras mejoras en la separación de fuentes y el filtrado de ruido en entornos acústicos complejos.
