import numpy as np
import matplotlib.pyplot as plt
import scipy
N=1000
t=np.linspace(0,1e-3,N) # un eje de tiempo de 1ms muestreado a 1us genera 1000 muestras
ts=t[1] # el primer diferencia entre la posicion "0" del vector de tiempo y la "1" es el tiempo de muestreo
fs=1/ts # la frecuencia de muestreo es la inversa del tiempo de muestreo
fmsj=1500 # frecuencia del mensaje Hz
fc=80000 # frecuencia de la portadora Hz

msj=np.sin(2*np.pi*fmsj*t)
carrier=np.cos(2*np.pi*fc*t)

plt.figure()
plt.plot(t,msj,label="mensaje")
plt.plot(t,carrier,label="portadora")
plt.legend(loc='upper right')
plt.xlabel("t[s]")
plt.title("Portadora y Mensaje")


######################
### la modulación ####
######################
def M_DBL(a,b):  # es muy sencillo...solamente es el producto, que en la practica se hace con un mezclador aca con numeros
    return a*b

DBL= M_DBL(msj,carrier)
plt.figure()
plt.plot(t,DBL,label="modulada")
plt.plot(t,msj,label="mensaje")
plt.legend(loc='upper right')
plt.xlabel("t[s]")
plt.title("Demodulada y mensaje")


plt.figure()
plt.plot(t,DBL)
plt.plot(t,DBL,label="modulada")
plt.legend(loc='upper right')
plt.xlabel("t[s]")
plt.title("Demodulada DBL")


fcia=np.linspace(-fs/2,fs/2,N)
Fcarrier=np.abs(np.fft(np.fft.fft(carrier)))/N
Fmsj=np.abs(np.fft.fftshift(np.fft.fft(msj)))/N
plt.figure()
plt.plot(fcia,Fcarrier,label="portadora")
plt.plot(fcia,Fmsj,label="mensaje")
plt.legend(loc='upper right')
plt.xlabel("f[Hz]")
plt.title("Espectro")


FDBL=np.abs(np.fft.fftshift(np.fft.fft(DBL)))/N
plt.figure()
plt.plot(fcia,FDBL,label="DBL")
plt.legend(loc='upper right')
plt.xlabel("f[Hz]")
plt.title("Espectro")

DBL_rx=DBL*carrier

FDBL_rx=np.abs(np.fft.fftshift(np.fft.fft(DBL_rx)))/N
plt.figure()
plt.plot(fcia,FDBL_rx)
plt.xlabel("f[Hz]")
plt.title("Producto de DBL.carrier")

from scipy import signal as dsp
taps=dsp.firwin(N,cutoff=20000,fs=fs)
LPF=np.abs(np.fft.fftshift(np.fft.fft(taps)))/N
plt.figure()
plt.plot(fcia,LPF*(N/4),label="LPF")  ## escale en (N/4) asi el filtro que sobre la es espectro como en el libro
plt.plot(fcia,FDBL_rx,label="DBL.Carrier")
plt.legend(loc='upper right')
plt.xlabel("f[Hz]")
plt.title("Espectro")

def DEMOD_DBL(a,b,taps):
    c=a*b
    return np.convolve(c,taps,'same')

Rx=DEMOD_DBL(DBL,carrier,taps)

Rx=np.convolve(taps,DBL_rx,'same')
plt.figure()
plt.plot(t,Rx,label="recuperado")
plt.plot(t,msj,label="original")
plt.legend(loc='upper right')
plt.title("Mensaje Original y Modulado")
plt.xlabel("t[ms]")

plt.show()