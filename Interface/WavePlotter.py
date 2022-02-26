import matplotlib.pyplot as plt
import numpy as np

def pianoWave(freq,X):
        # Y = np.sin(2 * np.pi * freq * X) * np.exp(-0.0004 * 2 * np.pi * freq * X)
        # Y += np.sin(2 * 2 * np.pi * freq * X) * np.exp(-0.0004 * 2 * np.pi * freq * X) / 2
        # Y += np.sin(3 * 2 * np.pi * freq * X) * np.exp(-0.0004 * 2 * np.pi * freq * X) / 4
        # Y += np.sin(4 * 2 * np.pi * freq * X) * np.exp(-0.0004 * 2 * np.pi * freq * X) / 8
        # Y += np.sin(5 * 2 * np.pi * freq * X) * np.exp(-0.0004 * 2 * np.pi * freq * X) / 16
        # Y += np.sin(6 * 2 * np.pi * freq * X) * np.exp(-0.0004 * 2 * np.pi * freq * X) / 32
        # Y += Y * Y * Y
        # Y *= 1 + 16 * X * np.exp(-6 * X)
        # Y = 0.6*np.sin(1*freq*X)* np.exp(-0.0015 * freq * X)
        # Y += 0.4*np.sin(2*freq*X)* np.exp(-0.0015 * freq * X)
        # Y += Y*Y*Y
        # Y *= 1 + 16 * X * np.exp(-6*X)
        Y = 0.6*np.sin(X)* np.exp(-0.0015 * X)
        Y += 0.4*np.sin(2*X)* np.exp(-0.0015 * X)
        Y += Y*Y*Y
        Y *= 1 + 16 * X * np.exp(-6*X)

        return Y

X = np.linspace(0, 2*np.pi,1000)
Y = pianoWave(220,X)


plt.plot(X,Y)
plt.show()