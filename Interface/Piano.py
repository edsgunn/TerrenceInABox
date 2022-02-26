import gensound as gs 
from gensound.signals import Oscillator, Curve
import numpy as np
from gensound.io import IO
IO.set_io("play", "playsound")

class Piano(Oscillator):
    
    def pianoWave(freq,X):

        Y = 0.5*np.sin(X)* np.exp(-0.0015 * X)
        Y += 0.3*np.sin(2*X)* np.exp(-0.0015 * X)
        Y += 0.2*np.sin(3*X)* np.exp(-0.0015 * X)
        Y += Y*Y*Y
        Y *= 1 + 16 * X * np.exp(-6*X/freq)/freq
        return Y

    def generate(self, sample_rate):
        # TODO currently the [:-1] after the integral is needed,
        # otherwise it would be one sample too long. perhaps there is more elegant solution,
        # maybe pasnp.sing an argument telling it to lose the last sample,
        # or better, having CompoundCurve give the extra argument telling its
        # children NOT to lose the last sample
        
        if hasattr(self, "_phase") and self.phase is None: # phase inference
            phase = self._phase
        else:
            phase = self.phase or 0
        
        if isinstance(self.frequency, Curve):
            return type(self).wave(self.frequency, phase + 2*np.pi * self.frequency.integral(sample_rate)[:-1])
        return type(self).wave(self.frequency, phase + 2*np.pi * self.frequency * self.sample_times(sample_rate))

    wave = pianoWave



