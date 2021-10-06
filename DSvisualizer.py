import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from matplotlib.backend_bases import MouseButton

from scipy.integrate import DOP853
from scipy.fft import fft, fftfreq

class DSvisualizer:
    def __init__(self, ode, sampling, calcEnergy, pickSample, nTracks, e, t, erange, trange=(400,4000), poincInd=(1,2), trajInd=(0,1), maxstep=1., devtol=1e-5, lyaptol=1e-1):
        self.ode = ode
        self.sampling = sampling
        self.calcEnergy = calcEnergy
        self.pickSample = pickSample
        self.nTracks = nTracks
        self.tracks = []
        self.points = {}

        self.poincInd = poincInd
        self.trajInd = trajInd

        self.maxstep = maxstep
        self.devtol = devtol
        self.lyaptol = lyaptol

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches((7, 6))
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)

        colors = cm.gist_rainbow(np.linspace(0, 1, int(self.nTracks)))
        y0_ar = self.sampling(e, self.nTracks)
        for i in range(int(self.nTracks)):
            pmap = np.transpose(self.__getPoincare(y0_ar[i], t))
            self.tracks.append( self.ax.scatter(pmap[0], pmap[1], s=0.3, color=colors[i]) )
        self.ax.set_xlabel(r'$y$')
        self.ax.set_ylabel(r'$dy/dt$')
        #self.ax.set_aspect("equal")
        print("Energy: "+str(e)+", Time: "+str(t))

        axcolor = 'lightgoldenrodyellow'
        self.ax.margins(x=0)

        axtime = plt.axes([0.15, 0.05, 0.45, 0.02], facecolor=axcolor)
        self.time_slider = Slider( ax=axtime, label='Time', valmin=trange[0], valmax=trange[1], valinit=t)
        self.time_slider.on_changed(self.__update)

        axenergy = plt.axes([0.15, 0.95, 0.7, 0.02], facecolor=axcolor)
        self.energy_slider = Slider( ax=axenergy, label='Energy', valmin=erange[0], valmax=erange[1], valinit=e,)
        self.energy_slider.on_changed(self.__update)

        resetax = plt.axes([0.68, 0.045, 0.12, 0.04])
        buttonr = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        buttonr.on_clicked(self.__reset)    

        clearax = plt.axes([0.83, 0.045, 0.12, 0.04])
        buttonc = Button(clearax, 'Clear', color=axcolor, hovercolor='0.975')
        buttonc.on_clicked(self.__clear)    
        
        self.fig.canvas.mpl_connect('button_press_event', self.__onClick)
        self.fig.canvas.mpl_connect('pick_event', self.__onPick)

        # Trajectory, Energy, FFT, Maximal Lyapunov
        self.fig1, self.ax1 = plt.subplots(1,4)
        self.fig1.set_size_inches((14, 3))
        self.fig1.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.15, wspace=0.3, hspace=0.1)

        self.ax1[0].set_xlabel(r'$x$')
        self.ax1[0].set_ylabel(r'$y$')
        self.ax1[0].set_aspect("equal")

        self.ax1[1].set_xlabel(r'$t$')
        self.ax1[1].set_ylabel(r'$E$')

        self.ax1[2].set_xlabel(r'$f$')
        self.ax1[2].set_ylabel("FFT"+r'$(x^2+y^2)$')

        self.ax1[3].set_xlabel(r'$t$')
        self.ax1[3].set_ylabel(r'$\lambda_{max}$')
        plt.show()

    def __getPoincare(self, y0, time):
        solver = DOP853(self.ode, 0, y0, time, max_step=self.maxstep)
        pmap = []
        prev_state = y0
        while True:
            try:
                solver.step()
            except Exception as e: break
            
            if ((solver.y[0] > 0. and prev_state[0] < 0.) or (solver.y[0] < 0. and prev_state[0] > 0.)):
                dx = solver.y[0] - prev_state[0]
                yp = prev_state + ((solver.y - prev_state)/dx)*(0.-prev_state[0])
                pmap.append((yp[self.poincInd[0]], yp[self.poincInd[1]]))
            prev_state = solver.y
        return pmap

    def __calcLines(self, key):
        y0 = self.pickSample(self.energy_slider.val, self.points[key]["xy"])
        dev = np.random.uniform(0., self.devtol, size=y0.shape[0])    
        eta = np.linalg.norm(dev)
        
        #calculating
        log_k = 0.
        y_dev = y0+dev
        t = np.linspace(0., self.time_slider.val, int(self.time_slider.val/(self.maxstep*0.1)))
        ene, sol, lyap = [self.calcEnergy(y0)], [y0], []
        for i in range(len(t)-1):
            solver_ref = DOP853(self.ode, t[i], sol[-1], t[i+1], max_step=self.maxstep*0.1)
            while True:
                try: solver_ref.step()
                except Exception as e: break
            sol.append(solver_ref.y)
            ene.append(self.calcEnergy(sol[-1]))

            solver_dev = DOP853(self.ode, t[i], y_dev, t[i+1], max_step=self.maxstep)
            while True:
                try: solver_dev.step()
                except Exception as e: break
            y_dev = solver_dev.y
            
            delta = np.linalg.norm(sol[-1]-y_dev)
            if delta > self.lyaptol:
                k = eta/delta
                # reskaliranje
                y_dev = sol[-1] + k * (y_dev - sol[-1])
                log_k -= np.log(k)
                delta = eta
            lyap.append( (np.log(delta/eta) + log_k)/t[i+1] )
        lyap = np.array(lyap)
        sol = np.transpose(sol)
        return t, ene, sol, lyap

    def __plotLines(self, key):
        t, ene, sol, lyap = self.__calcLines(key)
        #plotting
        self.points[key]["lines"].append( self.ax1[0].plot( sol[self.trajInd[0]], sol[self.trajInd[1]], linewidth=0.2 ) )
        self.points[key]["lines"].append( self.ax1[1].plot( t, ene ) )
        N = t.shape[0]
        yf = fft( np.power(sol[self.trajInd[0]], 2) + np.power(sol[self.trajInd[1]], 2) )
        xf = fftfreq(N, t[1]-t[0])[:N//2]
        self.points[key]["lines"].append( self.ax1[2].loglog( xf, 2./N * np.abs(yf[0:N//2]), linewidth=0.5 ) )       
        self.points[key]["lines"].append( self.ax1[3].plot( t[1:], lyap ) )
        self.__updateFig1()
    
    def __updateLines(self, val):
        for key in list( self.points.keys() ):
            if self.pickSample(self.energy_slider.val, self.points[key]["xy"]) is not None:
                t, ene, sol, lyap = self.__calcLines(key)
                #re-plotting
                self.points[key]["lines"][0][0].set_xdata( sol[self.trajInd[0]] )
                self.points[key]["lines"][0][0].set_ydata( sol[self.trajInd[1]] )

                self.points[key]["lines"][1][0].set_xdata( t )
                self.points[key]["lines"][1][0].set_ydata( ene )

                N = t.shape[0]
                yf = fft( np.power(sol[self.trajInd[0]], 2) + np.power(sol[self.trajInd[1]], 2) )
                xf = fftfreq(N, t[1]-t[0])[:N//2]
                self.points[key]["lines"][3][0].set_xdata( xf )
                self.points[key]["lines"][3][0].set_ydata( 2./N * np.abs(yf[0:N//2]) )    
                
                self.points[key]["lines"][3][0].set_xdata( t[1:] )
                self.points[key]["lines"][3][0].set_ydata( lyap )    
            else: self.__removeLines(key)
        self.__updateFig1()
    
    def __removeLines(self, key):
        for i in range(len(self.points[key]["lines"])):
            self.ax1[i].lines.remove(self.points[key]["lines"][i][0])
        self.points.pop(key)
        key.remove()
        self.fig.canvas.draw_idle()
        self.__updateFig1()
        
    def __updateFig1(self):
        self.fig1.canvas.draw_idle()
        for ax in self.ax1:
            ax.relim()
            ax.autoscale_view()

    def __update(self, val):
        y0_ar = self.sampling(self.energy_slider.val, self.nTracks)
        for i in range(int(self.nTracks)):
            pmap = self.__getPoincare(y0_ar[i], self.time_slider.val)
            self.tracks[i].set_offsets( pmap )
        self.fig.canvas.draw_idle()
        print("Energy: "+str(self.energy_slider.val)+", Time: "+str(self.time_slider.val))
        self.__updateLines(val)

    def __reset(self, event):
        self.energy_slider.reset()
        self.time_slider.reset()

    def __clear(self, event):
        for key in list(self.points.keys()):
            self.__removeLines(key)
        self.__updateFig1()
    
    def __onClick(self, event):
        if event.x > 0.15*700 and event.x < 0.95*700 and event.y > 0.15*600 and event.y < 0.93*600 and event.button is MouseButton.LEFT:
            if self.pickSample(self.energy_slider.val, (event.xdata, event.ydata)) is not None:
                key = self.ax.scatter(event.xdata, event.ydata, marker="X", edgecolors="k", s=50, picker=True, pickradius=1)
                print(event.x, event.y, event.xdata, event.ydata, event.button)
                self.fig.canvas.draw_idle()
                self.points[key] = {"xy":(event.xdata, event.ydata),"lines":[]}
                self.__plotLines(key)
    
    def __onPick(self, event):
        if event.mouseevent.button is MouseButton.RIGHT:
            print(event.artist, event.mouseevent.button)
            self.__removeLines(event.artist)