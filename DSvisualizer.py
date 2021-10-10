import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from matplotlib.backend_bases import MouseButton

from scipy.integrate import DOP853
from scipy.fft import fft, fftfreq

class DSvisualizer:
    def __init__(self, ode, sampling, pickSample, nTracks, pmapWrap, trajWrap, wrap3d, fftwrap, time, par, labels, poincInd = 0, maxstep=1., devtol=1e-5, lyaptol=1e+1, vectorized=False, equal=True):
        self.ode = ode
        self.sampling = sampling
        self.pickSample = pickSample

        self.nTracks = nTracks
        self.tracks = []
        self.points = {}
        
        self.pmapWrap = pmapWrap
        self.trajWrap = trajWrap
        self.wrap3d = wrap3d
        self.fftwrap = fftwrap
        self.poincInd = poincInd

        self.maxstep = maxstep
        self.devtol = devtol
        self.lyaptol = lyaptol

        self.labels = labels
        self.par = par
        self.vector = vectorized

        keys = list(self.par.keys())
        N = len(keys)
        a = 5.
        b = 0.1
        c = 0.07
        self.L = 2*b+2*c+b+a+5*b+c+(c+b)*(N+1)+b
        self.bottomL = (5*b+c+(c+b)*(N+1)+b)/self.L
        self.topL = (a+5*b+c+(c+b)*(N+1)+b)/self.L

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches((7, self.L))
        self.fig.subplots_adjust(left=0.15, right=0.92, top=self.topL, bottom=self.bottomL)

        axcolor = 'lightgoldenrodyellow'
        self.ax.margins(x=0)
        axtime = plt.axes([0.15, (b+a+5*b+c+(c+b)*(N+1)+b)/self.L, 0.7, c/self.L], facecolor=axcolor)
        self.time_slider = Slider( ax=axtime, label='Time', valmin=time[1], valmax=time[2], valinit=time[0])
        self.time_slider.on_changed(self.__update)

        # sliders
        self.sliders = []
        for i in range(N):
            axtemp = plt.axes([0.15, (2*b+(c+b)*(N-1-i))/self.L, 0.65, c/self.L], facecolor=axcolor)
            self.sliders.append( Slider( ax=axtemp, label=keys[i], valmin=self.par[keys[i]][1][0], valmax=self.par[keys[i]][1][1], valinit=self.par[keys[i]][0]) )
            self.sliders[-1].on_changed(self.__update)

        # buttons
        resetax = plt.axes([0.25, (2*b+N*(c+b))/self.L, 0.12, 2*c/self.L])
        buttonr = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        buttonr.on_clicked(self.__reset)    

        clearax = plt.axes([0.45, (2*b+N*(c+b))/self.L, 0.12, 2*c/self.L])
        buttonc = Button(clearax, 'Clear', color=axcolor, hovercolor='0.975')
        buttonc.on_clicked(self.__clear)  
        
        autoax = plt.axes([0.65, (2*b+N*(c+b))/self.L, 0.12, 2*c/self.L])
        buttona = Button(autoax, 'AutoView', color=axcolor, hovercolor='0.975')
        buttona.on_clicked(self.__auto)    
        
        self.fig.canvas.mpl_connect('button_press_event', self.__onClick)
        self.fig.canvas.mpl_connect('pick_event', self.__onPick)

        # Initial plot
        self.colors = cm.gist_rainbow(np.linspace(0, 1, int(self.nTracks)))
        y0_ar = self.sampling(self.nTracks, self.__slidersVal())
        for i in range(int(self.nTracks)):
            pmap = np.transpose(self.__getPoincare(y0_ar[i]) )
            try: self.tracks.append( self.ax.scatter(pmap[0], pmap[1], s=0.5, color=self.colors[i]) )
            except Exception as exc: self.tracks.append( self.ax.scatter([], [], s=0.5, color=self.colors[i]) )
        self.ax.set_xlabel(self.labels["poincare"][0])
        self.ax.set_ylabel(self.labels["poincare"][1])
        #self.ax.set_aspect("equal")
        print(self.__slidersVal)

        # Trajectory, 3D, FFT, Maximal Lyapunov
        self.fig1 = plt.figure()
        self.fig1.set_size_inches((14, 3))
        self.fig1.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.15, wspace=0.3, hspace=0.1)

        self.ax1 = []
        self.ax1.append( self.fig1.add_subplot(1, 4, 1) )
        self.ax1[0].set_xlabel(self.labels["trajectory"][0])
        self.ax1[0].set_ylabel(self.labels["trajectory"][1])
        if equal: self.ax1[0].set_aspect("equal")

        self.ax1.append( self.fig1.add_subplot(1, 4, 2, projection='3d') )
        self.ax1[1].view_init(30, 60)
        self.ax1[1].set_xlabel(self.labels["3D"][0])
        self.ax1[1].set_ylabel(self.labels["3D"][1])
        self.ax1[1].set_zlabel(self.labels["3D"][2])

        self.ax1.append( self.fig1.add_subplot(1, 4, 3) )
        self.ax1[2].set_xlabel(r'$f$')
        self.ax1[2].set_ylabel("FFT"+self.labels["FFT"])

        self.ax1.append( self.fig1.add_subplot(1, 4, 4) )
        self.ax1[3].set_xlabel(r'$t$')
        self.ax1[3].set_ylabel(r'$\lambda_{max}$')
        plt.show()
    
    def __slidersVal(self):
        values = []
        for s in self.sliders:
            values.append(s.val)
        return values

    def __odeWrap(self, t, y):
        return self.ode(t, y, self.__slidersVal())

    def __getPoincare(self, y0):
        solver = DOP853(self.__odeWrap, 0, y0, self.time_slider.val, max_step=self.maxstep, vectorized=self.vector)
        pmap = []
        prev_state = y0
        while True:
            try:
                solver.step()
            except Exception as e: break
            
            if ((solver.y[self.poincInd] > 0. and prev_state[self.poincInd] < 0.) or (solver.y[self.poincInd] < 0. and prev_state[self.poincInd] > 0.)):
                dx = solver.y[self.poincInd] - prev_state[self.poincInd]
                yp = prev_state + ((solver.y - prev_state)/dx)*(0.-prev_state[self.poincInd])
                pmap.append( self.pmapWrap(yp))
            prev_state = solver.y
        return pmap[1:]

    def __calcLines(self, key):
        y0 = self.pickSample(self.points[key]["xy"], self.__slidersVal())
        dev = np.random.uniform(0., self.devtol, size=y0.shape[0])    
        eta = np.linalg.norm(dev)
        
        #calculating
        log_k = 0.
        y_dev = y0+dev
        t = np.linspace(0., self.time_slider.val, int(self.time_slider.val/(self.maxstep*0.1)))
        sol, lyap = [y0], []
        for i in range(len(t)-1):
            solver_ref = DOP853(self.__odeWrap, t[i], sol[-1], t[i+1], max_step=self.maxstep*0.1, vectorized=self.vector)
            while True:
                try: solver_ref.step()
                except Exception as e: break
            sol.append(solver_ref.y)

            solver_dev = DOP853(self.__odeWrap, t[i], y_dev, t[i+1], max_step=self.maxstep, vectorized=self.vector)
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
        return t, self.trajWrap(sol, t), self.wrap3d(sol), self.fftwrap(sol), lyap

    def __plotLines(self, key):
        t, traj, traj3d, fftsol, lyap = self.__calcLines(key)
        #plotting
        self.points[key]["lines"].append( self.ax1[0].plot( traj[0], traj[1], linewidth=0.2 ) )
        self.points[key]["lines"].append( self.ax1[1].plot( traj3d[0], traj3d[1], traj3d[2], linewidth=0.2 ) )
        N = t.shape[0]
        yf = fft( fftsol )
        xf = fftfreq(N, t[1]-t[0])[:N//2]
        self.points[key]["lines"].append( self.ax1[2].loglog( xf, 2./N * np.abs(yf[0:N//2]), linewidth=0.5 ) )       
        self.points[key]["lines"].append( self.ax1[3].plot( t[1:], lyap ) )
        self.__updateFig1()
    
    def __updateLines(self, val):
        for key in list( self.points.keys() ):
            if self.pickSample(self.points[key]["xy"], self.__slidersVal()) is not None:
                t, traj, traj3d, fftsol, lyap = self.__calcLines(key)
                #re-plotting
                self.points[key]["lines"][0][0].set_xdata( traj[0] )
                self.points[key]["lines"][0][0].set_ydata( traj[1] )

                self.points[key]["lines"][1][0].set_xdata( traj3d[0] )
                self.points[key]["lines"][1][0].set_ydata( traj3d[1] )
                self.points[key]["lines"][1][0].set_3d_properties( traj3d[2] )

                N = t.shape[0]
                yf = fft( fftsol )
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

    @staticmethod        
    def __calcLimits(adata):
        adata = np.concatenate(adata, axis=None)
        a_lim = (np.amin(adata), np.amax(adata))
        return ( a_lim[0]-np.abs(a_lim[0])*0.1, a_lim[1]+np.abs(a_lim[1])*0.1 ) 

    def __updateFig1(self):
        self.fig1.canvas.draw_idle()
        for ax in self.ax1:
            ax.relim()
            ax.autoscale_view()

    def __update(self, val):
        y0_ar = self.sampling(self.nTracks, self.__slidersVal())
        for i in range(int(self.nTracks)):
            pmap = self.__getPoincare(y0_ar[i])
            if len(pmap) >= 1: self.tracks[i].set_offsets( pmap )
            else: 
                self.tracks[i].remove()
                self.tracks[i] = self.ax.scatter([], [], s=0.5, color=self.colors[i])
        self.fig.canvas.draw_idle()
        print(self.__slidersVal())
        self.__updateLines(val)

    def __reset(self, event):
        self.time_slider.reset()
        for s in self.sliders:
            s.reset()

    def __clear(self, event):
        for key in list(self.points.keys()):
            self.__removeLines(key)
        self.__updateFig1()

    def __auto(self, event):
        offx, offy = [], []
        for i in range(self.nTracks):
            off = np.array(self.tracks[i].get_offsets().data)
            if len(off) > 0:
                offx.append( off[:,0] )
                offy.append( off[:,1] )
        for key in self.points.keys():
            off = np.array(key.get_offsets().data)
            offx.append( off[:,0] )
            offy.append( off[:,1] )
        self.ax.set_xlim( self.__calcLimits(offx) ) 
        self.ax.set_ylim( self.__calcLimits(offy) )
        # self.ax.relim()
        # self.ax.autoscale()
    
    def __onClick(self, event):
        dpi = self.fig.get_dpi()
        if event.x > 0.15*7*dpi and event.x < 0.92*7*dpi and event.y > self.bottomL*self.L*dpi and event.y < self.topL*self.L*dpi and event.button is MouseButton.LEFT:
            if self.pickSample((event.xdata, event.ydata), self.__slidersVal()) is not None:
                key = self.ax.scatter(event.xdata, event.ydata, marker="X", edgecolors="k", s=50, picker=True, pickradius=1)
                print(event.x, event.y, event.xdata, event.ydata, event.button)
                self.fig.canvas.draw_idle()
                self.points[key] = {"xy":(event.xdata, event.ydata),"lines":[]}
                self.__plotLines(key)
    
    def __onPick(self, event):
        if event.mouseevent.button is MouseButton.RIGHT:
            print(event.artist, event.mouseevent.button)
            self.__removeLines(event.artist)