import tkinter as tk
import numpy as np
import pinocchio as pin
from functools import partial

class RobotFrame:
    '''
    Create a tk.Frame and add sliders corresponding to robot joints.
    Return the tk.Frame, that must be added to the container.
    '''
    NROW = 8 # Number of sliders per row
    
    def __init__(self,robotModel,q0,viz):
        self.rmodel = robotModel
        self.viz = viz
        self.auto_refresh = True
        self.q0 = q0.copy()

    def resetConfiguration(self,qref=None):
        if qref is not None:
            dq_ref = pin.difference(self.rmodel,self.q0,qref)
        for i,s in enumerate(self.slider_vars):
            s.set(0 if qref is None else dq_ref[i])
    def getConfiguration(self,verbose=True):
        values = [var.get() for var in self.slider_vars]
        dq = np.array(values)
        print(dq)
        q = pin.integrate(self.rmodel,self.q0,dq)
        return q
        
    # Fonction pour afficher les valeurs des sliders
    def slider_display(self,i,v):
        if self.auto_refresh:
            self.display()
    def display(self):
        q= self.getConfiguration()
        self.viz.display(q)

    def createSlider(self,tkParent,pack=True):
        # Frame pour les sliders
        frame = self.slidersFrame = tk.Frame(tkParent)

        # Création des sliders verticaux
        self.slider_vars = []
        iq = 0
        for j,name in enumerate(self.rmodel.names):
            if j==0: continue
            for iv in range(self.rmodel.joints[j].nv):
                var = tk.DoubleVar(value=0)
                self.slider_vars.append(var)
                slider_frame = tk.Frame(self.slidersFrame)
                row  =  iq // self.NROW
                slider_frame.grid(row=row*2, column=iq-self.NROW*row, padx=5, pady=5)
                name_i = name if self.rmodel.joints[j].nv==1 else name+f'{iv}'
                slider_label = tk.Label(slider_frame, text=name_i)
                slider_label.pack(side=tk.BOTTOM)
                slider = tk.Scale(slider_frame, variable=var, orient=tk.VERTICAL, from_=-3.0, to=3.0,
                                  resolution=0.01,command=partial(self.slider_display,iq))
                slider.pack(side=tk.BOTTOM)
                iq += 1
            class VisibilityChanger:
                def __init__(self,viz,name,var):
                    self.gv = viz.viewer.gui
                    self.name = name
                    self.var = var
                    self()
                def __call__(self):
                    gname = f'world/pinocchio/visuals/XYZ_{self.name}'
                    self.gv.setVisibility(gname,'ON'  if self.var.get() else 'OFF')
                    print(gname,'ON'  if self.var.get() else 'OFF',self.var,self.var.get())
            XYZon = tk.BooleanVar(value=False)
            tk.Checkbutton(slider_frame, text="",variable=XYZon,
                           command=VisibilityChanger(self.viz,name,XYZon)).pack(side=tk.RIGHT)

        if pack: frame.pack(side=tk.TOP)
        return frame

    def setRefresh(self,v=None):
        print(' set ' ,v)
        if v is None:
            self.auto_refresh = not self.auto_refresh
        else:
            self.auto_refresh = v
    
    def createRefreshButons(self,tkParent,pack=True):
        # Frame pour le bouton d'affichage et la checkbox
        self.buttonsFrame = tk.Frame(tkParent)
        if pack: self.buttonsFrame.pack(side=tk.BOTTOM)

        # Bouton pour afficher les valeurs des sliders manuellement
        manual_button = tk.Button(self.buttonsFrame, text="Display", command=self.display)
        manual_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Checkbox pour activer/désactiver l'auto-refresh
        self.auto_refresh_var = tk.BooleanVar(value=self.auto_refresh)
        auto_refresh_checkbox = tk.Checkbutton(self.buttonsFrame, text="Auto"+self.rmodel.name,
                                               variable=self.auto_refresh_var, command=self.checkboxCmd)
        auto_refresh_checkbox.pack(side=tk.LEFT, padx=10, pady=10)

    def checkboxCmd(self):
        print(' autt' )
        self.setRefresh(self.auto_refresh_var.get())

