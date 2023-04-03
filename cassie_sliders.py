### ROBOTS 

import pinocchio as pin
import hppfcl
import example_robot_data as robex
import numpy as np
from util_load_robots import defRootName,addXYZAxisToJoints,replaceGeomByXYZAxis,freeze,renameConstraints

cassie=robex.load('cassie')
defRootName(cassie.model,'cassie')
freeze(cassie,'left','standing',rebuildData=False)
addXYZAxisToJoints(cassie.model,cassie.visual_model)
cassie.rebuildData()
cassie.initViewer(loadModel=True)
replaceGeomByXYZAxis(cassie.visual_model,cassie.viz)
cassie.display(cassie.q0)


### WINDOWS

from tk_configuration import RobotFrame
import tkinter as tk

root = tk.Tk()
root.bind('<Escape>', lambda ev: root.destroy())
root.title("Cassie")
cassieFrame = RobotFrame(cassie.model,cassie.q0,cassie)
cassieFrame.createSlider(root)
cassieFrame.createRefreshButons(root)

# For fun ...
def resetAndDisp():
    cassieFrame.resetConfiguration()
    cassieFrame.display()
reset_button = tk.Button(root, text="Reset", command=resetAndDisp)
reset_button.pack(side=tk.LEFT, padx=10, pady=10)

q0 = cassie.q0
q0[0] += 1
cassieFrame.resetConfiguration(q0)

root.mainloop()
