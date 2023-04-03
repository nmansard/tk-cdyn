### ROBOTS 

import pinocchio as pin
import hppfcl
import example_robot_data as robex
import numpy as np

from util_load_robots import defRootName,addXYZAxisToJoints,replaceGeomByXYZAxis,freeze,renameConstraints

cassie=robex.load('cassie')
defRootName(cassie.model,'cassie')
freeze(cassie,'left','standing',rebuildData=False)
freeze(cassie,'-ip','standing',rebuildData=False)
cassie.constraint_models = cassie.constraint_models[3:4]
addXYZAxisToJoints(cassie.model,cassie.visual_model)
for cm in cassie.constraint_models:
    cm.type = pin.ContactType.CONTACT_3D
cassie.rebuildData()
cassie.initViewer(loadModel=True)
replaceGeomByXYZAxis(cassie.visual_model,cassie.viz)
cassie.display(cassie.q0)
renameConstraints(cassie)
cassie.constraint_datas = [ cm.createData() for cm in cassie.constraint_models]

### CONSTRAINTS

def constraintResidual6d(model,data,cmodel,cdata,q=None,recompute=True,pinspace=pin):
    assert(cmodel.type==pin.ContactType.CONTACT_6D)
    if recompute:
        pinspace.forwardKinematics(model,data,q)
    oMc1 = data.oMi[cmodel.joint1_id]*pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id]*pinspace.SE3(cmodel.joint2_placement)
    return pinspace.log6(oMc1.inverse()*oMc2).vector

def constraintResidual3d(model,data,cmodel,cdata,q=None,recompute=True,pinspace=pin):
    assert(cmodel.type==pin.ContactType.CONTACT_3D)
    if recompute:
        pinspace.forwardKinematics(model,data,q)
    oMc1 = data.oMi[cmodel.joint1_id]*pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id]*pinspace.SE3(cmodel.joint2_placement)
    return oMc1.translation-oMc2.translation

def constraintResidual(model,data,cmodel,cdata,q=None,recompute=True,pinspace=pin):
    if cmodel.type==pin.ContactType.CONTACT_6D:
        return constraintResidual6d(model,data,cmodel,cdata,q,recompute,pinspace)
    elif cmodel.type==pin.ContactType.CONTACT_3D:
        return constraintResidual3d(model,data,cmodel,cdata,q,recompute,pinspace)
    else:
        assert(False and "Should never happen")

def constraintsResidual(model,data,cmodels,cdatas,q=None,recompute=True,pinspace=pin):
    res = []
    for cm,cd in zip(cmodels,cdatas):
        res.append(constraintResidual(model,data,cm,cd,q,recompute,pinspace))
    if pinspace is pin:
        return np.concatenate(res)
    elif pinspace is caspin:
        return casadi.vertcat(*res)
    else:
        assert(False and "Should never happen")

### CASADI
from pinocchio import casadi as caspin
import casadi

cassie.casmodel = caspin.Model(cassie.model)
cassie.casdata = cassie.casmodel.createData()

cq = casadi.SX.sym("q",cassie.casmodel.nq,1)
cv = casadi.SX.sym("v",cassie.casmodel.nv,1)
caspin.forwardKinematics(cassie.casmodel,cassie.casdata,cq)

integrate = casadi.Function('integrate', [cq,cv],[ caspin.integrate(cassie.casmodel,cq,cv) ])
constraint = casadi.Function('constraint', [cq],[ constraintsResidual(cassie.casmodel,cassie.casdata,
                                                                      cassie.constraint_models,cassie.constraint_datas,cq,
                                                                      False,caspin) ])


### DG
cm = cassie.constraint_models[0]
cd = cm.createData()

## PROBLEM
class ProjectConfig:
    def __init__(self,qref = None,iv=None):
        if qref is None:
            qref =np.array([-0.527, -0.   ,  0.939,  0.   ,  0.001,  0.   ,  1.   , -0.013,  0.013,  0.   ,
                            0.017,  -0.017, -0.   ,  1.591, -1.397,  0.195, -0.006, -0.005, -0.084,  0.996,
                            -0.   ,  1.403, -1.63 , -0.227, -0.   , -0.   ,  0.693, -0.633, -1.754,  1.757,
                            -0.   , -0.   , -0.763,  0.646,  1.4   , -3.133, -1.734, -0.   ])
        self.qref = qref.copy()
        self.iv = iv
    def __call__(self):
        '''
        Project an input configuration <qref> to the nearby feasible configuration
        If <iv> is not null, then the DOF #iv is set as hard constraints, while the other are moved.
        '''
        self.opti = opti = casadi.Opti()

        # Decision variables
        self.vdq = vdq = opti.variable(cassie.model.nv)
        self.vq = vq = integrate(cassie.q0,vdq)

        # Cost and constraints
        totalcost = 0 #casadi.sumsqr(vq-self.qref)/100
        #opti.subject_to( constraint(vq) == 0 )
        totalcost += casadi.sumsqr(constraint(vq))

        # Solve
        opti.solver("ipopt") # set numerical backend
        if self.iv is not None:
            dvref = pin.difference(cassie.model,cassie.q0,self.qref)
            opti.subject_to( vdq[self.iv]==dvref[self.iv] )
        opti.minimize(totalcost)
        try:
            sol = opti.solve_limited()
            self.qopt = qopt =  opti.value(vq)
            self.dqopt = opti.value(vdq)
        except:
            print('ERROR in convergence, plotting debug info.')
            self.qopt = qopt = opti.debug.value(vq)
            self.dqopt = opti.debug.value(vdq)
        return qopt


def computeConstrainedConfig():
    qref = cassieFrame.getConfiguration(False)
    q = ProjectConfig(qref)()
    cassieFrame.resetConfiguration(q)
    cassieFrame.display()

def resetAndDisp():
    cassieFrame.resetConfiguration(cassie.q0)
    cassieFrame.display()

### WINDOWS
from tk_configuration import RobotFrame
import tkinter as tk

class RobotConstraintFrame(RobotFrame):
    # def __init__(self, *args, **kwargs):
    #     super(RobotFrame, self).__init__(*args, **kwargs)
    def slider_display(self,i,v):
        print('NEW SLIDE ... ',i,v)
        qref = cassieFrame.getConfiguration(False)
        q = ProjectConfig(qref,i)()
        cassieFrame.resetConfiguration(q)
        cassieFrame.display()
        
root = tk.Tk()
root.bind('<Escape>', lambda ev: root.destroy())
root.title("Cassie")
cassieFrame = RobotConstraintFrame(cassie.model,cassie.q0,cassie)
cassieFrame.createSlider(root)
cassieFrame.createRefreshButons(root)

reset_button = tk.Button(root, text="Reset", command=resetAndDisp)
reset_button.pack(side=tk.LEFT, padx=10, pady=10)
optim_button = tk.Button(root, text="Optim", command=computeConstrainedConfig)
optim_button.pack(side=tk.LEFT, padx=10, pady=10)

root.mainloop()
