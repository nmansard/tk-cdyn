### ROBOTS 

import pinocchio as pin
import hppfcl
import example_robot_data as robex
import numpy as np

from util_load_robots import defRootName,addXYZAxisToJoints,replaceGeomByXYZAxis,freeze,renameConstraints,classic_cassie_unecessary_constraints,classic_cassie_blocker,addXYZAxisToConstraints,fixCassieConstraints,cassie_spring_knee_joints

cassie=robex.load('cassie')
fixCassieConstraints(cassie)
defRootName(cassie.model,'cassie')
renameConstraints(cassie)
addXYZAxisToConstraints(cassie.model,cassie.visual_model,cassie.constraint_models)
cassie.constraint_models = [ cm for cm in cassie.constraint_models
                             if cm.name not in classic_cassie_unecessary_constraints ]
freeze(cassie,'left','standing',rebuildData=False)
for k in classic_cassie_blocker+cassie_spring_knee_joints:
    print(f'Freeze {k}')
    assert(len([ n for n in cassie.model.names if k in n])>0)
    freeze(cassie,k,'standing',rebuildData=False)

cassie.full_constraint_models = cassie.constraint_models
cassie.full_constraint_datas = { cm: cm.createData() for cm in cassie.constraint_models }
#cassie.constraint_models = [ cm for cm in cassie.constraint_models
#                             if cm.name=='right-achilles-spring-joint,right-tarsus-spring-joint']
addXYZAxisToJoints(cassie.model,cassie.visual_model)
cassie.rebuildData()
cassie.initViewer(loadModel=True)
replaceGeomByXYZAxis(cassie.visual_model,cassie.viz)
cassie.display(cassie.q0)
cassie.constraint_datas = [ cassie.full_constraint_datas[cm] for cm in cassie.constraint_models ]

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
        if len(cassie.constraint_models)==0:
            return self.qref

        
        constraint = casadi.Function('constraint', [cq],[ constraintsResidual(cassie.casmodel,cassie.casdata,
                                                                              cassie.constraint_models,cassie.constraint_datas,cq,
                                                                              False,caspin) ])
        
        self.opti = opti = casadi.Opti()

        # Decision variables
        self.vdq = vdq = opti.variable(cassie.model.nv)
        self.vq = vq = integrate(cassie.q0,vdq)

        # Cost and constraints
        totalcost = 0 
        totalcost = casadi.sumsqr(vq-self.qref)
        opti.subject_to( constraint(vq) == 0 )

        #totalcost += casadi.sumsqr(constraint(vq))

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
        
#'''
root = tk.Tk()
root.bind('<Escape>', lambda ev: root.destroy())
root.title("Cassie")
cassieFrame = RobotConstraintFrame(cassie.model,cassie.q0,cassie,
                                   motors = [ n for n in cassie.model.names if 'op' in n ])
cassieFrame.createSlider(root)
cassieFrame.createRefreshButons(root)

optimFrame = tk.Frame(root)
optimFrame.pack(side=tk.BOTTOM)
reset_button = tk.Button(optimFrame, text="Reset", command=resetAndDisp)
reset_button.pack(side=tk.LEFT, padx=10, pady=10)
optim_button = tk.Button(optimFrame, text="Optim", command=computeConstrainedConfig)
optim_button.pack(side=tk.LEFT, padx=10, pady=10)

constraintWindow = tk.Toplevel()
constraintWindow.bind('<Escape>', lambda ev: root.destroy())

class CheckboxConstraintCmd:
    def __init__(self,bvar,cm):
        self.bvar = bvar
        self.cm = cm
    def __call__(self):
        if self.bvar.get():
            print(f'Activate {self.cm.name}' )
            assert(self.cm not in cassie.constraint_models)
            cassie.constraint_models.append(self.cm)
            cassie.constraint_datas = [ cassie.full_constraint_datas[cm] for cm in cassie.constraint_models ]
        else:
            print(f'Deactivate {self.cm.name}' )
            assert(self.cm in cassie.constraint_models)
            cassie.constraint_models.remove(self.cm)
        cassie.constraint_datas = [ cassie.full_constraint_datas[cm] for cm in cassie.constraint_models ]

class CheckboxDisplayConstraintCmd:
    def __init__(self,bvar,cm,vm,viz):
        self.bvar = bvar
        self.cm = cm
        self.viz = viz
        # Get viewer object names with pinocchio convention
        idxs = [ vm.getGeometryId(f'XYZ_cst_{cm.name}_1'),
                 vm.getGeometryId(f'XYZ_cst_{cm.name}_2') ]
        self.gname = [ viz.getViewerNodeName(vm.geometryObjects[idx],pin.VISUAL)
                       for idx in idxs ]
    def __call__(self):
        print(f'Set display {self.cm.name} to {self.bvar.get()}' )
        for n in self.gname:
            self.viz.viewer.gui.setVisibility(n,'ON' if self.bvar.get() else 'OFF')
            
constraintFrame = tk.Frame(constraintWindow)
constraintFrame.pack(side=tk.BOTTOM)
actLabel = tk.Label(constraintFrame,text='active')
actLabel.grid(row=0,column=1)
dispLabel = tk.Label(constraintFrame,text='display')
dispLabel.grid(row=0,column=2)

for i,cm in enumerate(cassie.full_constraint_models):
    cstLabel = tk.Label(constraintFrame,text=cm.name)
    cstLabel.grid(row=i+1,column=0)
    
    active_constraint_var = tk.BooleanVar(value=cm in cassie.constraint_models)
    constraint_checkbox = tk.Checkbutton(constraintFrame,variable=active_constraint_var,
                                         command=CheckboxConstraintCmd(active_constraint_var,cm))
    #constraint_checkbox.pack(side=tk.LEFT, padx=1, pady=10)
    constraint_checkbox.grid(row=i+1, column=1)

    display_constraint_var = tk.BooleanVar(value=cm in cassie.constraint_models)
    display_constraint_cmd = CheckboxDisplayConstraintCmd(display_constraint_var,cm,cassie.visual_model,cassie.viz)
    display_constraint_cmd()
    display_constraint_checkbox = tk.Checkbutton(constraintFrame,variable=display_constraint_var,
                                                 command=display_constraint_cmd)
    display_constraint_checkbox.grid(row=i+1, column=2)


root.mainloop()
#'''

model = cassie.model
data = cassie.data
cm = cassie.constraint_models[0]
cd = pin.RigidConstraintData(cm)
q = cassie.q0

pin.forwardKinematics(model, data, q)
pin.computeAllTerms(model, data, q, np.zeros(model.nv))
J = pin.getConstraintJacobian(model,data,cm,cd)
pin.SE3.__repr__=pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True,threshold=1e6)
