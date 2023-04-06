import pinocchio as pin
import numpy as np
from pinocchio import casadi as caspin
import casadi

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
