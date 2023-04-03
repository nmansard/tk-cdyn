import pinocchio as pin
import hppfcl
import example_robot_data as robex
import numpy as np

def defRootName(model,name):
    '''
    Change the name of the root joint (to avoid ambiguities when merging two free-flyer robots
    '''
    assert(model.names[1] == 'root_joint')
    model.names[1] = f'root_{name}'
    model.frames[1].name = f'root_{name}'

def addXYZAxisToJoints(rm,vm,basename='XYZ'):
    '''
    Add a sphere object to each joint in the visual model.
    rm: robot model
    vm: visual model
    basename: the prefix of the new geometry objects (suffix are the joint names).
    '''
    for i,name in enumerate(rm.names):
        if i==0:continue
        vm.addGeometryObject(pin.GeometryObject(f'{basename}_{name}',i,pin.SE3.Identity(),hppfcl.Sphere(.001))) 

def replaceGeomByXYZAxis(vm,viz,prefix='XYZ_'):
    '''
    XYZaxis visuals cannot be set from URDF in Gepetto viewer. This function is used
    to replace some geometry objects with proper prefix by XYZAxis visuals.
    rm: robot model
    vm: visual model
    viz: a gepetto-viewer client (typically found in robot.viewer)
    prefix: the prefix of the geometry objects to replace
    '''
    gv = viz.viewer.gui
    for g in vm.geometryObjects:
        if g.name[:len(prefix)] == prefix:
            gname = viz.getViewerNodeName(g,pin.VISUAL)
            gv.deleteNode(gname,True)
            gv.addXYZaxis(gname,[1.,1,1.,1.],.01,.2) 

def freeze(robot,key,referenceConfigurationName=None,rebuildData=True):
    '''
    Reduce the model by freezing all joint whose name contain the key string.
    robot: a robot wrapper where the result is stored (destructive mode)
    key: the string to search in the joint names to lock.
    '''
    idx = [ i for i,n in enumerate(robot.model.names) if key in n] # to lock
    rmbak = robot.model
    robot.model,(robot.visual_model,robot.collision_model) = \
        pin.buildReducedModel(robot.model,[robot.visual_model,robot.collision_model],idx,robot.q0)
    if referenceConfigurationName is None:
        del robot.q0
    else:
        robot.q0 = robot.model.referenceConfigurations[referenceConfigurationName]
    if rebuildData:
        robot.rebuildData()
    if hasattr(robot,'constraint_models'):
        toremove = []
        for cm in robot.constraint_models:
            # Convert previous indexes to new joint list (after some joints are frozen)
            n1 = rmbak.names[cm.joint1_id]
            n2 = rmbak.names[cm.joint2_id]
            cm.joint1_id = robot.model.getJointId(n1)
            cm.joint2_id = robot.model.getJointId(n2)
            # If some constraints are now useless, remove them
            # Todo: this might be overrestrictive
            if cm.joint2_id==robot.model.njoints or cm.joint2_id==robot.model.njoints:
                f1 = robot.model.frames[robot.model.getFrameId(n1)]
                f2 = robot.model.frames[robot.model.getFrameId(n2)]
                # Simple assert to raise an error when the TODO will become necessary
                # ... we may have frozen the joints, but they are attached to two
                # ... moving frame. In that case, rebuild the constraint with different
                # ... joint_placement
                assert(f1.parentJoint == f2.parentJoint)
                toremove.append(cm)
                print(f'Remove constraint {n1}//{n2}')
        robot.constraint_models = [ cm for cm in robot.constraint_models if cm not in toremove ]
                
                
def renameConstraints(robot):
    for cm in robot.constraint_models:
        cm.name = f'{robot.model.names[cm.joint1_id]},{robot.model.names[cm.joint2_id]}'
