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

def addXYZAxisToConstraints(rm,vm,cms,basename='XYZ_cst'):
    '''
    Add a sphere object to each joint in the visual model.
    rm: robot model
    vm: visual model
    cms: constraint models (in a list)
    basename: the prefix of the new geometry objects (suffix are the joint names).
    '''
    for cm in cms:
        i = cm.joint1_id
        vm.addGeometryObject(pin.GeometryObject(f'{basename}_{cm.name}_1',i,cm.joint1_placement,hppfcl.Sphere(.001))) 
        i = cm.joint2_id
        vm.addGeometryObject(pin.GeometryObject(f'{basename}_{cm.name}_2',i,cm.joint2_placement,hppfcl.Sphere(.001))) 

def replaceGeomByXYZAxis(vm,viz,prefix='XYZ_',visible=False):
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
            gv.setVisibility(gname,'OFF')

def freeze(robot,indexToLock,referenceConfigurationName=None,rebuildData=True,verbose=True):
    '''
    Reduce the model by freezing all joint whose name contain the key string.
    robot: a robot wrapper where the result is stored (destructive mode)
    indexToLock: indexes of the joints to lock
    referenceConfigurationName: if given, robot.q0 is set to the value of
    robot.model.referenceConfigurations[referenceConfigurationName] after the reduction.
    rebuildData: if true, robot.rebuildData() is called.
    '''
    robot.rmbak = rmbak = robot.model
    robot.model,(robot.visual_model,robot.collision_model) = \
        pin.buildReducedModel(robot.model,[robot.visual_model,robot.collision_model],indexToLock,robot.q0)

    if referenceConfigurationName is None:
        del robot.q0
    else:
        robot.q0 = robot.model.referenceConfigurations[referenceConfigurationName]

    if rebuildData:
        robot.rebuildData()

    if hasattr(robot,'constraint_models'):
        toremove = []
        for cm in robot.constraint_models:
            if verbose: print('Freeze: Reset the placement of constraint ',cm.name)
            n1 = rmbak.names[cm.joint1_id]
            n2 = rmbak.names[cm.joint2_id]

            # The reference joints might have been frozen
            # Then seek for the corresponding frame, that might be either a joint frame
            # or a op frame. 
            idf1 = robot.model.getFrameId(n1)
            f1 = robot.model.frames[idf1]
            idf2 = robot.model.getFrameId(n2)
            f2 = robot.model.frames[idf2]

            # Make the new reference joints the parent of the frame.
            cm.joint1_id = f1.parentJoint
            cm.joint2_id = f2.parentJoint
            # In the best case, the joint still exist, then it corresponds to a joint frame
            if f1.type != pin.JOINT:
                assert(f1.type == pin.FIXED_JOINT)
                # If the joint has be freezed, the contact now should be referenced with respect
                # to the new joint, which was a parent of the previous.
                cm.joint1_placement = f1.placement*cm.joint1_placement
            # Same for the second joint
            if f2.type != pin.JOINT:
                assert(f2.type == pin.FIXED_JOINT)
                cm.joint2_placement = f2.placement*cm.joint2_placement
            
            if cm.joint1_id == cm.joint2_id:
                toremove.append(cm)
                if verbose: print(f'Freeze: Remove constraint {n1}//{n2} (during freeze)')

            '''
            # Convert previous indexes to new joint list (after some joints are frozen)
            n1 = rmbak.names[cm.joint1_id]
            n2 = rmbak.names[cm.joint2_id]
            cm.joint1_id = robot.model.getJointId(n1)
            cm.joint2_id = robot.model.getJointId(n2)
            # If some constraints are now useless, remove them
            # Todo: this might be overrestrictive
            if cm.joint1_id==robot.model.njoints or cm.joint2_id==robot.model.njoints:
                f1 = robot.model.frames[robot.model.getFrameId(n1)]
                f2 = robot.model.frames[robot.model.getFrameId(n2)]
                # Simple assert to raise an error when the TODO will become necessary
                # ... we may have frozen the joints, but they are attached to two
                # ... moving frame. In that case, rebuild the constraint with different
                # ... joint_placement
                assert(f1.parentJoint == f2.parentJoint)
                toremove.append(cm)
                print(f'Remove constraint {n1}//{n2}')
            '''
        robot.constraint_models = [ cm for cm in robot.constraint_models if cm not in toremove ]
                
                
def renameConstraints(robot):
    for cm in robot.constraint_models:
        cm.name = f'{robot.model.names[cm.joint1_id]},{robot.model.names[cm.joint2_id]}'
