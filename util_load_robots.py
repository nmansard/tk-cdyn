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

def freeze(robot,key,referenceConfigurationName=None,rebuildData=True):
    '''
    Reduce the model by freezing all joint whose name contain the key string.
    robot: a robot wrapper where the result is stored (destructive mode)
    key: the string to search in the joint names to lock.
    '''
    idx = [ i for i,n in enumerate(robot.model.names) if key in n] # to lock
    robot.rmbak = rmbak = robot.model
    print('reduce')
    robot.model,(robot.visual_model,robot.collision_model) = \
        pin.buildReducedModel(robot.model,[robot.visual_model,robot.collision_model],idx,robot.q0)
    print('q0')
    if referenceConfigurationName is None:
        del robot.q0
    else:
        robot.q0 = robot.model.referenceConfigurations[referenceConfigurationName]
    print('rebuild')
    if rebuildData:
        robot.rebuildData()
    if hasattr(robot,'constraint_models'):
        print('cmodel')
        toremove = []
        for cm in robot.constraint_models:
            print(cm.name)
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
                print(f'Remove constraint {n1}//{n2} (during freeze)')

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

# List of joint string keys that should typically be locked
classic_cassie_blocker = [
    '-ip',
    'right-roll-joint',
    'right-yaw-joint',
    'right-pitch-joint',
    'right-knee-joint',
    
    'right-achilles-spring-joint',
    'right-plantar-foot-joint',
    'right-foot-joint',
]
cassie_spring_knee_joints = [
    # Useless parallel spring in knee
    'right-knee-spring-joint',
    'right-knee-shin-joint',
    'right-shin-spring-joint',
    #'right-shin-tarsus-joint',
    'right-tarsus-spring-joint',
]
classic_cassie_unecessary_constraints =\
    [
        'right-roll-joint,right-roll-op',
        'right-yaw-joint,right-yaw-op',
        'right-pitch-joint,right-pitch-op',
        'right-knee-joint,right-knee-op',
        'right-foot-joint,right-foot-op',

        # Useless parallel spring in knee
        #'right-shin-spring-joint,right-knee-spring-joint'
    ]


def fixCassieConstraints(cassie):
    '''
    Some constraints are unproperly define (why?). This hack fixes them. But we should 
    understand why and make a better fix.
    '''
    i = cassie.model.getJointId('right-crank-rod-joint')
    idx_q,idx_v = cassie.model.joints[i].idx_q,cassie.model.joints[i].idx_v
    cassie.model.joints[i] = pin.JointModelSpherical()
    cassie.model.joints[i].setIndexes(i,idx_q,idx_v)
    j1 = cassie.model.joints[i]
    ax1 = cassie.data.joints[j1.id].S[3:]
    print(j1)
    
    i = cassie.model.getJointId('right-plantar-foot-joint')
    cassie.model.joints[i] = pin.JointModelRZ()
    cassie.model.joints[i].setIndexes(i,idx_q+4,idx_v+3)
    cassie.data = cassie.model.createData()
    j2 = cassie.model.joints[i]
    print(j2)
    
    for nq in cassie.model.referenceConfigurations:
        n,q = nq.key(),nq.data().copy()
        print(n,j1,j1.id)
        rot = pin.AngleAxis(q[j1.idx_q],ax1)
        quat = pin.Quaternion(rot.matrix())
        #pin.Quaternion(pin.utils.rotate('x', q[j1.idx_q])).coeffs()
        cassie.model.referenceConfigurations[n][j1.idx_q:j1.idx_q+j1.nq] = quat.coeffs()
        cassie.model.referenceConfigurations[n][j2.idx_q] = 0 #pin.Quaternion(pin.utils.rotate('x', q[j1.idx_q])).coeffs()
        #np.arctan2(*((q[32:34]*2).tolist()))
        
    for cm in (cm for cm in cassie.constraint_models
               if 'achilles-spring-joint' in cassie.model.names[cm.joint1_id]
               and 'tarsus-spring-joint' in cassie.model.names[cm.joint2_id]):
        print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()  # M2 is Id, so inverse not needed, but I have the intuition it is more generic.
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D
    
    for cm in (cm for cm in cassie.constraint_models
               if 'shin-spring-joint' in cassie.model.names[cm.joint1_id]
               and 'right-knee-spring-joint' in cassie.model.names[cm.joint2_id]):
        print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D

    for cm in (cm for cm in cassie.constraint_models
               if 'plantar-foot-joint' in cassie.model.names[cm.joint1_id]
               and 'foot-op' in cassie.model.names[cm.joint2_id]):
        print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D
