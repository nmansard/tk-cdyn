import pinocchio as pin
import hppfcl
import example_robot_data as robex
import numpy as np
from util_load_robots import defRootName,addXYZAxisToJoints,replaceGeomByXYZAxis,freeze,renameConstraints,addXYZAxisToConstraints

# List of joint string keys that should typically be locked
classic_cassie_blocker = [
    # *-ip and *-joint are the gearbox parallel models. 
    '-ip',
    '-roll-joint',
    '-yaw-joint',
    '-pitch-joint',
    '-knee-joint',
    '-foot-joint',

    # The two following joints are collocated to the rod constraints
    # hence useless.
    '-achilles-spring-joint',
    '-plantar-foot-joint',
]
# The following joints are used to model the flexibilities
# in the shin parallel actuation. 
cassie_spring_knee_joints = [
    '-knee-spring-joint',
    '-knee-shin-joint',
    '-shin-spring-joint',
    '-tarsus-spring-joint',
]
# The following constraints are used to close the loop on the gear-boxes
# They must be deactivated otherwise they would lock the motors.
classic_cassie_unecessary_constraints =\
    [
        '{pre}-roll-joint,{pre}-roll-op',
        '{pre}-yaw-joint,{pre}-yaw-op',
        '{pre}-pitch-joint,{pre}-pitch-op',
        '{pre}-knee-joint,{pre}-knee-op',
        '{pre}-foot-joint,{pre}-foot-op',
    ]
# See https://stackoverflow.com/questions/42497625/how-to-postpone-defer-the-evaluation-of-f-strings
# for understanding the eval(...) syntax
classic_cassie_unecessary_constraints = \
    [ eval(f"f'{c}'") for c in classic_cassie_unecessary_constraints for pre in [ 'right', 'left'] ]

def fixCassieConstraints(cassie,referenceConfigurationName=None,verbose=False):
    '''
    Some constraints are unproperly define (why?). This hack fixes them. But we should 
    understand why and make a better fix.
    '''

    # First invert the two joints around the tarsus crank:
    # - the first should be a sphere, the second a revolute.
    for prefix in ['right', 'left']:

        # Make the crank joint a sphere
        i = cassie.model.getJointId(f'{prefix}-crank-rod-joint')
        idx_q,idx_v = cassie.model.joints[i].idx_q,cassie.model.joints[i].idx_v
        cassie.model.joints[i] = pin.JointModelSpherical()
        cassie.model.joints[i].setIndexes(i,idx_q,idx_v)
        j1 = cassie.model.joints[i]
        cassie.model.idx_qs[i] = j1.idx_q
        cassie.model.idx_vs[i] = j1.idx_v
        cassie.model.nqs[i] = j1.nq
        cassie.model.nvs[i] = j1.nv
        ax1 = cassie.data.joints[i].S[3:]
        if verbose:
            print(f'New top-rod {prefix} joint:',j1)

        # Make the rod joint a revolute
        i = cassie.model.getJointId(f'{prefix}-plantar-foot-joint')
        cassie.model.joints[i] = pin.JointModelRZ()
        cassie.model.joints[i].setIndexes(i,idx_q+4,idx_v+3)
        cassie.data = cassie.model.createData()
        j2 = cassie.model.joints[i]
        cassie.model.idx_qs[i] = j2.idx_q
        cassie.model.idx_vs[i] = j2.idx_v
        cassie.model.nqs[i] = j2.nq
        cassie.model.nvs[i] = j2.nv
        if verbose:
            print(f'New bottom-rod {prefix} joint:',j2)

        # The reference configurations should be changed to account for the quaternion
        # at crank joint.
        for nq in cassie.model.referenceConfigurations:
            # Reorder the joint in the configuration. What follows is an ad-hoc tentative
            # to automatically find a configuration respecting the same constraint as the
            # previous joint organisation. Not fully clean and succesfull, but good enough
            n,q = nq.key(),nq.data().copy()
            rot = pin.AngleAxis(q[j1.idx_q],ax1)
            quat = pin.Quaternion(rot.matrix())
            cassie.model.referenceConfigurations[n][j1.idx_q:j1.idx_q+j1.nq] = quat.coeffs()
            cassie.model.referenceConfigurations[n][j2.idx_q] = 0
    if referenceConfigurationName is not None:
        cassie.q0 = cassie.model.referenceConfigurations[referenceConfigurationName].copy()

    # Reset the placement of the tarsus contraint
    for cm in (cm for cm in cassie.constraint_models
               if 'achilles-spring-joint' in cassie.model.names[cm.joint1_id]
               and 'tarsus-spring-joint' in cassie.model.names[cm.joint2_id]):
        if verbose: print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()  # M2 is Id, so inverse not needed, but I have the intuition it is more generic.
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D

    # Reset the placement of the knee spring contraint
    for cm in (cm for cm in cassie.constraint_models
               if 'shin-spring-joint' in cassie.model.names[cm.joint1_id]
               and 'right-knee-spring-joint' in cassie.model.names[cm.joint2_id]):
        if verbose: print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D

    # Reset the placement of the feet constraint
    for cm in (cm for cm in cassie.constraint_models
               if 'plantar-foot-joint' in cassie.model.names[cm.joint1_id]
               and 'foot-op' in cassie.model.names[cm.joint2_id]):
        if verbose: print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D


def loadCassieAndFixIt(fixConstraints=True,initViewer=False,verbose=False,forceXYZaxis=False):
    '''
    Load Cassie from example robot data, then fix the model errors, 
    freeze the unecessary joints, and load the model in the viewer.
    param fixConstraint: if true, call fixCassieConstraints()
    '''

    # Load cassie from SDF using Pinocchio SDF parser
    cassie=robex.load('cassie')

    # Some constraints are ill-posed, so fix them
    if fixConstraints: fixCassieConstraints(cassie, 'standing',verbose=verbose)

    # Change the name of the root of cassie (to easier merge it with other robots)
    defRootName(cassie.model,'cassie')

    # SDF parser gives no name to constraints, rename if "joint1 - joint2"
    renameConstraints(cassie)

    # Remove unecessary constraints (those for the internal gear boxes)
    cassie.constraint_models = [ cm for cm in cassie.constraint_models
                                 if cm.name not in classic_cassie_unecessary_constraints ]

    # For display: add geometry objects for each constraints
    # that will be changed to XYZaxis in gepetto viewer
    addXYZAxisToConstraints(cassie.model,cassie.visual_model,cassie.constraint_models)

    # Reduce the model by freezing unecessary joints.
    # First make the list of joints to freeze
    jointsToLock = []
    for key in classic_cassie_blocker+cassie_spring_knee_joints:
        jointsToLock += [ i for i,n in enumerate(cassie.model.names) if key in n]
    # Freeze expect a list of uniq identifiers, so use list(set()) to enforce that
    freeze(cassie,list(set(jointsToLock)),'standing',rebuildData=False,verbose=verbose)

    # For display: add geometry objects for each joints
    # that will be changed to XYZaxis in gepetto viewer
    addXYZAxisToJoints(cassie.model,cassie.visual_model)

    # Rebuild data after resize of the model
    cassie.rebuildData()

    # Init the viewer
    if initViewer:
        # Load the model in viewer
        cassie.initViewer(loadModel=True)
        # Gepetto-viewer allow XYZaxis object that cannot be specified in the
        # pinocchio geomtry model thus replace manually the visual objects in
        # gepetto viewer by XYZaxis
        if forceXYZaxis:
            replaceGeomByXYZAxis(cassie.visual_model,cassie.viz)
            print(
                '''
                !!! Beware that Gepetto viewer does not properly deallocating
                its memory when calling this function. Try to limit the number
                of time that this message is displayed
                ''')
        # Force the initial display
        cassie.display(cassie.q0)

    return cassie
