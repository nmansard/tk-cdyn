'''
Unit test for the cassie model.
Load cassie in two version, then fix (manually) the model in one of the two versions.
Then compare the constraint placements in both: they should match the reference values in the
fixed version, and not in the broken (default) version.
Finally, if you have Gepetto Viewer open, display the constraints and highlight the discrepancy.
'''

import pinocchio as pin
import numpy as np
from util_load_cassie import loadCassieAndFixIt
import util_load_cassie as ulc
import time

WITH_DISP = not False
if not WITH_DISP:
    print('You should consider starting Gepetto Viewer')

# These internal variables set the joints to be locked when building the model.
# Change the default values here to lock the full left leg, but no other joints of the right leg.
ulc.classic_cassie_blocker = ['left']
ulc.cassie_spring_knee_joints = []

# Load two versions of Cassie
# Classic version, without modification of the model.
print('Load basic cassie model (brok)')
cassieBrok = loadCassieAndFixIt(fixConstraints=False,initViewer=WITH_DISP,verbose=True)
# Fixed version, where the constraints have been replaced.
print('Load cassie and fix it')
cassie = loadCassieAndFixIt(initViewer=WITH_DISP,verbose=True,forceXYZaxis=True)

# Reference placement of the constraints, stored for comparison.
# The following values have been obtain on nmansard's laptop with pinocchio installed from source
# (6ec680523097a31987575e72b73deb6d811c5183) and SDF 12 (apt libsdformat12-dev)
M = {}
M["right-achilles-spring-joint,right-tarsus-spring-joint"] = \
    pin.SE3(np.array([[0.1681157853165133, -0.9829808837038772, 0.07406527671509773, -0.8689068296071387],
                      [0.036699191324859055, -0.06884146040719646, -0.9969523673100465, -0.12466241922444474],
                      [0.9850838808436302, 0.1703215659141855, 0.02450126734108854, 0.4438323746256995],
                      [0.0, 0.0, 0.0, 1.0]]))
M["right-plantar-foot-joint,right-foot-op"] = \
    pin.SE3(np.array([[0.7632139583506243, 0.6461458455943018, -7.444051763894066e-10, -0.621012881509093],
                      [0.00016015029283055968, -0.0001891673283380968, -0.9999999692838025, -0.12255318016765288],
                      [-0.6461458257472994, 0.7632139349074748, -0.00024785559183424996, 0.019378267727866855],
                      [0.0, 0.0, 0.0, 1.0]]))
M["right-shin-spring-joint,right-knee-spring-joint"] = \
    pin.SE3(np.array([[-0.731146771296769, 0.6822201980834116, 1.2220628471919658e-05, -0.6330442333414052],
                      [0.00016025006130932412, 0.00018965569091280556, -0.9999999691753177, -0.13068079958724582],
                      [-0.6822201793719025, -0.7311467468010461, -0.00024799197463757874, 0.5374706837342822],
                      [0.0, 0.0, 0.0, 1.0]]))
q0fixed = np.array([-5.27093838e-01, -2.76381629e-04,  9.39151616e-01,  9.61380040e-06,
                    9.38217754e-04,  2.14890379e-05,  9.99999560e-01, -1.29520800e-02,
                    1.31779500e-02,  2.25860000e-04,  1.65861400e-02, -1.66263000e-02,
                    -4.01635934e-05,  1.59122513e+00, -1.39653428e+00,  1.94690847e-01,
                    -6.04403000e-03, -4.60930000e-03, -8.35047900e-02,  9.96478390e-01,
                    -1.63406124e-14,  1.40275306e+00, -1.63016464e+00, -2.27411580e-01,
                    -7.87499336e-06, -7.87499336e-06,  6.92937810e-01, -6.33415540e-01,
                    -1.75391974e+00,  1.57796813e-02,  7.52173967e-02,  7.65956408e-01,
                    6.38282168e-01,  0.00000000e+00,  1.39961783e+00, -3.13328631e+00,
                    -1.73366848e+00, -2.75627693e-07])
q0brok = np.array([-5.27093838e-01, -2.76381629e-04,  9.39151616e-01,  9.61380040e-06,
                   9.38217754e-04,  2.14890379e-05,  9.99999560e-01, -1.29520800e-02,
                   1.31779500e-02,  2.25860000e-04,  1.65861400e-02, -1.66263000e-02,
                   -4.01635934e-05,  1.59122513e+00, -1.39653428e+00,  1.94690847e-01,
                   -6.04403000e-03, -4.60930000e-03, -8.35047900e-02,  9.96478390e-01,
                   -1.63406124e-14,  1.40275306e+00, -1.63016464e+00, -2.27411580e-01,
                   -7.87499336e-06, -7.87499336e-06,  6.92937810e-01, -6.33415540e-01,
                   -1.75391974e+00,  1.75706331e+00, -4.83732315e-14, -2.80375818e-06,
                   -7.63300126e-01,  6.46044053e-01,  1.39961783e+00, -3.13328631e+00,
                   -1.73366848e+00, -2.75627693e-07])
assert(np.allclose(cassie.q0,q0fixed))
assert(np.allclose(cassieBrok.q0,q0brok))

# Check placement of the constraint in cassie-fixed. They should match the
# reference placements.
pin.framesForwardKinematics(cassie.model,cassie.data,cassie.q0)
for cm in cassie.constraint_models:
    M1 = cassie.data.oMi[cm.joint1_id]*cm.joint1_placement
    M2 = cassie.data.oMi[cm.joint2_id]*cm.joint2_placement
    if cm.name == 'right-plantar-foot-joint,right-foot-op':
        # Due to the model correction, this constraint is only aligned in 3D
        assert( np.allclose(M1.translation-M2.translation,0) )
    else:
        assert( np.allclose(pin.log(M1.inverse()*M2),0) )
    print(f'M["{cm.name}"] = pin.SE3(np.array({M1.homogeneous.tolist()}))')
    assert( np.allclose(pin.log(M1.inverse()*M[cm.name]),0) )

# Check placement of the constraints in cassie-broken. They should not match the
# reference placements.
pin.framesForwardKinematics(cassieBrok.model,cassieBrok.data,cassieBrok.q0)
for cm in cassieBrok.constraint_models:
    M1 = cassieBrok.data.oMi[cm.joint1_id]*cm.joint1_placement
    M2 = cassieBrok.data.oMi[cm.joint2_id]*cm.joint2_placement
    assert( np.allclose(pin.log(M1.inverse()*M2),0) )
    assert( not np.allclose(pin.log(M1.inverse()*M[cm.name]),0) )

# ### Visual debug #############################################################################

if WITH_DISP:
    # Unhide the XYZaxis in gepetto viewer, and store the geom names in the constraints of cassie-fixed
    for cm in cassie.constraint_models:
        idxs = [ cassie.visual_model.getGeometryId(f'XYZ_cst_{cm.name}_1'),
                 cassie.visual_model.getGeometryId(f'XYZ_cst_{cm.name}_2') ]
        gnames = [ cassie.viz.getViewerNodeName(cassie.visual_model.geometryObjects[idx],pin.VISUAL)
                   for idx in idxs ]
        cm.joint1_gname,cm.joint2_gname = gnames
        for n in gnames:
            cassie.viz.viewer.gui.setVisibility(n,'ON')

    # Store the geom names in the constraints of cassie-fixed
    for cm in cassieBrok.constraint_models:
        idxs = [ cassie.visual_model.getGeometryId(f'XYZ_cst_{cm.name}_1'),
                 cassie.visual_model.getGeometryId(f'XYZ_cst_{cm.name}_2') ]
        gnames = [ cassie.viz.getViewerNodeName(cassie.visual_model.geometryObjects[idx],pin.VISUAL)
                   for idx in idxs ]
        cm.joint1_gname,cm.joint2_gname = gnames

    print('Blinking from the broken configuration to the fixed one or the next 5 seconds')
    for i in range(5):
        cassie.display(cassie.q0)
        time.sleep(.5)
        cassieBrok.display(cassieBrok.q0)
        time.sleep(.5)
