import pinocchio as pin
import numpy as np
from constraints import constraintsResidual
from pinocchio import casadi as caspin
import casadi


class ProjectConfig:
    '''
    Define a projection function using NLP
    Given a reference configuration qref and an (optional) joint index iv (of Qdot=T_qQ),
    solve the following problem:

    search q
    which minimizes || q - qref ||**2
    subject to:
      c(q) = 0        # Kinematic constraint satisfied
      q_iv = qref_iv  # The commanded joint iv should move exactly
    '''

    def __init__(self,robot):

        self.robot = robot
        robot.casmodel = caspin.Model(robot.model)
        robot.casdata = robot.casmodel.createData()
        
        cq = self.cq = casadi.SX.sym("q",robot.casmodel.nq,1)
        cv = self.cv = casadi.SX.sym("v",robot.casmodel.nv,1)
        caspin.forwardKinematics(robot.casmodel,robot.casdata,cq)

        self.integrate = casadi.Function('integrate', [cq,cv],[ caspin.integrate(robot.casmodel,cq,cv) ])
        self.recomputeConstraints()
        self.verbose = True
        
    def recomputeConstraints(self):
        '''
        Call this function when the constraint activation changes. 
        This will force the recomputation of the computation graph of the constraint function.
        '''
        robot = self.robot
        constraint = constraintsResidual(robot.casmodel,robot.casdata,
                                         robot.constraint_models,robot.constraint_datas,self.cq,
                                         False,caspin)
        self.constraint = casadi.Function('constraint', [self.cq], [ constraint ])

    def __call__(self,qref,iv=None):
        '''
        Project an input configuration <qref> to the nearby feasible configuration
        If <iv> is not null, then the DOF #iv is set as hard constraints, while the other are moved.
        '''
        robot = self.robot
        if len(robot.constraint_models)==0:
            return qref
       
        self.opti = opti = casadi.Opti()

        # Decision variables
        self.vdq = vdq = opti.variable(robot.model.nv)
        self.vq = vq = self.integrate(robot.q0,vdq)

        # Cost and constraints
        totalcost = 0 
        totalcost = casadi.sumsqr(vq-qref)
        opti.subject_to( self.constraint(vq) == 0 )

        if iv is not None:
            dvref = pin.difference(robot.model,robot.q0,qref)
            opti.subject_to( vdq[iv]==dvref[iv] )

        # Solve
        if self.verbose:
            opts = {}
        else:
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver('ipopt', opts) # set numerical backend
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
