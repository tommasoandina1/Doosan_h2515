import casadi as cs
import numpy as np
from typing import Union

from adam.casadi.casadi_like import SpatialMath
from adam.core import RBDAlgorithms
from adam.core.constants import Representations
from adam.casadi import KinDynComputations
from adam.model import Model, URDFModelFactory

class NewKinDynComputations (KinDynComputations):

    def __init__( #creazione costruttore, self rappresenta istanza
        self,
        urdfstring: str,
        joints_name_list: list = None,
        root_link: str = "root_link",
        gravity: np.array = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0]),
        f_opts: dict = dict(jit=False, jit_options=dict(flags="-Ofast")),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        math = SpatialMath()
        factory = URDFModelFactory(path=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        self.g = gravity
        self.f_opts = f_opts
       

    def ddq_fun(self) -> cs.Function:
        
        
        C = cs.SX.sym("C", self.NDoF)
        h = cs.SX.sym("h", self.NDoF)
        ddq = C + h
    
        return cs.Function(
            "ddq",
            [C, h],  # Lista delle variabili simboliche in ingresso
            [ddq.array],  # Lista delle espressioni simboliche in uscita
            self.f_opts
        
            )

   

