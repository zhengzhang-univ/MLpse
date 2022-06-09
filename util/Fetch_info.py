from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os.path

import yaml

from drift.telescope import (
    cylinder,
    gmrt,
    focalplane,
    restrictedcylinder,
    exotic_cylinder,
)
from drift.core import beamtransfer

from drift.core import kltransform, doublekl
from drift.core import psestimation, psmc, crosspower
from drift.core import skymodel

teltype_dict = {
    "UnpolarisedCylinder": cylinder.UnpolarisedCylinderTelescope,
    "PolarisedCylinder": cylinder.PolarisedCylinderTelescope,
    "GMRT": gmrt.GmrtUnpolarised,
    "FocalPlane": focalplane.FocalPlaneArray,
    "RestrictedCylinder": restrictedcylinder.RestrictedCylinder,
    "RestrictedPolarisedCylinder": restrictedcylinder.RestrictedPolarisedCylinder,
    "RestrictedExtra": restrictedcylinder.RestrictedExtra,
    "GradientCylinder": exotic_cylinder.GradientCylinder,
}


## KLTransform configuration
kltype_dict = {"KLTransform": kltransform.KLTransform, "DoubleKL": doublekl.DoubleKL}


## Power spectrum estimation configuration
pstype_dict = {
    "Full": psestimation.PSExact,
    "MonteCarlo": psmc.PSMonteCarlo,
    "MonteCarloAlt": psmc.PSMonteCarloAlt,
    "Cross": crosspower.CrossPower,
}

def _resolve_class(clstype, clsdict, objtype=""):
    # If clstype is a dict, try and resolve the class from `module` and
    # `class` properties. If it's a string try and resolve the class from
    # either its name and a lookup dictionary.

    if isinstance(clstype, dict):
        # Lookup custom type

        modname = clstype["module"]
        clsname = clstype["class"]

        if "file" in clstype:
            import imp

            module = imp.load_source(modname, clstype["file"])
        else:
            import importlib

            module = importlib.import_module(modname)
        cls_ref = module.__dict__[clsname]
    #hirax_transfer.core

    elif clstype in clsdict:
        cls_ref = clsdict[clstype]
    else:
        raise Exception("Unsupported %s" % objtype)

    return cls_ref

class Parameters_collection(object):
    
    @classmethod
    def from_config(cls, configfile):
        if not os.path.exists(configfile):
            raise Exception("Configuration file does not exist.")
            
        c = cls()
        c.load_config(configfile)
        
        return c.kltransforms
    
    def load_config(self, configfile):
        
        with open(configfile) as f:
            yconf = yaml.safe_load(f)
            
        self.directory = yconf["config"]["output_directory"]
        self.directory = os.path.expanduser(self.directory)
        self.directory = os.path.expandvars(self.directory)
        
        teltype = yconf["telescope"]["type"]

        telclass = _resolve_class(teltype, teltype_dict, "telescope")

        self.telescope = telclass.from_config(yconf["telescope"])
        
                ## Beam transfer generation
        if "nosvd" in yconf["config"] and yconf["config"]["nosvd"]:
            self.beamtransfer = beamtransfer.BeamTransferNoSVD(
                self.directory + "/bt/", telescope=self.telescope
            )
        else:
            self.beamtransfer = beamtransfer.BeamTransfer(
                self.directory + "/bt/", telescope=self.telescope
            )

        ## Use the full SVD if requested
        if "fullsvd" in yconf["config"] and yconf["config"]["fullsvd"]:
            self.beamtransfer = beamtransfer.BeamTransferFullSVD(
                self.directory + "/bt/", telescope=self.telescope
            )
        else:
            self.beamtransfer = beamtransfer.BeamTransfer(
                self.directory + "/bt/", telescope=self.telescope
            )

        ## Set the singular value cut for the beamtransfers
        if "svcut" in yconf["config"]:
            self.beamtransfer.svcut = float(yconf["config"]["svcut"])

        ## Set the singular value cut for the *polarisation* beamtransfers
        if "polsvcut" in yconf["config"]:
            self.beamtransfer.polsvcut = float(yconf["config"]["polsvcut"])

        if yconf["config"]["beamtransfers"]:
            self.gen_beams = True

        if "skip_svd" in yconf["config"] and yconf["config"]["skip_svd"]:
            self.skip_svd = True

        self.kltransforms = {}

        if "kltransform" in yconf:

            for klentry in yconf["kltransform"]:
                kltype = klentry["type"]
                klname = klentry["name"]

                klclass = _resolve_class(kltype, kltype_dict, "KL filter")

                kl = klclass.from_config(klentry, self.beamtransfer, subdir=klname)
                self.kltransforms[klname] = kl

        if yconf["config"]["kltransform"]:
            self.gen_kl = True
    
