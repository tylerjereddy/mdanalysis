# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

"""
GRO topology parser
===================

Read a list of atoms from a GROMOS/Gromacs GRO coordinate file to
build a basic topology.

Atom types and masses are guessed.

See Also
--------
:mod:`MDAnalysis.coordinates.GRO`



Classes
-------

.. autoclass:: GROParser
   :members:
   :inherited-members:

"""
import re
import numpy as np

from ..lib.util import openany
from ..core.topologyattrs import (
    Atomnames,
    Atomtypes,
    Atomids,
    Masses,
    Resids,
    Resnames,
    Resnums,
    Segids,
)
from ..core.topology import Topology
from .base import TopologyReaderBase, change_squash
from . import guessers


class GROParser(TopologyReaderBase):
    """Reads a Gromacs GRO file

    Reads the following attributes:
      - resids
      - resnames
      - atomids
      - atomnames

    Guesses the following attributes
      - atomtypes
      - masses
    """
    format = 'GRO'

    def parse(self, **kwargs):
        """Return the *Topology* object for this file"""
        # Gro has the following columns
        # resid, resname, name, index, (x,y,z)
        with openany(self.filename) as inf:
            next(inf)
            n_atoms = int(next(inf))

            # Allocate shizznizz
            resids = np.zeros(n_atoms, dtype=np.int32)
            resnames = np.zeros(n_atoms, dtype=object)
            names = np.zeros(n_atoms, dtype=object)
            indices = np.zeros(n_atoms, dtype=np.int32)

            # compile regex to parse full file string
            # data to avoid pure Python iteration over
            # lines
            prog = re.compile(r'''
                               # start of line and any spaces
                               ^
                               # then any digits are the resids
                               (.{5})
                               # followed by the resnames
                               (.{5})
                               # then space before "atom" names
                               (.{5})
                               # then (maybe) spaces and atom numbers
                               (.{5})
                               # then 3 groups of nums
                               # and EOL
                               \s+[+-]?\d+\.?\d+\b
                               \s+[+-]?\d+\.?\d+\b
                               \s+[+-]?\d+\.?\d+\s*
                               $
                               ''',
                              re.M|re.X|re.I)

            str_data = inf.read()
            try:
                result = prog.findall(str_data)
                resids[:] = [int(ele[0].strip()) for ele in result]
                resnames[:] = [ele[1].strip() for ele in result]
                names[:] = [ele[2].strip() for ele in result]
                indices[:] = [int(ele[3].strip()) for ele in result]
            except ValueError:
                raise IOError

        # Check all lines had names
        if not np.all(names):
            missing = np.where(names == '')
            raise IOError("Missing atom name on line: {0}"
                          "".format(missing[0][0] + 3))  # 2 header, 1 based

        # Fix wrapping of resids (if we ever saw a wrap)
        if np.any(resids == 0):
            # find places where resid hit zero again
            wraps = np.where(resids == 0)[0]
            # group these places together:
            # find indices of first 0 in each block of zeroes
            # 1) find large changes in index, (ie non sequential blocks)
            diff = np.diff(wraps) != 1
            # 2) make array of where 0-blocks start
            starts = np.hstack([wraps[0], wraps[1:][diff]])

            # remove 0 in starts, ie the first residue **can** be 0
            if starts[0] == 0:
                starts = starts[1:]

            # for each resid after a wrap, add 100k (5 digit wrap)
            for s in starts:
                resids[s:] += 100000

        # Guess types and masses
        atomtypes = guessers.guess_types(names)
        masses = guessers.guess_masses(atomtypes)

        residx, (new_resids, new_resnames) = change_squash(
                                (resids, resnames), (resids, resnames))

        # new_resids is len(residues)
        # so resindex 0 has resid new_resids[0]
        attrs = [
            Atomnames(names),
            Atomids(indices),
            Atomtypes(atomtypes, guessed=True),
            Resids(new_resids),
            Resnums(new_resids.copy()),
            Resnames(new_resnames),
            Masses(masses, guessed=True),
            Segids(np.array(['SYSTEM'], dtype=object))
        ]

        top = Topology(n_atoms=n_atoms, n_res=len(new_resids), n_seg=1,
                       attrs=attrs,
                       atom_resindex=residx,
                       residue_segindex=None)

        return top
