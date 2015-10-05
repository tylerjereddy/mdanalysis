import MDAnalysis as mda
import numpy as np

from numpy.testing import (assert_equal, assert_array_almost_equal,
                           raises)
import tempdir

from MDAnalysisTests.coordinates.data import PDB, XTC


class BaseReference(object):
    def __init__(self):
        self.topology = PDB
        self.trajectory = None
        self.n_atoms = 5
        self.n_frames = 5
        # default for the numpy test functions
        self.prec = 6
        ar = np.arange(5)
        self.first_frame = np.vstack((ar, ar, ar)).T
        self.second_frame = 2 * self.first_frame
        self.last_frame = 16 * self.first_frame
        self.jump_to_frame = 3  # second to last frame
        self.jump_to_frame_positions = 8 * self.first_frame
        self.has_forces = False
        self.has_velocities = False
        self.dimensions = np.array([80, 80, 80, 60, 60, 90], dtype=np.float32)
        self.volume = 0
        self.time = 0
        self.dt = 0.0
        self.totaltime = 0


class BaseReaderTest(object):
    def __init__(self, reference):
        self.ref = reference
        self.u = mda.Universe(self.ref.topology, self.ref.trajectory)

    def test_n_atoms(self):
        assert_equal(self.u.atoms.n_atoms, self.ref.n_atoms)

    def test_n_frames(self):
        assert_equal(len(self.u.trajectory), self.ref.n_frames)

    def test_first_frame(self):
        self.u.trajectory.rewind()
        assert_array_almost_equal(self.u.atoms.positions, self.ref.first_frame,
                                  decimal=self.ref.prec)

    def test_last_frame(self):
        self.u.trajectory[-1]
        assert_array_almost_equal(self.u.atoms.positions, self.ref.last_frame,
                                  decimal=self.ref.prec)

    def test_next_gives_second_frame(self):
        u = mda.Universe(self.ref.topology, self.ref.trajectory)
        ts = u.trajectory.next()
        assert_array_almost_equal(ts.positions, self.ref.second_frame,
                                  decimal=self.ref.prec)

    @raises(IndexError)
    def test_go_over_last_frame(self):
        self.u.trajectory[self.ref.n_frames + 1]

    def test_frame_jump(self):
        self.u.trajectory[self.ref.jump_to_frame]
        assert_array_almost_equal(self.u.atoms.positions,
                                  self.ref.jump_to_frame_positions,
                                  decimal=self.ref.prec)

    def test_reader_pick(self):
        assert(isinstance(self.u.trajectory, self.ref.readerclass))

    def test_get_writer(self):
        with tempdir.in_tempdir():
            self.outfile = 'test-writer' + self.ref.ext
            with self.u.trajectory.Writer(self.outfile) as W:
                assert_equal(isinstance(W, self.ref.writerclass), True)
                assert_equal(W.n_atoms, self.u.trajectory.n_atoms)

    def test_get_writer_2(self):
        with tempdir.in_tempdir():
            self.outfile = 'test-writer' + self.ref.ext
            with self.u.trajectory.Writer(self.outfile, n_atoms=100) as W:
                assert_equal(isinstance(W, self.ref.writerclass), True)
                assert_equal(W.n_atoms, 100)

    def test_has_velocities(self):
        assert(self.u.trajectory.ts.has_velocities, self.ref.has_velocities)

    def test_velocities(self):
        if self.ref.has_velocities:
            self.u.ts.rewind()
            assert_array_almost_equal(self.u.atoms.velocities,
                                      self.ref.velocities,
                                      self.ref.prec)

    def test_has_forces(self):
        assert(self.u.trajectory.ts.has_forces, self.ref.has_forces)

    def test_forces(self):
        if self.ref.has_forces:
            self.u.ts.rewind()
            assert_array_almost_equal(self.u.atoms.forces, self.ref.forces,
                                      self.ref.prec)

    def test_time(self):
        assert_equal(self.u.trajectory.time, self.ref.time)

    def test_dt(self):
        assert_equal(self.u.trajectory.dt, self.ref.dt)

    def test_total_time(self):
        assert_equal(self.u.trajectory.totaltime, self.ref.totaltime)

    # def test_residues(self):
    #     for res, ref_res in zip(self.u.atoms.residues, self.ref.residues):
    #         assert_equal(res == ref_res)

    def test_dimensions(self):
        assert_array_almost_equal(self.u.trajectory.ts.dimensions,
                                  self.ref.dimensions,
                                  decimal=self.ref.prec)

    def test_volume(self):
        self.u.trajectory.rewind()
        vol = self.u.trajectory.ts.volume
        assert_array_almost_equal(vol, self.ref.volume)

    # @raises(ValueError)
    # def test_load_incompatible_file(self):
    #     # wrong trj should contain a different number of atoms
    #     self.universe.load_new(self.ref.wrong_trj)


class XTCReference(BaseReference):
    def __init__(self):
        super(XTCReference, self).__init__()
        self.trajectory = XTC
        self.prec = 3
        self.readerclass = mda.coordinates.XTC.XTCReader
        self.writerclass = mda.coordinates.XTC.XTCWriter
        self.ext = 'xtc'


class TestXTCReader(BaseReaderTest):
    def __init__(self):
        super(TestXTCReader, self).__init__(XTCReference())

    def test_xtc_specific(self):
        assert(True)
