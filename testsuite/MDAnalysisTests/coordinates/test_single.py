import MDAnalysis as mda
import numpy as np

from numpy.testing import (assert_equal, assert_array_almost_equal,
                           raises)
import tempdir

from MDAnalysisTests.coordinates.data import XTC, XYZ


class BaseReference(object):
    def __init__(self):
        self.trajectory = None
        self.n_atoms = 5
        self.n_frames = 5
        # default for the numpy test functions
        self.prec = 6
        self.first_frame = np.arange(3 * self.n_atoms).reshape(self.n_atoms, 3)
        self.second_frame = 2 ** 1 * self.first_frame
        self.last_frame = 2 ** 4 * self.first_frame
        self.jump_to_frame = 3  # second to last frame
        # remember frames are 0 indexed
        self.jump_to_frame_positions = 2 ** 3 * self.first_frame
        self.has_forces = False
        self.has_velocities = False
        self.dimensions = np.array([80, 80, 80, 60, 60, 90], dtype=np.float32)
        self.volume = mda.lib.mdamath.box_volume(self.dimensions)
        self.time = 0
        self.dt = 1
        self.totaltime = 5


class BaseReaderTest(object):
    def __init__(self, reference):
        self.ref = reference
        self.reader = self.ref.reader(self.ref.trajectory)

    def test_n_atoms(self):
        assert_equal(self.reader.n_atoms, self.ref.n_atoms)

    def test_n_frames(self):
        assert_equal(len(self.reader), self.ref.n_frames)

    def test_first_frame(self):
        self.reader.rewind()
        assert_array_almost_equal(self.reader.ts.positions,
                                  self.ref.first_frame, decimal=self.ref.prec)

    def test_last_frame(self):
        ts = self.reader[-1]
        assert_array_almost_equal(ts.positions, self.ref.last_frame,
                                  decimal=self.ref.prec)

    def test_next_gives_second_frame(self):
        reader = self.ref.reader(self.ref.trajectory)
        ts = reader.next()
        assert_array_almost_equal(ts.positions, self.ref.second_frame,
                                  decimal=self.ref.prec)

    @raises(IndexError)
    def test_go_over_last_frame(self):
        self.reader[self.ref.n_frames + 1]

    def test_frame_jump(self):
        ts = self.reader[self.ref.jump_to_frame]
        assert_array_almost_equal(ts.positions,
                                  self.ref.jump_to_frame_positions,
                                  decimal=self.ref.prec)

    def test_get_writer(self):
        with tempdir.in_tempdir():
            self.outfile = 'test-writer' + self.ref.ext
            with self.reader.Writer(self.outfile) as W:
                assert_equal(isinstance(W, self.ref.writerclass), True)
                assert_equal(W.n_atoms, self.reader.n_atoms)

    def test_get_writer_2(self):
        with tempdir.in_tempdir():
            self.outfile = 'test-writer' + self.ref.ext
            with self.reader.Writer(self.outfile, n_atoms=100) as W:
                assert_equal(isinstance(W, self.ref.writerclass), True)
                assert_equal(W.n_atoms, 100)

    def test_has_velocities(self):
        assert(self.reader.ts.has_velocities, self.ref.has_velocities)

    def test_velocities(self):
        if self.ref.has_velocities:
            ts = self.reader.rewind()
            assert_array_almost_equal(ts.velocities,
                                      self.ref.velocities,
                                      self.ref.prec)

    def test_has_forces(self):
        assert(self.reader.ts.has_forces, self.ref.has_forces)

    def test_forces(self):
        if self.ref.has_forces:
            ts = self.reader.rewind()
            assert_array_almost_equal(ts.forces, self.ref.forces,
                                      self.ref.prec)

    def test_dt(self):
        assert_equal(self.reader.dt, self.ref.dt)

    def test_total_time(self):
        assert_equal(self.reader.totaltime, self.ref.totaltime)

    def test_dimensions(self):
        assert_array_almost_equal(self.reader.ts.dimensions,
                                  self.ref.dimensions,
                                  decimal=self.ref.prec)

    def test_volume(self):
        self.reader.rewind()
        vol = self.reader.ts.volume
        assert_array_almost_equal(vol, self.ref.volume)

    def test_iter(self):
        for i, ts in enumerate(self.reader):
            assert_array_almost_equal(ts.positions,
                                      2**i * self.ref.first_frame,
                                      decimal=self.ref.prec)
            assert_equal(i, ts.time)


class XTCReference(BaseReference):
    def __init__(self):
        super(XTCReference, self).__init__()
        self.trajectory = XTC
        self.reader = mda.coordinates.XTC.XTCReader
        self.prec = 3
        self.writerclass = mda.coordinates.XTC.XTCWriter
        self.ext = 'xtc'


class TestXTCReader(BaseReaderTest):
    def __init__(self):
        super(TestXTCReader, self).__init__(XTCReference())

    def test_xtc_specific(self):
        assert(True)


class XYZReference(BaseReference):
    def __init__(self):
        super(XYZReference, self).__init__()
        self.trajectory = XYZ
        self.reader = mda.coordinates.XYZ.XYZReader
        self.writerclass = mda.coordinates.XYZ.XYZWriter
        self.ext = 'xyz'
        self.volume = 0
        self.dimensions = np.zeros(6)


class TestXYZReader(BaseReaderTest):
    def __init__(self):
        super(TestXYZReader, self).__init__(XYZReference())
