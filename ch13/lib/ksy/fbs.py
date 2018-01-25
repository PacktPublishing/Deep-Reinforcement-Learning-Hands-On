# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import array
import struct
import zlib
from enum import Enum
from pkg_resources import parse_version

from kaitaistruct import __version__ as ks_version, KaitaiStruct, KaitaiStream, BytesIO


if parse_version(ks_version) < parse_version('0.7'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.7 or later is required, but you have %s" % (ks_version))

class Fbs(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self.magic = (self._io.read_bytes_term(10, False, True, True)).decode(u"ascii")
        self.meta_start = (self._io.read_bytes_term(10, False, True, True)).decode(u"ascii")
        self.blocks = []
        while True:
            _ = self._root.Block(self._io, self, self._root)
            self.blocks.append(_)
            if _.len == 0:
                break
        self.meta_stop = (self._io.read_bytes_term(10, False, True, True)).decode(u"ascii")

    class Block(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.len = self._io.read_u4be()
            self.data = self._io.read_bytes(self.len)
            if self.len != 0:
                self.delta_millisec = self._io.read_u4be()
