# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import array
import struct
import zlib
from enum import Enum
from pkg_resources import parse_version

from kaitaistruct import __version__ as ks_version, KaitaiStruct, KaitaiStream, BytesIO


if parse_version(ks_version) < parse_version('0.7'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.7 or later is required, but you have %s" % (ks_version))

class RfpClient(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self.header = self._root.Header(self._io, self, self._root)
        self.messages = []
        while not self._io.is_eof():
            self.messages.append(self._root.Message(self._io, self, self._root))


    class MsgSetPixelFormat(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.padding = self._io.read_bytes(3)
            self.pixel_format = self._io.read_bytes(16)


    class MsgCutText(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.padding = self._io.read_bytes(3)
            self.length = self._io.read_u4be()
            self.text = self._io.read_bytes(self.length)


    class MsgKeyEvent(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.down_flag = self._io.read_u1()
            self.padding = self._io.read_bytes(2)
            self.key = self._io.read_u4be()


    class MsgPointerEvent(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.button_mask = self._io.read_u1()
            self.pos_x = self._io.read_u2be()
            self.pos_y = self._io.read_u2be()


    class Header(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.magic = (self._io.read_bytes_term(10, False, True, True)).decode(u"ascii")
            self.challenge_response = self._io.read_bytes(16)
            self.client_init = self._io.ensure_fixed_contents(struct.pack('1b', 1))


    class Message(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.message_type = self._io.read_u1()
            _on = self.message_type
            if _on == 0:
                self.message_body = self._root.MsgSetPixelFormat(self._io, self, self._root)
            elif _on == 4:
                self.message_body = self._root.MsgKeyEvent(self._io, self, self._root)
            elif _on == 6:
                self.message_body = self._root.MsgCutText(self._io, self, self._root)
            elif _on == 3:
                self.message_body = self._root.MsgFbUpdateReq(self._io, self, self._root)
            elif _on == 5:
                self.message_body = self._root.MsgPointerEvent(self._io, self, self._root)
            elif _on == 2:
                self.message_body = self._root.MsgSetEncoding(self._io, self, self._root)


    class MsgSetEncoding(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.padding = self._io.read_bytes(1)
            self.num_encodings = self._io.read_u2be()
            self.encodings = [None] * (self.num_encodings)
            for i in range(self.num_encodings):
                self.encodings[i] = self._io.read_s4be()



    class MsgFbUpdateReq(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.incremental = self._io.read_u1()
            self.pos_x = self._io.read_u2be()
            self.pos_y = self._io.read_u2be()
            self.width = self._io.read_u2be()
            self.height = self._io.read_u2be()
