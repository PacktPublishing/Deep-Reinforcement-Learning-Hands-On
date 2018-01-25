# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import array
import struct
import zlib
from enum import Enum
from pkg_resources import parse_version

from kaitaistruct import __version__ as ks_version, KaitaiStruct, KaitaiStream, BytesIO


if parse_version(ks_version) < parse_version('0.7'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.7 or later is required, but you have %s" % (ks_version))

class RfpServer(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self.magic = (self._io.read_bytes_term(10, False, True, True)).decode(u"ascii")
        self.some_data = self._io.read_bytes(4)
        self.challenge = self._io.read_bytes(16)
        self.security_status = self._io.read_u4be()
        self.server_init = self._root.ServerInit(self._io, self, self._root)
        self.messages = []
        while not self._io.is_eof():
            self.messages.append(self._root.Message(self._io, self, self._root))


    class RectZrleEncoding(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.length = self._io.read_u4be()
            self.zlib_data = self._io.read_bytes(self.length)


    class RectCursorPseudoEncoding(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.data = self._io.read_bytes(((self._parent.header.width * self._parent.header.height) * self._root.server_init.pixel_format.bpp // 8))
            self.bitmask = self._io.read_bytes((self._parent.header.height * ((self._parent.header.width + 7) >> 3)))


    class RectCopyRectEncoding(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.src_x = self._io.read_u2be()
            self.src_y = self._io.read_u2be()


    class RectRawEncoding(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.data = self._io.read_bytes(((self._parent.header.width * self._parent.header.height) * self._root.server_init.pixel_format.bpp // 8))


    class PixelFormat(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.bpp = self._io.read_u1()
            self.depth = self._io.read_u1()
            self.big_endian = self._io.read_u1()
            self.true_color = self._io.read_u1()
            self.red_max = self._io.read_u2be()
            self.green_max = self._io.read_u2be()
            self.blue_max = self._io.read_u2be()
            self.red_shift = self._io.read_u1()
            self.green_shift = self._io.read_u1()
            self.blue_shift = self._io.read_u1()
            self.padding = self._io.read_bytes(3)


    class RectHeader(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.pos_x = self._io.read_u2be()
            self.pos_y = self._io.read_u2be()
            self.width = self._io.read_u2be()
            self.height = self._io.read_u2be()
            self.encoding = self._io.read_s4be()


    class RectRreEncoding(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.subrects_count = self._io.read_u4be()
            self.background = self._io.read_bytes(self._root.server_init.pixel_format.bpp // 8)


    class Message(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.message_type = self._io.read_u1()
            _on = self.message_type
            if _on == 0:
                self.message_body = self._root.MsgFbUpdate(self._io, self, self._root)
            elif _on == 1:
                self.message_body = self._root.MsgFbUpdate(self._io, self, self._root)


    class MsgFbUpdate(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.padding = self._io.read_u1()
            self.rects_count = self._io.read_u2be()
            self.rects = [None] * (self.rects_count)
            for i in range(self.rects_count):
                self.rects[i] = self._root.Rectangle(self._io, self, self._root)



    class Rectangle(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.header = self._root.RectHeader(self._io, self, self._root)
            _on = self.header.encoding
            if _on == 0:
                self.body = self._root.RectRawEncoding(self._io, self, self._root)
            elif _on == 1:
                self.body = self._root.RectCopyRectEncoding(self._io, self, self._root)
            elif _on == -239:
                self.body = self._root.RectCursorPseudoEncoding(self._io, self, self._root)
            elif _on == 16:
                self.body = self._root.RectZrleEncoding(self._io, self, self._root)
            elif _on == 2:
                self.body = self._root.RectRreEncoding(self._io, self, self._root)


    class ServerInit(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.width = self._io.read_u2be()
            self.height = self._io.read_u2be()
            self.pixel_format = self._root.PixelFormat(self._io, self, self._root)
            self.name_len = self._io.read_u4be()
            self.name = (self._io.read_bytes(self.name_len)).decode(u"ascii")
