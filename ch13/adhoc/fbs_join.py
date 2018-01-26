#!/usr/bin/env python3
import io
import os
import sys
sys.path.append(os.getcwd())
sys.path.append("..")
import argparse
import struct

from universe.vncdriver import fbs_reader, server_messages, vnc_client
from lib.ksy import rfp_client, rfp_server
from kaitaistruct import KaitaiStream

from PIL import Image


def read_fbp_file(file_name, msg_root_class, msg_header_class, msg_class):
    reader = fbs_reader.FBSReader(file_name)
    buf = io.BytesIO()
    stream = KaitaiStream(buf)
    header = None
    messages = []
    last_ofs = 0
    _root = None

    for dat, ts in reader:
        buf.seek(0, io.SEEK_END)
        buf.write(dat)
        buf.seek(last_ofs, io.SEEK_SET)

        try:
            if header is None:
                header = msg_header_class(stream, _root=msg_root_class)
                buf.seek(0, io.SEEK_SET)
                _root = msg_root_class(stream)
            else:
                msg = msg_class(stream, _root=_root, _parent=_root)
                messages.append((ts, msg))
            last_ofs = buf.tell()
        except Exception as e:
            pass
    return header, messages


def decode_rectangle(client, msg_rect):
    """
    Convert message rectangle into VNC driver rectangle object
    :param msg_rect:
    :return:
    """
    assert isinstance(msg_rect, rfp_server.RfpServer.Rectangle)
    if msg_rect.header.encoding == rfp_server.RfpServer.Encoding.raw:
        return server_messages.RAWEncoding.parse_rectangle(
            client, msg_rect.header.pos_x, msg_rect.header.pos_y,
            msg_rect.header.width, msg_rect.header.height,
            msg_rect.body.data)
    elif msg_rect.header.encoding == rfp_server.RfpServer.Encoding.cursor:
        return server_messages.PseudoCursorEncoding.parse_rectangle(
            client, msg_rect.header.pos_x, msg_rect.header.pos_y,
            msg_rect.header.width, msg_rect.header.height,
            msg_rect.body.data)
    else:
        print("Warning! Unsupported encoding requested: %s" % msg_rect.header.encoding)

class Client:
    def __init__(self, server_header):
        assert isinstance(server_header, rfp_server.RfpServer.Header)
        srv_init = server_header.server_init
        pixel_format_block = struct.pack("!BBBBHHHBBBxxx", srv_init.pixel_format.bpp,
                                         srv_init.pixel_format.depth, srv_init.pixel_format.big_endian,
                                         srv_init.pixel_format.true_color, srv_init.pixel_format.red_max,
                                         srv_init.pixel_format.green_max, srv_init.pixel_format.blue_max,
                                         srv_init.pixel_format.red_shift, srv_init.pixel_format.green_shift,
                                         srv_init.pixel_format.blue_shift)
        self.framebuffer = vnc_client.Framebuffer(server_header.server_init.width,
                                                  server_header.server_init.height,
                                                  pixel_format_block,
                                                  bytes(server_header.server_init.name, encoding='utf-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--demo", required=True, help="Demo directory path")
    args = parser.parse_args()

    file_name = os.path.join(args.demo, "client.fbs")
    client_header, client_messages = \
        read_fbp_file(file_name, rfp_client.RfpClient, rfp_client.RfpClient.Header, rfp_client.RfpClient.Message)
    print("Client file processed, it has %d messages" % len(client_messages))

    file_name = os.path.join(args.demo, "server.fbs")
    srv_header, srv_messages = \
        read_fbp_file(file_name, rfp_server.RfpServer, rfp_server.RfpServer.Header, rfp_server.RfpServer.Message)
    print("Server file processed, it has %d messages" % len(srv_messages))

    client = Client(srv_header)
    numpy_screen = client.framebuffer.numpy_screen

    for idx, (ts, msg) in enumerate(srv_messages):
        # framebuffer update
        if msg.message_type == rfp_server.RfpServer.MessageType.fb_update:
            rects = []
            for msg_r in msg.message_body.rects:
                rect = decode_rectangle(client, msg_r)
                if rect is not None:
                    rects.append(rect)
            update = server_messages.FramebufferUpdate(rects)
            numpy_screen.flip()
            numpy_screen.apply(update)
            numpy_screen.flip()
            n = "img_%04d_%.4f.png" % (idx, ts)
            img = Image.fromarray(numpy_screen.peek())
            img.save(n)




    pass

