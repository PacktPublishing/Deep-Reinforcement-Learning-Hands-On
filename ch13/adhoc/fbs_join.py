#!/usr/bin/env python3
import io
import os
import sys
sys.path.append(os.getcwd())
sys.path.append("..")
import argparse

from universe.vncdriver.fbs_reader import FBSReader
from lib.ksy import rfp_client, rfp_server
from kaitaistruct import KaitaiStream


def read_fbp_file(file_name, msg_root_class, msg_header_class, msg_class):
    reader = FBSReader(file_name)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--demo", required=True, help="Demo directory path")
    args = parser.parse_args()

    file_name = os.path.join(args.demo, "client.fbs")
    header, messages = read_fbp_file(file_name, rfp_client.RfpClient,
                                     rfp_client.RfpClient.Header,
                                     rfp_client.RfpClient.Message)
    print("Client file processed, it has %d messages" % len(messages))

    file_name = os.path.join(args.demo, "server.fbs")
    header, messages = read_fbp_file(file_name, rfp_server.RfpServer,
                                     rfp_server.RfpServer.Header,
                                     rfp_server.RfpServer.Message)
    print("Server file processed, it has %d messages" % len(messages))

    pass

