#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd())
sys.path.append("..")
import argparse
import collections

from lib.ksy import rfp_client, rfp_server
from lib import vnc_demo

from universe.spaces import vnc_event
from universe.vncdriver import server_messages

from PIL import Image, ImageDraw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--demo", required=True, help="Demo directory path")
    args = parser.parse_args()

    file_name = os.path.join(args.demo, "client.fbs")
    client_header, client_messages = \
        vnc_demo.read_fbp_file(file_name, rfp_client.RfpClient, rfp_client.RfpClient.Header, rfp_client.RfpClient.Message)
    print("Client file processed, it has %d messages" % len(client_messages))

    file_name = os.path.join(args.demo, "server.fbs")
    srv_header, srv_messages = \
        vnc_demo.read_fbp_file(file_name, rfp_server.RfpServer, rfp_server.RfpServer.Header, rfp_server.RfpServer.Message)
    print("Server file processed, it has %d messages" % len(srv_messages))

    client = vnc_demo.Client(srv_header)
    numpy_screen = client.framebuffer.numpy_screen
    numpy_screen.set_paint_cursor(True)

    server_deque = collections.deque(srv_messages)

    start_ts = None
    last_save = None

    for idx, (ts, msg) in enumerate(client_messages):
        if start_ts is None:
            start_ts = ts

        # apply server messages to the framebuffer
        while server_deque:
            top_ts = server_deque[0][0]

            if top_ts > ts:
                break
            msg = server_deque.popleft()[1]

            # framebuffer update
            if msg.message_type == rfp_server.RfpServer.MessageType.fb_update:
                rects = []
                for msg_r in msg.message_body.rects:
                    rect = client.decode_rectangle(msg_r)
                    if rect is not None:
                        rects.append(rect)
                update = server_messages.FramebufferUpdate(rects)
                numpy_screen.flip()
                numpy_screen.apply(update)
                numpy_screen.flip()

        # pass client action to framebuffer to track cursor position
        if msg.message_type == 5:   # TODO: enum
            event = vnc_event.PointerEvent(msg.message_body.pos_x, msg.message_body.pos_y, msg.message_body.button_mask)
            numpy_screen.flip()
            numpy_screen.apply_action(event)
            numpy_screen.flip()

            # if button was pressed, record the image
            if msg.message_body.button_mask or last_save is None or (ts - last_save) > 0.5:
                n = "img_%04d_%.4f_%d.png" % (idx, ts - start_ts, msg.message_body.button_mask)
                img = Image.fromarray(numpy_screen.peek())
                draw = ImageDraw.Draw(img)
                y_ofs = msg.message_body.pos_y
                x_ofs = msg.message_body.pos_x
                if msg.message_body.button_mask:
                    size = 10
                else:
                    size = 2
                draw.ellipse(
                    (x_ofs, y_ofs, x_ofs + size, y_ofs + size),
                    (0, 0, 255, 128))
                img.save(n)
                last_save = ts

    pass

