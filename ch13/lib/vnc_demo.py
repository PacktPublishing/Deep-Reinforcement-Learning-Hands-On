import io
import glob
import struct
import os.path
import collections
import gym
import universe

import numpy as np

from universe.spaces import vnc_event
from universe.vncdriver import fbs_reader, server_messages, vnc_client
from kaitaistruct import KaitaiStream

from .ksy import rfp_client, rfp_server
from . import wob_vnc


def iterate_demo_dirs(dir_name, env_name):
    for env_file_name in glob.glob(os.path.join(dir_name, "**", "env_id.txt"), recursive=True):
        with open(env_file_name, "r", encoding='utf-8') as fd:
            dir_env_name = fd.readline()
            if dir_env_name != env_name:
                continue
        yield os.path.dirname(env_file_name)


def load_demo(dir_name, env_name):
    """
    Loads demonstration from the specified directory, filtering by env name
    :param dir_name:
    :param env_name:
    :return: list of (obs, action) tuples
    """
    result = []

    env = gym.make(env_name)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)

    def mouse_to_action(pointer_event):
        return env._action_to_discrete(pointer_event)

    for demo_dir in iterate_demo_dirs(dir_name, env_name):
        client_header, client_messages = \
            read_fbp_file(os.path.join(demo_dir, "client.fbs"),
                          rfp_client.RfpClient, rfp_client.RfpClient.Header,
                          rfp_client.RfpClient.Message)

        srv_header, srv_messages = \
            read_fbp_file(os.path.join(demo_dir, "server.fbs"),
                          rfp_server.RfpServer, rfp_server.RfpServer.Header,
                          rfp_server.RfpServer.Message)

        samples = extract_samples(client_header, client_messages,
                                  srv_header, srv_messages, mouse_to_action=mouse_to_action)
        result.extend(samples)

    return result


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

    def decode_rectangle(self, msg_rect):
        """
        Convert message rectangle into VNC driver rectangle object
        :param msg_rect:
        :return:
        """
        assert isinstance(msg_rect, rfp_server.RfpServer.Rectangle)
        if msg_rect.header.encoding == rfp_server.RfpServer.Encoding.raw:
            return server_messages.RAWEncoding.parse_rectangle(
                self, msg_rect.header.pos_x, msg_rect.header.pos_y,
                msg_rect.header.width, msg_rect.header.height,
                msg_rect.body.data)
        elif msg_rect.header.encoding == rfp_server.RfpServer.Encoding.cursor:
            return server_messages.PseudoCursorEncoding.parse_rectangle(
                self, msg_rect.header.pos_x, msg_rect.header.pos_y,
                msg_rect.header.width, msg_rect.header.height,
                msg_rect.body.data)
        else:
            print("Warning! Unsupported encoding requested: %s" % msg_rect.header.encoding)


def default_mouse_to_action(pointer_event):
    pos_x, pos_y = pointer_event.x, pointer_event.y
    x = pos_x - wob_vnc.X_OFS
    y = pos_y - wob_vnc.Y_OFS - 50
    action = (y // 10) + (x // 10) * 16
    if action > 255 or action < 0:
        return None
    return action


def extract_samples(client_header, client_messages, srv_header, srv_messages,
                    mouse_to_action=default_mouse_to_action):
    samples = []
    client = Client(srv_header)
    numpy_screen = client.framebuffer.numpy_screen
    numpy_screen.set_paint_cursor(True)

    server_deque = collections.deque(srv_messages)

    for idx, (ts, msg) in enumerate(client_messages):
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

            # if button was pressed, record the observation and the event
            if msg.message_body.button_mask:
                img = crop_image(numpy_screen.peek().copy())
                action = mouse_to_action(event)
                if action is not None:
                    samples.append((img, action))

    return samples


def crop_image(buffer):
    img = buffer[wob_vnc.Y_OFS:wob_vnc.Y_OFS+wob_vnc.HEIGHT, wob_vnc.X_OFS:wob_vnc.X_OFS+wob_vnc.WIDTH, :]
    return np.transpose(img, (2, 0, 1))
