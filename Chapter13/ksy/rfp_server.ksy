meta:
  id: rfp_server
  file-extension: rfp_server
  endian: be
seq:
  - id: header
    type: header
  - id: messages
    type: message
    repeat: eos
enums:
  message_type:
    0: fb_update
    1: set_colormap
    2: bell
    3: cut_text
  encoding:
    0xFFFFFF11: cursor
    0: raw
    1: copy_rect
    2: rre
    16: zrle
types:
  header:
    seq:
      - id: magic
        type: str
        terminator: 10
        encoding: ascii
        doc: ProtocolVersion
      - id: some_data
        size: 4
      - id: challenge
        size: 16
        doc: Challenge bytes
      - id: security_status
        type: u4
      - id: server_init
        type: server_init
  server_init:
    seq:
      - id: width
        type: u2
      - id: height
        type: u2
      - id: pixel_format
        type: pixel_format
      - id: name_len
        type: u4
      - id: name
        type: str
        size: name_len
        encoding: ascii
  pixel_format:
    seq:
      - id: bpp
        type: u1
      - id: depth
        type: u1
      - id: big_endian
        type: u1
      - id: true_color
        type: u1
      - id: red_max
        type: u2
      - id: green_max
        type: u2
      - id: blue_max
        type: u2
      - id: red_shift
        type: u1
      - id: green_shift
        type: u1
      - id: blue_shift
        type: u1
      - id: padding
        size: 3
  message:
    seq:
      - id: message_type
        type: u1
        enum: message_type
      - id: message_body
        type:
          switch-on: message_type
          cases:
            'message_type::fb_update': msg_fb_update
            'message_type::set_colormap': msg_set_colormap
            'message_type::bell': msg_bell
            'message_type::cut_text': msg_cut_text
  msg_fb_update:
    seq:
      - id: padding
        type: u1
      - id: rects_count
        type: u2
      - id: rects
        type: rectangle
        repeat: expr
        repeat-expr: rects_count
  rectangle:
    seq:
      - id: header
        type: rect_header
      - id: body
        type:
          switch-on: header.encoding
          cases:
            'encoding::cursor': rect_cursor_pseudo_encoding
            'encoding::raw': rect_raw_encoding
            'encoding::copy_rect': rect_copy_rect_encoding
            'encoding::rre': rect_rre_encoding
            'encoding::zrle': rect_zrle_encoding
# TODO: other encodings
  rect_header:
    seq:
      - id: pos_x
        type: u2
      - id: pos_y
        type: u2
      - id: width
        type: u2
      - id: height
        type: u2
      - id: encoding
        type: u4
        enum: encoding
  rect_cursor_pseudo_encoding:
    seq:
      - id: data
        size: _parent.header.width * _parent.header.height * (_root.header.server_init.pixel_format.bpp / 8) + _parent.header.height * ((_parent.header.width + 7) >> 3)
  rect_raw_encoding:
    seq:
      - id: data
        size: _parent.header.width * _parent.header.height * (_root.header.server_init.pixel_format.bpp / 8)
  rect_copy_rect_encoding:
    seq:
      - id: data
        size: 4
  rect_rre_encoding:
    seq:
      - id: subrects_count
        type: u4
      - id: background
        size: _root.header.server_init.pixel_format.bpp / 8
      - id: data
        size: subrects_count * (_root.header.server_init.pixel_format.bpp / 8 + 8)
  rect_zrle_encoding:
    seq:
      - id: length
        type: u4
      - id: data
        size: length
  msg_set_colormap:
    seq:
      - id: padding
        size: 1
      - id: first_color
        type: u2
      - id: number_colors
        type: u2
      - id: data
        size: number_colors * 6
  msg_bell:
    seq:
      - id: empty
        size: 0
  msg_cut_text:
    seq:
      - id: padding
        size: 3
      - id: length
        type: u4
      - id: text
        type: str
        encoding: ascii
        size: length
