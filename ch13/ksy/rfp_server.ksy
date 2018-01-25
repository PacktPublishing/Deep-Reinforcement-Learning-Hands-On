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
      - id: message_body
        type:
          switch-on: message_type
          cases:
            0: msg_fb_update
            1: msg_fb_update
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
            -239: rect_cursor_pseudo_encoding
            0: rect_raw_encoding
            1: rect_copy_rect_encoding
            2: rect_rre_encoding
            16: rect_zrle_encoding
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
        type: s4
  rect_cursor_pseudo_encoding:
    seq:
      - id: data
        size: _parent.header.width * _parent.header.height * (_root.header.server_init.pixel_format.bpp / 8)
      - id: bitmask
        size: _parent.header.height * ((_parent.header.width + 7) >> 3)
  rect_raw_encoding:
    seq:
      - id: data
        size: _parent.header.width * _parent.header.height * (_root.header.server_init.pixel_format.bpp / 8)
  rect_copy_rect_encoding:
    seq:
      - id: src_x
        type: u2
      - id: src_y
        type: u2
  rect_rre_encoding:
    seq:
      - id: subrects_count
        type: u4
      - id: background
        size: _root.header.server_init.pixel_format.bpp / 8
  rect_zrle_encoding:
    seq:
      - id: length
        type: u4
      - id: zlib_data
        size: length
