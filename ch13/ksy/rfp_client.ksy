meta:
  id: rfp_client
  file-extension: rfp_client
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
      - id: challenge_response
        size: 16
        doc: Encrypted challenge bytes
      - id: client_init
        contents: [1]
  message:
    seq:
      - id: message_type
        type: u1
      - id: message_body
        type:
          switch-on: message_type
          cases:
            0: msg_set_pixel_format
            2: msg_set_encoding
            3: msg_fb_update_req
            4: msg_key_event
            5: msg_pointer_event
            6: msg_cut_text
  msg_set_pixel_format:
    seq:
      - id: padding
        size: 3
      - id: pixel_format
        size: 16
  msg_set_encoding:
    seq:
      - id: padding
        size: 1
      - id: num_encodings
        type: u2
      - id: encodings
        type: s4
        repeat: expr
        repeat-expr: num_encodings
  msg_fb_update_req:
    seq:
      - id: incremental
        type: u1
      - id: pos_x
        type: u2
      - id: pos_y
        type: u2
      - id: width
        type: u2
      - id: height
        type: u2
  msg_key_event:
    seq:
      - id: down_flag
        type: u1
      - id: padding
        size: 2
      - id: key
        type: u4
  msg_pointer_event:
    seq:
      - id: button_mask
        type: u1
      - id: pos_x
        type: u2
      - id: pos_y
        type: u2
  msg_cut_text:
    seq:
      - id: padding
        size: 3
      - id: length
        type: u4
      - id: text
        size: length
