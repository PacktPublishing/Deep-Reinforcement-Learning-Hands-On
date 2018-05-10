meta:
  id: fbs
  file-extension: fbs
  endian: be
seq:
  - id: magic
    type: str
    terminator: 10
    encoding: ascii
  - id: meta_start
    type: str
    terminator: 10
    encoding: ascii
  - id: blocks
    type: block
    repeat: until
    repeat-until: _.len == 0
  - id: meta_stop
    type: str
    terminator: 10
    encoding: ascii
types:
  block:
    seq:
      - id: len
        type: u4
      - id: data
        size: len
      - id: delta_millisec
        type: u4
        if: len != 0
