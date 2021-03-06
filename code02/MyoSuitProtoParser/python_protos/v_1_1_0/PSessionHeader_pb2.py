# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: PSessionHeader.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='PSessionHeader.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x14PSessionHeader.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"\x9f\x06\n\x0ePSessionHeader\x12\x12\n\nEntityType\x18\x01 \x01(\x05\x12\x15\n\rNexusProtVers\x18\x02 \x01(\t\x12\x11\n\tPatientId\x18\x03 \x01(\x05\x12\x36\n\x12GenerationDateTime\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0e\n\x06Height\x18\x05 \x01(\x05\x12\x0e\n\x06Weight\x18\x06 \x01(\x05\x12\x10\n\x08\x43oncROMR\x18\x07 \x01(\x05\x12\x0f\n\x07\x45\x63\x63ROMR\x18\x08 \x01(\x05\x12\x10\n\x08\x43oncROML\x18\t \x01(\x05\x12\x0f\n\x07\x45\x63\x63ROML\x18\n \x01(\x05\x12\x0e\n\x06Size01\x18\x0b \x01(\t\x12\x0e\n\x06Size02\x18\x0c \x01(\t\x12\x0e\n\x06Size03\x18\r \x01(\t\x12\x0e\n\x06Size04\x18\x0e \x01(\t\x12\x0e\n\x06Size05\x18\x0f \x01(\t\x12\x0e\n\x06Size06\x18\x10 \x01(\t\x12\x0e\n\x06Size07\x18\x11 \x01(\t\x12\x0e\n\x06\x43oncFR\x18\x1a \x01(\x05\x12\x0e\n\x06\x43oncFL\x18\x1b \x01(\x05\x12\x0e\n\x06\x45\x63\x63\x65\x46R\x18\x1c \x01(\x05\x12\x0e\n\x06\x45\x63\x63\x65\x46L\x18\x1d \x01(\x05\x12\x0e\n\x06IsomFR\x18\x1e \x01(\x05\x12\x0e\n\x06IsomFL\x18\x1f \x01(\x05\x12\x12\n\nCalAngMaxR\x18  \x01(\x05\x12\x12\n\nCalAngMinR\x18! \x01(\x05\x12\x12\n\nCalAngMaxL\x18\" \x01(\x05\x12\x12\n\nCalAngMinL\x18# \x01(\x05\x12\x17\n\x0f\x46irmwareVersion\x18$ \x01(\t\x12\x18\n\x10UnitSerialNumber\x18% \x01(\t\x12\x32\n\x0eStartOfSession\x18& \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x30\n\x0c\x45ndOfSession\x18\' \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x12\n\nOperatorId\x18( \x01(\x05\x12\x10\n\x08\x43linicId\x18) \x01(\x05\x12\x0f\n\x07StepNum\x18* \x01(\x05\x12\x16\n\x0eUnitStatusCode\x18+ \x01(\x05\x12\x13\n\x0bSessionGUID\x18, \x01(\t\x12\x14\n\x0cOperatorType\x18- \x01(\x05\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,])




_PSESSIONHEADER = _descriptor.Descriptor(
  name='PSessionHeader',
  full_name='PSessionHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='EntityType', full_name='PSessionHeader.EntityType', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='NexusProtVers', full_name='PSessionHeader.NexusProtVers', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='PatientId', full_name='PSessionHeader.PatientId', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='GenerationDateTime', full_name='PSessionHeader.GenerationDateTime', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Height', full_name='PSessionHeader.Height', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Weight', full_name='PSessionHeader.Weight', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ConcROMR', full_name='PSessionHeader.ConcROMR', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='EccROMR', full_name='PSessionHeader.EccROMR', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ConcROML', full_name='PSessionHeader.ConcROML', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='EccROML', full_name='PSessionHeader.EccROML', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Size01', full_name='PSessionHeader.Size01', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Size02', full_name='PSessionHeader.Size02', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Size03', full_name='PSessionHeader.Size03', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Size04', full_name='PSessionHeader.Size04', index=13,
      number=14, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Size05', full_name='PSessionHeader.Size05', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Size06', full_name='PSessionHeader.Size06', index=15,
      number=16, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Size07', full_name='PSessionHeader.Size07', index=16,
      number=17, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ConcFR', full_name='PSessionHeader.ConcFR', index=17,
      number=26, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ConcFL', full_name='PSessionHeader.ConcFL', index=18,
      number=27, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='EcceFR', full_name='PSessionHeader.EcceFR', index=19,
      number=28, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='EcceFL', full_name='PSessionHeader.EcceFL', index=20,
      number=29, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='IsomFR', full_name='PSessionHeader.IsomFR', index=21,
      number=30, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='IsomFL', full_name='PSessionHeader.IsomFL', index=22,
      number=31, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='CalAngMaxR', full_name='PSessionHeader.CalAngMaxR', index=23,
      number=32, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='CalAngMinR', full_name='PSessionHeader.CalAngMinR', index=24,
      number=33, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='CalAngMaxL', full_name='PSessionHeader.CalAngMaxL', index=25,
      number=34, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='CalAngMinL', full_name='PSessionHeader.CalAngMinL', index=26,
      number=35, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='FirmwareVersion', full_name='PSessionHeader.FirmwareVersion', index=27,
      number=36, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='UnitSerialNumber', full_name='PSessionHeader.UnitSerialNumber', index=28,
      number=37, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='StartOfSession', full_name='PSessionHeader.StartOfSession', index=29,
      number=38, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='EndOfSession', full_name='PSessionHeader.EndOfSession', index=30,
      number=39, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='OperatorId', full_name='PSessionHeader.OperatorId', index=31,
      number=40, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ClinicId', full_name='PSessionHeader.ClinicId', index=32,
      number=41, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='StepNum', full_name='PSessionHeader.StepNum', index=33,
      number=42, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='UnitStatusCode', full_name='PSessionHeader.UnitStatusCode', index=34,
      number=43, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='SessionGUID', full_name='PSessionHeader.SessionGUID', index=35,
      number=44, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='OperatorType', full_name='PSessionHeader.OperatorType', index=36,
      number=45, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=857,
)

_PSESSIONHEADER.fields_by_name['GenerationDateTime'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_PSESSIONHEADER.fields_by_name['StartOfSession'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_PSESSIONHEADER.fields_by_name['EndOfSession'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
DESCRIPTOR.message_types_by_name['PSessionHeader'] = _PSESSIONHEADER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PSessionHeader = _reflection.GeneratedProtocolMessageType('PSessionHeader', (_message.Message,), {
  'DESCRIPTOR' : _PSESSIONHEADER,
  '__module__' : 'PSessionHeader_pb2'
  # @@protoc_insertion_point(class_scope:PSessionHeader)
  })
_sym_db.RegisterMessage(PSessionHeader)


# @@protoc_insertion_point(module_scope)
