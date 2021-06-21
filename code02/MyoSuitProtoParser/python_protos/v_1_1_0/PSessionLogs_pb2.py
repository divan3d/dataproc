# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: PSessionLogs.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='PSessionLogs.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x12PSessionLogs.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"N\n\x0cPSessionLogs\x12\x12\n\nEntityType\x18\x01 \x01(\x05\x12\x15\n\rNexusProtVers\x18\x02 \x01(\t\x12\x13\n\x0bSessionGUID\x18\x03 \x01(\t\"\xfa\x0b\n\x12PSessionLogsDetail\x12\x31\n\rDataTimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0f\n\x07RS_AccA\x18\x02 \x01(\x05\x12\x0f\n\x07RS_AccB\x18\x03 \x01(\x05\x12\x0f\n\x07RS_GyrC\x18\x04 \x01(\x05\x12\x0f\n\x07RS_AFlt\x18\x05 \x01(\x05\x12\x0f\n\x07LS_AccA\x18\x06 \x01(\x05\x12\x0f\n\x07LS_AccB\x18\x07 \x01(\x05\x12\x0f\n\x07LS_GyrC\x18\x08 \x01(\x05\x12\x0f\n\x07LS_AFlt\x18\t \x01(\x05\x12\x0f\n\x07RT_AccA\x18\n \x01(\x05\x12\x0f\n\x07RT_AccB\x18\x0b \x01(\x05\x12\x0f\n\x07RT_GyrC\x18\x0c \x01(\x05\x12\x0f\n\x07RT_AFlt\x18\r \x01(\x05\x12\x0f\n\x07LT_AccA\x18\x0e \x01(\x05\x12\x0f\n\x07LT_AccB\x18\x0f \x01(\x05\x12\x0f\n\x07LT_GyrC\x18\x10 \x01(\x05\x12\x0f\n\x07LT_AFlt\x18\x11 \x01(\x05\x12\x0f\n\x07Tr_AccA\x18\x12 \x01(\x05\x12\x0f\n\x07Tr_AccB\x18\x13 \x01(\x05\x12\x0f\n\x07Tr_GyrC\x18\x14 \x01(\x05\x12\x0f\n\x07Tr_AFlt\x18\x15 \x01(\x05\x12\x0e\n\x06RM_Enc\x18\x16 \x01(\x05\x12\x0e\n\x06LM_Enc\x18\x17 \x01(\x05\x12\x0f\n\x07RM_CurR\x18\x18 \x01(\x05\x12\x0f\n\x07LM_CurR\x18\x19 \x01(\x05\x12\x0f\n\x07RM_CurS\x18\x1a \x01(\x05\x12\x0f\n\x07LM_CurS\x18\x1b \x01(\x05\x12\x0f\n\x07RM_Temp\x18\x1c \x01(\x05\x12\x0f\n\x07LM_Temp\x18\x1d \x01(\x05\x12\x0e\n\x06RM_Vel\x18\x1e \x01(\x05\x12\x0e\n\x06LM_Vel\x18\x1f \x01(\x05\x12\x0c\n\x04\x42t_U\x18  \x01(\r\x12\x0e\n\x06\x45rr_Al\x18! \x01(\x05\x12\x0f\n\x07Sys_inf\x18\" \x01(\x05\x12\r\n\x05R_leg\x18# \x01(\x05\x12\r\n\x05L_leg\x18$ \x01(\x05\x12\x0c\n\x04Mode\x18% \x01(\x05\x12\x11\n\tRLegForce\x18& \x01(\x05\x12\x11\n\tLLegForce\x18\' \x01(\x05\x12\x11\n\tRLegCalib\x18( \x01(\x05\x12\x11\n\tLLegCalib\x18) \x01(\x05\x12\x10\n\x08PatState\x18* \x01(\x05\x12\x0f\n\x07\x44ummy06\x18+ \x01(\x05\x12\x0f\n\x07\x44ummy07\x18, \x01(\x05\x12\x0f\n\x07\x44ummy08\x18- \x01(\x05\x12\x0f\n\x07\x44ummy09\x18. \x01(\x05\x12\x0f\n\x07\x44ummy10\x18/ \x01(\x05\x12\x0f\n\x07\x44ummy11\x18\x30 \x01(\x05\x12\x0f\n\x07\x44ummy12\x18\x31 \x01(\x05\x12\x0f\n\x07\x44ummy13\x18\x32 \x01(\x05\x12\x0f\n\x07\x44ummy14\x18\x33 \x01(\x05\x12\x0f\n\x07\x44ummy15\x18\x34 \x01(\x05\x12\x0f\n\x07\x44ummy16\x18\x35 \x01(\x05\x12\x0f\n\x07\x44ummy17\x18\x36 \x01(\x05\x12\x0f\n\x07\x44ummy18\x18\x37 \x01(\x05\x12\x0f\n\x07\x44ummy19\x18\x38 \x01(\x05\x12\x12\n\nDataLength\x18\x39 \x01(\x05\x12\x0f\n\x07RS_AccC\x18: \x01(\x05\x12\x0f\n\x07RS_GyrB\x18; \x01(\x05\x12\x0f\n\x07RS_GyrA\x18< \x01(\x05\x12\x0f\n\x07RT_AccC\x18= \x01(\x05\x12\x0f\n\x07RT_GyrB\x18> \x01(\x05\x12\x0f\n\x07RT_GyrA\x18? \x01(\x05\x12\x0f\n\x07LS_AccC\x18@ \x01(\x05\x12\x0f\n\x07LS_GyrB\x18\x41 \x01(\x05\x12\x0f\n\x07LS_GyrA\x18\x42 \x01(\x05\x12\x0f\n\x07LT_AccC\x18\x43 \x01(\x05\x12\x0f\n\x07LT_GyrB\x18\x44 \x01(\x05\x12\x0f\n\x07LT_GyrA\x18\x45 \x01(\x05\x12\x0f\n\x07Tr_AccC\x18\x46 \x01(\x05\x12\x0f\n\x07Tr_GyrB\x18G \x01(\x05\x12\x0f\n\x07Tr_GyrA\x18H \x01(\x05\x12\x0f\n\x07RS_MagA\x18I \x01(\x05\x12\x0f\n\x07RS_MagB\x18J \x01(\x05\x12\x0f\n\x07RS_MagC\x18K \x01(\x05\x12\x0f\n\x07RT_MagA\x18L \x01(\x05\x12\x0f\n\x07RT_MagB\x18M \x01(\x05\x12\x0f\n\x07RT_MagC\x18N \x01(\x05\x12\x0f\n\x07LS_MagA\x18O \x01(\x05\x12\x0f\n\x07LS_MagB\x18P \x01(\x05\x12\x0f\n\x07LS_MagC\x18Q \x01(\x05\x12\x0f\n\x07LT_MagA\x18R \x01(\x05\x12\x0f\n\x07LT_MagB\x18S \x01(\x05\x12\x0f\n\x07LT_MagC\x18T \x01(\x05\x12\x0f\n\x07Tr_MagA\x18U \x01(\x05\x12\x0f\n\x07Tr_MagB\x18V \x01(\x05\x12\x0f\n\x07Tr_MagC\x18W \x01(\x05\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,])




_PSESSIONLOGS = _descriptor.Descriptor(
  name='PSessionLogs',
  full_name='PSessionLogs',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='EntityType', full_name='PSessionLogs.EntityType', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='NexusProtVers', full_name='PSessionLogs.NexusProtVers', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='SessionGUID', full_name='PSessionLogs.SessionGUID', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=55,
  serialized_end=133,
)


_PSESSIONLOGSDETAIL = _descriptor.Descriptor(
  name='PSessionLogsDetail',
  full_name='PSessionLogsDetail',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='DataTimestamp', full_name='PSessionLogsDetail.DataTimestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_AccA', full_name='PSessionLogsDetail.RS_AccA', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_AccB', full_name='PSessionLogsDetail.RS_AccB', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_GyrC', full_name='PSessionLogsDetail.RS_GyrC', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_AFlt', full_name='PSessionLogsDetail.RS_AFlt', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_AccA', full_name='PSessionLogsDetail.LS_AccA', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_AccB', full_name='PSessionLogsDetail.LS_AccB', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_GyrC', full_name='PSessionLogsDetail.LS_GyrC', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_AFlt', full_name='PSessionLogsDetail.LS_AFlt', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_AccA', full_name='PSessionLogsDetail.RT_AccA', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_AccB', full_name='PSessionLogsDetail.RT_AccB', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_GyrC', full_name='PSessionLogsDetail.RT_GyrC', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_AFlt', full_name='PSessionLogsDetail.RT_AFlt', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_AccA', full_name='PSessionLogsDetail.LT_AccA', index=13,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_AccB', full_name='PSessionLogsDetail.LT_AccB', index=14,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_GyrC', full_name='PSessionLogsDetail.LT_GyrC', index=15,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_AFlt', full_name='PSessionLogsDetail.LT_AFlt', index=16,
      number=17, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_AccA', full_name='PSessionLogsDetail.Tr_AccA', index=17,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_AccB', full_name='PSessionLogsDetail.Tr_AccB', index=18,
      number=19, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_GyrC', full_name='PSessionLogsDetail.Tr_GyrC', index=19,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_AFlt', full_name='PSessionLogsDetail.Tr_AFlt', index=20,
      number=21, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RM_Enc', full_name='PSessionLogsDetail.RM_Enc', index=21,
      number=22, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LM_Enc', full_name='PSessionLogsDetail.LM_Enc', index=22,
      number=23, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RM_CurR', full_name='PSessionLogsDetail.RM_CurR', index=23,
      number=24, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LM_CurR', full_name='PSessionLogsDetail.LM_CurR', index=24,
      number=25, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RM_CurS', full_name='PSessionLogsDetail.RM_CurS', index=25,
      number=26, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LM_CurS', full_name='PSessionLogsDetail.LM_CurS', index=26,
      number=27, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RM_Temp', full_name='PSessionLogsDetail.RM_Temp', index=27,
      number=28, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LM_Temp', full_name='PSessionLogsDetail.LM_Temp', index=28,
      number=29, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RM_Vel', full_name='PSessionLogsDetail.RM_Vel', index=29,
      number=30, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LM_Vel', full_name='PSessionLogsDetail.LM_Vel', index=30,
      number=31, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Bt_U', full_name='PSessionLogsDetail.Bt_U', index=31,
      number=32, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Err_Al', full_name='PSessionLogsDetail.Err_Al', index=32,
      number=33, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Sys_inf', full_name='PSessionLogsDetail.Sys_inf', index=33,
      number=34, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='R_leg', full_name='PSessionLogsDetail.R_leg', index=34,
      number=35, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='L_leg', full_name='PSessionLogsDetail.L_leg', index=35,
      number=36, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Mode', full_name='PSessionLogsDetail.Mode', index=36,
      number=37, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RLegForce', full_name='PSessionLogsDetail.RLegForce', index=37,
      number=38, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LLegForce', full_name='PSessionLogsDetail.LLegForce', index=38,
      number=39, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RLegCalib', full_name='PSessionLogsDetail.RLegCalib', index=39,
      number=40, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LLegCalib', full_name='PSessionLogsDetail.LLegCalib', index=40,
      number=41, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='PatState', full_name='PSessionLogsDetail.PatState', index=41,
      number=42, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy06', full_name='PSessionLogsDetail.Dummy06', index=42,
      number=43, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy07', full_name='PSessionLogsDetail.Dummy07', index=43,
      number=44, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy08', full_name='PSessionLogsDetail.Dummy08', index=44,
      number=45, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy09', full_name='PSessionLogsDetail.Dummy09', index=45,
      number=46, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy10', full_name='PSessionLogsDetail.Dummy10', index=46,
      number=47, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy11', full_name='PSessionLogsDetail.Dummy11', index=47,
      number=48, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy12', full_name='PSessionLogsDetail.Dummy12', index=48,
      number=49, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy13', full_name='PSessionLogsDetail.Dummy13', index=49,
      number=50, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy14', full_name='PSessionLogsDetail.Dummy14', index=50,
      number=51, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy15', full_name='PSessionLogsDetail.Dummy15', index=51,
      number=52, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy16', full_name='PSessionLogsDetail.Dummy16', index=52,
      number=53, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy17', full_name='PSessionLogsDetail.Dummy17', index=53,
      number=54, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy18', full_name='PSessionLogsDetail.Dummy18', index=54,
      number=55, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Dummy19', full_name='PSessionLogsDetail.Dummy19', index=55,
      number=56, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='DataLength', full_name='PSessionLogsDetail.DataLength', index=56,
      number=57, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_AccC', full_name='PSessionLogsDetail.RS_AccC', index=57,
      number=58, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_GyrB', full_name='PSessionLogsDetail.RS_GyrB', index=58,
      number=59, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_GyrA', full_name='PSessionLogsDetail.RS_GyrA', index=59,
      number=60, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_AccC', full_name='PSessionLogsDetail.RT_AccC', index=60,
      number=61, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_GyrB', full_name='PSessionLogsDetail.RT_GyrB', index=61,
      number=62, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_GyrA', full_name='PSessionLogsDetail.RT_GyrA', index=62,
      number=63, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_AccC', full_name='PSessionLogsDetail.LS_AccC', index=63,
      number=64, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_GyrB', full_name='PSessionLogsDetail.LS_GyrB', index=64,
      number=65, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_GyrA', full_name='PSessionLogsDetail.LS_GyrA', index=65,
      number=66, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_AccC', full_name='PSessionLogsDetail.LT_AccC', index=66,
      number=67, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_GyrB', full_name='PSessionLogsDetail.LT_GyrB', index=67,
      number=68, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_GyrA', full_name='PSessionLogsDetail.LT_GyrA', index=68,
      number=69, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_AccC', full_name='PSessionLogsDetail.Tr_AccC', index=69,
      number=70, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_GyrB', full_name='PSessionLogsDetail.Tr_GyrB', index=70,
      number=71, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_GyrA', full_name='PSessionLogsDetail.Tr_GyrA', index=71,
      number=72, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_MagA', full_name='PSessionLogsDetail.RS_MagA', index=72,
      number=73, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_MagB', full_name='PSessionLogsDetail.RS_MagB', index=73,
      number=74, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RS_MagC', full_name='PSessionLogsDetail.RS_MagC', index=74,
      number=75, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_MagA', full_name='PSessionLogsDetail.RT_MagA', index=75,
      number=76, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_MagB', full_name='PSessionLogsDetail.RT_MagB', index=76,
      number=77, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RT_MagC', full_name='PSessionLogsDetail.RT_MagC', index=77,
      number=78, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_MagA', full_name='PSessionLogsDetail.LS_MagA', index=78,
      number=79, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_MagB', full_name='PSessionLogsDetail.LS_MagB', index=79,
      number=80, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LS_MagC', full_name='PSessionLogsDetail.LS_MagC', index=80,
      number=81, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_MagA', full_name='PSessionLogsDetail.LT_MagA', index=81,
      number=82, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_MagB', full_name='PSessionLogsDetail.LT_MagB', index=82,
      number=83, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='LT_MagC', full_name='PSessionLogsDetail.LT_MagC', index=83,
      number=84, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_MagA', full_name='PSessionLogsDetail.Tr_MagA', index=84,
      number=85, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_MagB', full_name='PSessionLogsDetail.Tr_MagB', index=85,
      number=86, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tr_MagC', full_name='PSessionLogsDetail.Tr_MagC', index=86,
      number=87, type=5, cpp_type=1, label=1,
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
  serialized_start=136,
  serialized_end=1666,
)

_PSESSIONLOGSDETAIL.fields_by_name['DataTimestamp'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
DESCRIPTOR.message_types_by_name['PSessionLogs'] = _PSESSIONLOGS
DESCRIPTOR.message_types_by_name['PSessionLogsDetail'] = _PSESSIONLOGSDETAIL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PSessionLogs = _reflection.GeneratedProtocolMessageType('PSessionLogs', (_message.Message,), {
  'DESCRIPTOR' : _PSESSIONLOGS,
  '__module__' : 'PSessionLogs_pb2'
  # @@protoc_insertion_point(class_scope:PSessionLogs)
  })
_sym_db.RegisterMessage(PSessionLogs)

PSessionLogsDetail = _reflection.GeneratedProtocolMessageType('PSessionLogsDetail', (_message.Message,), {
  'DESCRIPTOR' : _PSESSIONLOGSDETAIL,
  '__module__' : 'PSessionLogs_pb2'
  # @@protoc_insertion_point(class_scope:PSessionLogsDetail)
  })
_sym_db.RegisterMessage(PSessionLogsDetail)


# @@protoc_insertion_point(module_scope)
