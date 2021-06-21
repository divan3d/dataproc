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
  serialized_pb=b'\n\x12PSessionLogs.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"v\n\x0cPSessionLogs\x12\x12\n\nEntityType\x18\x01 \x01(\x05\x12\x15\n\rNexusProtVers\x18\x02 \x01(\t\x12\x13\n\x0bSessionGUID\x18\x03 \x01(\t\x12&\n\tLogDetail\x18\x04 \x03(\x0b\x32\x13.PSessionLogsDetail\"\xf7\x07\n\x12PSessionLogsDetail\x12\x31\n\rDataTimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0f\n\x07RS_AccA\x18\x02 \x01(\x05\x12\x0f\n\x07RS_AccB\x18\x03 \x01(\x05\x12\x0e\n\x06RS_Gyr\x18\x04 \x01(\x05\x12\x0f\n\x07RS_AFlt\x18\x05 \x01(\x05\x12\x0f\n\x07LS_AccA\x18\x06 \x01(\x05\x12\x0f\n\x07LS_AccB\x18\x07 \x01(\x05\x12\x0e\n\x06LS_Gyr\x18\x08 \x01(\x05\x12\x0f\n\x07LS_AFlt\x18\t \x01(\x05\x12\x0f\n\x07RT_AccA\x18\n \x01(\x05\x12\x0f\n\x07RT_AccB\x18\x0b \x01(\x05\x12\x0e\n\x06RT_Gyr\x18\x0c \x01(\x05\x12\x0f\n\x07RT_AFlt\x18\r \x01(\x05\x12\x0f\n\x07LT_AccA\x18\x0e \x01(\x05\x12\x0f\n\x07LT_AccB\x18\x0f \x01(\x05\x12\x0e\n\x06LT_Gyr\x18\x10 \x01(\x05\x12\x0f\n\x07LT_AFlt\x18\x11 \x01(\x05\x12\x0f\n\x07Tr_AccA\x18\x12 \x01(\x05\x12\x0f\n\x07Tr_AccB\x18\x13 \x01(\x05\x12\x0e\n\x06Tr_Gyr\x18\x14 \x01(\x05\x12\x0f\n\x07Tr_AFlt\x18\x15 \x01(\x05\x12\x0e\n\x06RM_Enc\x18\x16 \x01(\x05\x12\x0e\n\x06LM_Enc\x18\x17 \x01(\x05\x12\x0f\n\x07RM_CurR\x18\x18 \x01(\x05\x12\x0f\n\x07LM_CurR\x18\x19 \x01(\x05\x12\x0f\n\x07RM_CurS\x18\x1a \x01(\x05\x12\x0f\n\x07LM_CurS\x18\x1b \x01(\x05\x12\x0f\n\x07RM_Temp\x18\x1c \x01(\x05\x12\x0f\n\x07LM_Temp\x18\x1d \x01(\x05\x12\x0e\n\x06RM_Vel\x18\x1e \x01(\x05\x12\x0e\n\x06LM_Vel\x18\x1f \x01(\x05\x12\x0c\n\x04\x42t_U\x18  \x01(\r\x12\x0e\n\x06\x45rr_Al\x18! \x01(\x05\x12\x0f\n\x07Sys_inf\x18\" \x01(\x05\x12\r\n\x05R_leg\x18# \x01(\x05\x12\r\n\x05L_leg\x18$ \x01(\x05\x12\x0c\n\x04Mode\x18% \x01(\x05\x12\x11\n\tRLegForce\x18& \x01(\x05\x12\x11\n\tLLegForce\x18\' \x01(\x05\x12\x11\n\tRLegCalib\x18( \x01(\x05\x12\x11\n\tLLegCalib\x18) \x01(\x05\x12\x10\n\x08PatState\x18* \x01(\x05\x12\x0f\n\x07\x44ummy06\x18+ \x01(\x05\x12\x0f\n\x07\x44ummy07\x18, \x01(\x05\x12\x0f\n\x07\x44ummy08\x18- \x01(\x05\x12\x0f\n\x07\x44ummy09\x18. \x01(\x05\x12\x0f\n\x07\x44ummy10\x18/ \x01(\x05\x12\x0f\n\x07\x44ummy11\x18\x30 \x01(\x05\x12\x0f\n\x07\x44ummy12\x18\x31 \x01(\x05\x12\x0f\n\x07\x44ummy13\x18\x32 \x01(\x05\x12\x0f\n\x07\x44ummy14\x18\x33 \x01(\x05\x12\x0f\n\x07\x44ummy15\x18\x34 \x01(\x05\x12\x0f\n\x07\x44ummy16\x18\x35 \x01(\x05\x12\x0f\n\x07\x44ummy17\x18\x36 \x01(\x05\x12\x0f\n\x07\x44ummy18\x18\x37 \x01(\x05\x12\x0f\n\x07\x44ummy19\x18\x38 \x01(\x05\x12\x12\n\nDataLength\x18\x39 \x01(\x05\x62\x06proto3'
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
    _descriptor.FieldDescriptor(
      name='LogDetail', full_name='PSessionLogs.LogDetail', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_end=173,
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
      name='RS_Gyr', full_name='PSessionLogsDetail.RS_Gyr', index=3,
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
      name='LS_Gyr', full_name='PSessionLogsDetail.LS_Gyr', index=7,
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
      name='RT_Gyr', full_name='PSessionLogsDetail.RT_Gyr', index=11,
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
      name='LT_Gyr', full_name='PSessionLogsDetail.LT_Gyr', index=15,
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
      name='Tr_Gyr', full_name='PSessionLogsDetail.Tr_Gyr', index=19,
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
  serialized_start=176,
  serialized_end=1191,
)

_PSESSIONLOGS.fields_by_name['LogDetail'].message_type = _PSESSIONLOGSDETAIL
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
