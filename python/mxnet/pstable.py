# coding: utf-8
""" Petuum PS interface of MXNet for parameter synchronization."""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import check_call, c_array
from .base import PSTableHandle, NDArrayHandle
from .ndarray import NDArray

def _ctype_key_value(keys, vals):
    """
    Return ctype arrays for the key-value args, for internal use
    """
    if isinstance(keys, int):
        if isinstance(vals, NDArray):
            return (c_array(ctypes.c_int, [keys]),
                    c_array(NDArrayHandle, [vals.handle]))
        else:
            for value in vals:
                assert(isinstance(value, NDArray))
            return (c_array(ctypes.c_int, [keys] * len(vals)),
                    c_array(NDArrayHandle, [value.handle for value in vals]))
    else:
        assert(len(keys) == len(vals))
        for k in keys:
            assert(isinstance(k, int))
        c_keys = []
        c_vals = []
        for i in range(len(keys)):
            c_key_i, c_val_i = _ctype_key_value(keys[i], vals[i])
            c_keys += c_key_i
            c_vals += c_val_i
        return (c_array(ctypes.c_int, c_keys), c_array(NDArrayHandle, c_vals))

class PSTable(object):
    @staticmethod
    def register_dense_row(dtype, row_type):
        check_call(_LIB.MXPSRegisterDenseRow(ctypes.c_int(dtype), ctypes.c_int(row_type)))

    @staticmethod
    def init(stats_path = "",
             num_comm_channels_per_client = 1,
             num_tables = 1,
             num_total_clients = 1,
             num_local_app_threads = 2,
             client_id = 0,
             aggressive_clock = False,
             aggressive_cpu = False,
             consistency_model = 1, #SSPPush
             snapshot_clock = -1,
             resume_clock = -1,
             update_sort_policy = 1,#Random
             bg_idle_milli = 2,
             client_bandwidth_mbps = 40,
             server_bandwidth_mbps = 40,
             thread_oplog_batch_size = 100*1000*1000,
             row_candidate_factor = 5,
             numa_opt = False,
             numa_index = 0,
             numa_policy = 0,#Even
             naive_table_oplog_meta = True,
             suppression_on = False,
             use_approx_sort = False,
             num_zmq_threads = 1,
             num_hosts = None,
             ids = None,
             hosts = None,
             ports = None,
             table_access = False):
        init_thread_id = ctypes.c_int()
        check_call(_LIB.MXPSInit(
            ctypes.c_char_p(stats_path),
            ctypes.c_int(num_comm_channels_per_client),
            ctypes.c_int(num_tables),
            ctypes.c_int(num_total_clients),
            ctypes.c_int(num_local_app_threads),
            ctypes.c_int(client_id),
            ctypes.c_bool(aggressive_clock),
            ctypes.c_bool(aggressive_cpu),
            ctypes.c_int(consistency_model),
            ctypes.c_int(snapshot_clock),
            ctypes.c_int(resume_clock),
            ctypes.c_int(update_sort_policy),
            ctypes.c_int(bg_idle_milli),
            ctypes.c_double(client_bandwidth_mbps),
            ctypes.c_double(server_bandwidth_mbps),
            ctypes.c_size_t(thread_oplog_batch_size),
            ctypes.c_long(row_candidate_factor),
            ctypes.c_bool(numa_opt),
            ctypes.c_int(numa_index),
            ctypes.c_int(numa_policy),
            ctypes.c_bool(naive_table_oplog_meta),
            ctypes.c_bool(suppression_on),
            ctypes.c_bool(use_approx_sort),
            ctypes.c_size_t(num_zmq_threads),
            ctypes.c_int(num_hosts),
            c_array(ctypes.c_int, ids),
            c_array(ctypes.c_char_p, hosts),
            c_array(ctypes.c_char_p, ports),
            ctypes.c_bool(table_access),
            ctypes.byref(init_thread_id)))
        return init_thread_id.value

    @staticmethod
    def create_table(table_id = 0,
                     table_staleness = 0,
                     row_type = 0,
                     row_capacity = 1,
                     oplog_dense_serialized = False,
                     row_oplog_type = 1,
                     dense_row_oplog_capacity = 0,
                     server_push_row_upper_bound = 100,
                     server_table_logic = -1,
                     version_maintain = False,
                     process_cache_capacity = 0,
                     thread_cache_capacity = 1,
                     oplog_capacity = 0,
                     oplog_type = 0,#Sparse
                     append_only_oplog_type = 0,#Inc
                     append_only_buff_capacity = 1024*1024,
                     per_thread_append_only_buff_pool_size = 3,
                     bg_apply_append_oplog_freq = 4,
                     process_storage_type = 1,#BoundedSparse
                     no_oplog_replay = False,
                     client_send_oplog_upper_bound = 100):
        ret = ctypes.c_bool()
        check_call(_LIB.MXPSCreateTable(
            ctypes.c_int(table_id),
            ctypes.c_int(table_staleness),
            ctypes.c_int(row_type),
            ctypes.c_size_t(row_capacity),
            ctypes.c_bool(oplog_dense_serialized),
            ctypes.c_int(row_oplog_type),
            ctypes.c_size_t(dense_row_oplog_capacity),
            ctypes.c_size_t(server_push_row_upper_bound),
            ctypes.c_int(server_table_logic),
            ctypes.c_bool(version_maintain),
            ctypes.c_size_t(process_cache_capacity),
            ctypes.c_size_t(thread_cache_capacity),
            ctypes.c_size_t(oplog_capacity),
            ctypes.c_int(oplog_type),
            ctypes.c_int(append_only_oplog_type),
            ctypes.c_size_t(append_only_buff_capacity),
            ctypes.c_size_t(per_thread_append_only_buff_pool_size),
            ctypes.c_int(bg_apply_append_oplog_freq),
            ctypes.c_int(process_storage_type),
            ctypes.c_bool(no_oplog_replay),
            ctypes.c_size_t(client_send_oplog_upper_bound),
            ctypes.byref(ret)))
        return ret.value

    @staticmethod
    def create_table_done():
        check_call(_LIB.MXPSCreateTableDone())

    @staticmethod
    def register_thread():
        ret = ctypes.c_int()
        check_call(_LIB.MXPSRegisterThread(ctypes.byref(ret)))
        return ret.value

    @staticmethod
    def get_table_or_die(table_id):
        handle = PSTableHandle()
        check_call(_LIB.MXPSGetTableOrDie(0, ctypes.c_int(table_id), ctypes.byref(handle)))
        return handle

    @staticmethod
    def deregister_thread():
        check_call(_LIB.MXPSDeregisterThread())

    @staticmethod
    def wait_thread_register():
        check_call(_LIB.MXPSWaitThreadRegister())

    @staticmethod
    def shut_down():
        check_call(_LIB.MXPSShutDown())

    @staticmethod
    def clock():
        check_call(_LIB.MXPSClock())

    @staticmethod
    def global_barrier():
        check_call(_LIB.MXPSGlobalBarrier())

    @staticmethod
    def global_barrier():
        check_call(_LIB.MXPSGlobalBarrier())

    @staticmethod
    def table_batch_inc(handle, idx, num, update):
        check_call(_LIB.MXPSTableBatchInc(handle,
                                          ctypes.c_int(idx),
                                          ctypes.c_int(num),
                                          c_array(ctypes.c_float, update)))

    @staticmethod
    def table_get(handle, idx, num):
        row = ctypes.POINTER(ctypes.c_float)()
        check_call(_LIB.MXPSTableGet(handle,
                                     ctypes.c_int(idx),
                                     ctypes.c_int(num),
                                     ctypes.byref(row)))
        out = [tuple(row[i]) for i in range(num)]
        return out

    @staticmethod
    def table_release(row):
        check_call(_LIB.MXPSTableRelease(c_array(ctypes.c_float, row)))

    @staticmethod
    def table_batch_incx(handle,
                         num_rows_per_table,
                         table_row_capacity,
                         key,
                         out):
        ckeys, cvals = _ctype_key_value(key, out)
        check_call(_LIB.MXPSTableBatchIncX(handle,
                                           ctypes.c_int(num_rows_per_table),
                                           ctypes.c_int(table_row_capacity),
                                           ctypes.c_int(len(ckeys)),
                                           ckeys,
                                           cvals))

    @staticmethod
    def table_getx(handle,
                   num_rows_per_table,
                   table_row_capacity,
                   clock,
                   key,
                   out):
        assert(out is not None)
        ckeys, cvals = _ctype_key_value(key, out)
        check_call(_LIB.MXPSTableGetX(handle,
                                      ctypes.c_int(num_rows_per_table),
                                      ctypes.c_int(table_row_capacity),
                                      ctypes.c_int(clock),
                                      ctypes.c_int(len(ckeys)),
                                      ckeys,
                                      cvals))