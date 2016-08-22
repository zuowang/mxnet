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
    def init(stats_path,
              num_comm_channels_per_client,
              num_tables,
              num_total_clients,
              num_local_app_threads,
              aggressive_clock,
              aggressive_cpu,
              snapshot_clock,
              resume_clock,
              update_sort_policy,
              bg_idle_milli,
              client_bandwidth_mbps,
              server_bandwidth_mbps,
              thread_oplog_batch_size,
              row_candidate_factor,
              numa_opt,
              numa_index,
              numa_policy,
              naive_table_oplog_meta,
              suppression_on,
              use_approx_sort,
              num_zmq_threads,
              num_hosts,
              ids,
              hosts,
              ports,
              table_access):
        init_thread_id = ctypes.c_int()
        check_call(_LIB.MXPSInit(
            ctypes.c_char_p(stats_path),
            ctypes.c_int(num_comm_channels_per_client),
            ctypes.c_int(num_tables),
            ctypes.c_int(num_total_clients),
            ctypes.c_int(num_local_app_threads),
            ctypes.c_bool(aggressive_clock),
            ctypes.c_bool(aggressive_cpu),
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
    def create_table(table_id,
                       table_staleness,
                       row_type,
                       row_capacity,
                       oplog_dense_serialized,
                       row_oplog_type,
                       dense_row_oplog_capacity,
                       server_push_row_upper_bound,
                       server_table_logic,
                       version_maintain,
                       process_cache_capacity,
                       thread_cache_capacity,
                       oplog_capacity,
                       oplog_type,
                       append_only_oplog_type,
                       append_only_buff_capacity,
                       per_thread_append_only_buff_pool_size,
                       bg_apply_append_oplog_freq,
                       process_storage_type,
                       no_oplog_replay,
                       client_send_oplog_upper_bound):
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
    def get_table_or_die():
        handle = PSTableHandle()
        check_call(_LIB.MXPSGetTableOrDie(ctypes.byref(handle)))
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