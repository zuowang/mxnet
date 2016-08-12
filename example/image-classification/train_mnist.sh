#!/bin/bash -u

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
example_dir=`dirname $script_dir`
app_dir=`dirname $example_dir`
progname=python
prog_path=/usr/bin/${progname}

host_filename="${app_dir}/machinefiles/localserver"
host_file=$(readlink -f $host_filename)

dataset=mnist

##=====================================
## Parameters
##=====================================

# Input files:
#solver_filename="${app_dir}/examples/mnist/lenet_solver.prototxt"
 # Uncomment this and line-90 if (re-)start training from a snapshot
#snapshot_filename="${app_dir}/examples/mnist/lenet_iter_100.solverstate"
#download to "${app_dir}/${dataset}"

# System parameters:
num_app_threads=1
num_table_threads=$(( num_app_threads + 1 ))
param_table_staleness=0
num_comm_channels_per_client=1
num_rows_per_table=1 #set to 1 for small NN
consistency_model="SSPPush"
svb=false

##=====================================

ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"

# Parse hostfile
host_list=`cat $host_file | awk '{ print $2 }'`
unique_host_list=`cat $host_file | awk '{ print $2 }' | uniq`
num_unique_hosts=`cat $host_file | awk '{ print $2 }' | uniq | wc -l`
devices="0"

output_dir=$app_dir/output
output_dir="${output_dir}/caffe.${dataset}.S${param_table_staleness}"
output_dir="${output_dir}.M${num_unique_hosts}"
output_dir="${output_dir}.T${num_app_threads}"
log_dir=$output_dir/logs
net_outputs_prefix="${output_dir}/${dataset}"

# Kill previous instances of this program
echo "Killing previous instances of '$progname' on servers, please wait..."
for ip in $unique_host_list; do
  ssh $ssh_options $ip \
    killall -q $progname
done
echo "All done!"

# Spawn program instances
client_id=0
for ip in $unique_host_list; do
  echo Running client $client_id on $ip
  log_path=${log_dir}.${client_id}

  cmd="'mkdir -p ${output_dir}; \
      mkdir -p ${log_path}; \
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64; \
      GLOG_logtostderr=false \
      GLOG_stderrthreshold=0 \
      GLOG_log_dir=$log_path \
      GLOG_v=-1 \
      GLOG_minloglevel=0 \
      GLOG_vmodule="" \
      $prog_path train_mnist.py \
      --consistency_model $consistency_model \
      --init_thread_access_table=true \
      --hostfile $host_file \
      --client_id ${client_id} \
      --num_clients $num_unique_hosts \
      --num_table_threads $num_table_threads \
      --table_staleness $param_table_staleness \
      --num_comm_channels_per_client $num_comm_channels_per_client \
      --num_rows_per_table $num_rows_per_table \
      --svb=$svb \
      --stats_path ${output_dir}/caffe_stats.yaml \
      --solver=${solver_filename} \
      --net_outputs=${net_outputs_prefix} \
      --gpu=${devices}'" #\
      #--snapshot=${snapshot_filename}'"

  ssh $ssh_options $ip bash -c $cmd &
  #eval $cmd  # Use this to run locally (on one machine).

  # Wait a few seconds for the name node (client 0) to set up
  if [ $client_id -eq 0 ]; then
    echo $cmd   # echo the cmd for just the first machine.
    echo "Waiting for name node to set up..."
    sleep 3
  fi
  client_id=$(( client_id+1 ))
done
