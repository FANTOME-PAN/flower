export PYTHONUNBUFFERED=1
NUM_CLIENTS=4 # TODO: change the number of clients here

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    # Staggered loading of clients: clients are loaded 8s apart.
    # At the start, each client loads the entire CIFAR-10 dataset before selecting
    # their own partition. For a large number of clients this causes a memory usage
    # spike that can cause client processes to get terminated. 
    # Staggered loading prevents this.
    sleep 8s  
    python3 client.py \
      --cid=$i \
      --num_partitions=${NUM_CLIENTS} \
      --iid_fraction=1.0 \
      --server_address="localhost:24338" \
      --exp_name="federated_${NUM_CLIENTS}_clients" &
done
echo "Started $NUM_CLIENTS clients."