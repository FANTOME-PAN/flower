PYTHONUNBUFFERED=1 python3 server.py \
  --rounds=10 \
  --epochs=1 \
  --sample_fraction=1.0 \
  --min_sample_size=4 \
  --min_num_clients=4 \
  --server_address="localhost:24338"
