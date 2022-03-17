PYTHONUNBUFFERED=1 python3 server.py \
  --rounds=500 \
  --epochs=1 \
  --sample_fraction=1.0 \
  --min_sample_size=8 \
  --min_num_clients=8 \
  --server_address="localhost:24338"
