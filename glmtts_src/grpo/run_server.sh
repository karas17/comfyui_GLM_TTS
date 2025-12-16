for i in {0..7}; do
  LOCAL_RANK=$((i%2)) uvicorn reward_server:app --host 0.0.0.0 --port 808$((+i)) &
done