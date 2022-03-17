mkdir logs/
mkdir partitions/
chmod +x clients.sh server.sh
((./server.sh & sleep 8s); ./clients.sh)
