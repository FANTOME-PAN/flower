mkdir logs/
chmod +x clients.sh server.sh
((./server.sh & sleep 1s); ./clients.sh)
