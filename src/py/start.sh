mkdir logs/
chmod +x clients.sh server.sh
((./server.sh & sleep 5s); ./clients.sh)