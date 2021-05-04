import select
import socket
import struct

server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server_sock.bind(("192.168.2.1", 7860))
server_sock.bind(("127.0.0.1", 7860))

# Listen for a connection.
server_sock.listen(1)
print(f"Listening for a connection on 192.168.2.1:7860...")

client_sock = None

while client_sock is None:
    timeout = 0.1
    readable, _, _ = select.select([server_sock], [], [], timeout)

    for s in readable:
        if s is server_sock:
            client_sock, client_endpoint = server_sock.accept()
            print(f"Accepted connection from client @ {client_endpoint}")

msg = ""
while msg != "exit":
    buf = client_sock.recv(struct.calcsize("<i"))
    length = struct.unpack("<i", buf)[0]
    print(length)
    buf = client_sock.recv(length)
    msg = buf.decode("utf-8")
    print(msg)

    tokens = msg.split(" ")
    if tokens[0] == "control":
        rate = int(tokens[1])
        if rate > 30:
            rate = 30
        if rate < -30:
            rate = -30
        print(rate)

server_sock.shutdown(socket.SHUT_RDWR)
server_sock.close()

client_sock.shutdown(socket.SHUT_RDWR)
client_sock.close()
