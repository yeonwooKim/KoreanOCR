import socket

addr = '127.0.0.1'
port = 1255

def send(byte_arr):
    txt = "Could not connect to analysis daemon"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.connect((addr, port))
        print ("%s:%d - connected" % (addr, port))
        s.send(len(byte_arr).to_bytes(8, byteorder='big'))
        s.send(byte_arr)
        print ("%s:%d - data sent, waiting for response.." % (addr, port))
        txt = recv_txt(s)
        s.close()
    except ConnectionRefusedError:
        print ("%s:%d - connection refused" % (addr, port))
    return txt

def recv_txt(conn):
    byte_arr = bytearray()
    while True:
        blob = conn.recv(4096)
        if not blob: break
        byte_arr.extend(blob)
    return byte_arr.decode(encoding='utf-8')

if __name__ == "__main__":
    print("running standalone")
    send(bytearray("running standalong", encoding='utf-8'))