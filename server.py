import io
import sys
import socket

import matplotlib.pyplot as plt
from PIL import Image

import examine

def listen():
    addr = '127.0.0.1'
    port = 1255
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((addr, port))
        s.listen(32)
        print ("%s:%d listening" % (addr, port))
    except ConnectionRefusedError:
        print ("%s:%d - connection refused" % (addr, port))
        return

    while True:
        conn, cliaddr = s.accept()
        print("new connection from %s:%d" % (cliaddr[0], cliaddr[1]))
        with conn:
            try:
                remain = int.from_bytes(conn.recv(8), byteorder='big')
                print("image size %d" % remain)
                byte_arr = bytearray()
                while True:
                    if remain > 4096:
                        blob = conn.recv(4096)
                    else:
                        blob = conn.recv(remain)
                    if not blob: break
                    byte_arr.extend(blob)
                    remain -= len(blob)
                    if remain <= 0: break
                byte_f = io.BytesIO(byte_arr)
                img = examine.pil_to_cv(Image.open(byte_f))
                analyzed = examine.get_txt("", img, verbose=True)
                conn.send(bytearray(analyzed, encoding='utf-8'))
            except IOError as e:
                print(e)
                conn.send(bytearray("Invalid image file", encoding='utf-8'))
            except BaseException as e:
                print(e)

            print("connection ended")

if __name__ == "__main__":
    listen()
