from socket import socket, AF_INET, SOCK_DGRAM
from threading import Thread


class UDPService(Thread):

    def __init__(self, address, port, queue):
        Thread.__init__(self)
        self.queue = queue
        self.socket = socket(AF_INET, SOCK_DGRAM)
        self.address = address
        self.port = port

    def run(self):
        print("Binding UDP socket on " + str(self.address) + ":" + str(self.port))
        self.socket.bind((self.address, self.port))
        self.socket.listen(1)
        self.conn, self.addr = self.socket.accept()

        while True:
            textJSON = self.queue.get()
            if self.addr is not None:
                self.conn.send(textJSON.encode())
