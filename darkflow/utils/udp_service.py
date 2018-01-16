from socket import socket, AF_INET, SOCK_DGRAM
from threading import Thread


class UDPService(Thread):

    def __init__(self, address, port, queue):
        Thread.__init__(self)
        self.daemon = True
        self.queue = queue
        self.socket = socket(AF_INET, SOCK_DGRAM)
        self.address = address
        self.port = port
        self.start()

    def run(self):
        print("Binding UDP socket on " + str(self.address) + ":" + str(self.port))
        while True:
            textJSON = self.queue.get()
            self.socket.sendto(textJSON.encode(), (self.address, self.port))
