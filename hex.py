import os

key = os.urandom(32)
print("key: {key}")
print(key.hex())

nonce = os.urandom(16)
print("nonce: {nonce}")
print(nonce.hex())

message = "6c746865726faasdfsfhdjgea"
print("message: {message}")
print(message.encode().hex())
