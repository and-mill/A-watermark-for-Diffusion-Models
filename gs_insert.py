from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm
import os
from datetime import datetime

def gs_watermark_init_noise(opt, message=""):
    if message:
        # Convert the message to a byte string
        message_bytes = message.encode()
        # Ensure the encoded message is 256 bits (32 bytes)
        if len(message_bytes) < 32:
            padded_message = message_bytes + b'\x00' * (32 - len(message_bytes))
        else:
            padded_message = message_bytes[:32]
        k = padded_message
    else:
        # If the message is empty, generate a random 256-bit watermark message k
        k = os.urandom(32)
    
    # Diffusion process, replicate 64 copies
    s_d = k * 64
    
    # Use ChaCha20 for encryption
    # Default to using the key_hex and nonce_hex parameters passed in
    if opt.key_hex != "" and opt.nonce_hex != "":
        # Convert hexadecimal strings to byte strings
        key = bytes.fromhex(opt.key_hex)
        # Use the provided nonce_hex
        nonce = bytes.fromhex(opt.nonce_hex)
    # nonce_hex can be omitted, use the central 16 bytes of key_hex
    elif opt.key_hex != "" and opt.nonce_hex == "":
        # Convert hexadecimal strings to byte strings
        key = bytes.fromhex(opt.key_hex)
        # Use a fixed nonce
        nonce_hex = opt.key_hex[16:48]
        # Convert nonce_hex to bytes
        nonce = bytes.fromhex(nonce_hex)
    else:
        key = os.urandom(32)
        nonce = os.urandom(16)
    
    # Encrypt
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    m = encryptor.update(s_d) + encryptor.finalize()
    # Convert m to binary form; m follows a uniform distribution
    m_bits = ''.join(format(byte, '08b') for byte in m)
    # Initialize the results list to store the processing results for each window
    results = []
    # Window size could be 1, and also could be other value
    l = 1  

    index = 0
    Z_s_T_array = np.zeros((4, 64, 64))
    # Traverse the binary representation of m, cutting according to window size l
    for i in range(0, len(m_bits), l):
        window = m_bits[i:i + l]
        y = int(window, 2)  # Convert the binary sequence inside the window into integer y
        # Generate random u
        u = np.random.uniform(0, 1)
        # Calculate z^s_T
        z_s_T = norm.ppf((u + y) / 2**l)
        Z_s_T_array[index // (64 * 64), (index // 64) % 64, index % 64] = z_s_T
        index += 1

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'info_data.txt', 'a') as f:
        f.write(f"Time: {current_time}\n")
        f.write(f'key: {key.hex()}\n')
        f.write(f'nonce: {nonce.hex()}\n')
        f.write(f'message: {k.hex()}\n')
        f.write('----------------------\n')
    return Z_s_T_array